import random
from typing import Tuple, List, Iterator, Dict, Type, Callable, Union

import numpy as np
import torch
from numpy.typing import NDArray
from refpydst.data_types import Turn
from scipy.spatial import KDTree
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from refpydst.retriever.abstract_example_retriever import ExampleRetriever
from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
from refpydst.retriever.code.data_management import get_string_transformation_by_type, data_item_to_string
from refpydst.retriever.decoders.top_k import TopKDecoder


TurnLabel = str


class Retriever:
    label_to_idx: Dict[str, int]

    def normalize(self, emb):
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)
        
    def __init__(self, emb_dict, bm25_emb_dict):

        # to query faster, stack all search embeddings and record keys
        self.emb_keys: List[str] = list(emb_dict.keys())
        self.label_to_idx = {k: i for i, k in enumerate(self.emb_keys)}
        emb_dim = emb_dict[self.emb_keys[0]].shape[-1]

        self.emb_values = np.zeros((len(self.emb_keys), emb_dim))
        for i, k in enumerate(self.emb_keys):
            self.emb_values[i] = emb_dict[k]

        # normalize for cosine distance (kdtree only support euclidean when p=2)
        self.emb_values = self.normalize(self.emb_values)
        self.kdtree = KDTree(self.emb_values)

        self.bm25 = BM25Okapi(corpus=list(bm25_emb_dict.values()))
    
    def bm25_iterate_nearest_dialogs(self, query, k=5) -> Iterator[Tuple[str, float]]:
        tokenized_query = query
        i = 0
        fetch_size: int = k
        scores = self.bm25.get_scores(tokenized_query)
        idx_score_dict = {i:score for i, score in enumerate(scores)}
        sorted_dict = np.array(sorted(idx_score_dict.items(),key=lambda x: x[1], reverse=True))
        sorted_scores = np.array([sorted_dict[:,1]])
        query_result = np.array([sorted_dict[:,0]],dtype=np.int64)
        while i < len(self.emb_keys):
            if query_result.shape == (1,):
                i += 1
                yield self.emb_keys[query_result.item()], sorted_scores.item()
            else:
                for item, score_item in zip(query_result.squeeze(0)[i:], sorted_scores.squeeze(0)[i:]):
                    i += 1
                    if item.item() >= len(self.emb_keys):
                        return  # stop iteration!
                    yield self.emb_keys[item.item()], score_item.item()
            fetch_size = min(2 * fetch_size, len(self.emb_keys))
    
    def iterate_nearest_dialogs(self, query_emb, k=5) -> Iterator[Tuple[str, float]]:
        query_emb = self.normalize(query_emb)
        i = 0
        fetch_size: int = k
        while i < len(self.emb_keys):
            scores, query_result = self.kdtree.query(query_emb, k=fetch_size, p=2)
            if query_result.shape == (1,):
                i += 1
                yield self.emb_keys[query_result.item()], scores.item()
            else:
                for item, score_item in zip(query_result.squeeze(0)[i:], scores.squeeze(0)[i:]):
                    i += 1
                    if item.item() >= len(self.emb_keys):
                        return  # stop iteration!
                    yield self.emb_keys[item.item()], score_item.item()
            fetch_size = min(2 * fetch_size, len(self.emb_keys))

    def topk_nearest_dialogs(self, query_emb, k=5):
        query_emb = self.normalize(query_emb)
        scores = self.bm25.get_scores(query_emb)
        score_idx_dict = {score:i for i, score in enumerate(scores)}
        sorted_scores = sorted(score_idx_dict, reverse=True)
        query_result = np.array([[score_idx_dict[top_score] 
                                    for top_score in sorted_scores[:k]]])
        sorted_scores = np.array([sorted_scores[:k]])
        if k == 1:
            return [self.emb_keys[i] for i in query_result[1]]
        return [self.emb_keys[i] for i in query_result[1][0]]

    def topk_nearest_distinct_dialogs(self, query_emb, k=5):
        return self.topk_nearest_dialogs(query_emb, k=k)

    def random_retrieve(self, k=5):
        return random.sample(self.emb_keys, k)


class MixedRetriever(ExampleRetriever):

    # sample selection
    def random_sample_selection_by_turn(self, embs, ratio=0.1):
        n_selected = int(ratio * len(embs))
        print(f"randomly select {ratio} of turns, i.e. {n_selected} turns")
        selected_keys = random.sample(list(embs), n_selected)
        return {k: v for k, v in embs.items() if k in selected_keys}

    def random_sample_selection_by_dialog(self, embs, ratio=0.1):
        dial_ids = set([turn_label.split('_')[0] for turn_label in embs.keys()])
        n_selected = int(len(dial_ids) * ratio)
        print(f"randomly select {ratio} of dialogs, i.e. {n_selected} dialogs")
        selected_dial_ids = random.sample(dial_ids, n_selected)
        return {k: v for k, v in embs.items() if k.split('_')[0] in selected_dial_ids}

    def pre_assigned_sample_selection(self, embs, examples):
        selected_dial_ids = set([dial['ID'] for dial in examples])
        return {k: v for k, v in embs.items() if k.split('_')[0] in selected_dial_ids}

    def __init__(self, datasets, model_path, search_index_filename:str = None, sampling_method="none", 
                 ratio=1.0,  model=None, search_embeddings=None, full_history=False, 
                 retriever_type: Type[Retriever] = Retriever, 
                 string_transformation: Union[str, Callable[[Turn], str]] = None, **kwargs):

        # data_items: list of datasets in this notebook. Please include datasets for both search and query
        # sampling method: "random_by_turn", "random_by_dialog", "kmeans_cosine", "pre_assigned"
        # ratio: how much portion is selected

        # embedding : tokenized text
        # string: [CONTEXT]~ [SYS]~ [USER]~

        def default_transformation(turn):
            return data_item_to_string(turn, full_history=full_history)

        if type(string_transformation) == str:
            # configs can also specify known functions by a string, e.g. 'default'
            self.string_transformation = get_string_transformation_by_type(string_transformation)
        else:
            self.string_transformation = string_transformation or default_transformation

        self.data_items = []
        for dataset in datasets:
            self.data_items += dataset

        self.model = model

        if model is None:
            self.model = SentenceTransformer(model_path)
        
        # load the search index embeddings
        if search_embeddings is not None:
            self.search_embeddings = search_embeddings
        elif search_index_filename:
            self.search_embeddings = np.load(search_index_filename, allow_pickle=True).item()
        else:
            raise ValueError("unable to instantiate a retreiver without embeddings. Supply pre-loaded search_embeddings"
                             " or a search_index_filename")
        
        # sample selection of search index
        if sampling_method == "none":
            emb_dict = self.search_embeddings
        elif sampling_method == 'random_by_dialog':
            emb_dict = self.random_sample_selection_by_dialog(self.search_embeddings, ratio=ratio)
        elif sampling_method == 'random_by_turn':
            emb_dict = self.random_sample_selection_by_turn(self.search_embeddings, ratio=ratio)
        elif sampling_method == 'pre_assigned':
            emb_dict = self.pre_assigned_sample_selection(self.search_embeddings, self.data_items)
        else:
            raise ValueError("selection method not supported")
        
        self.bm25_embedding = list(map(self.data_item_to_bm25_embedding, self.data_items))

        self.bm25_emb_dict = {}
        for idx, turn in enumerate(self.data_items):
            id_turn_label = self.data_item_to_label(turn)
            self.bm25_emb_dict.update({id_turn_label: self.bm25_embedding[idx]})

        self.retriever = retriever_type(emb_dict, self.bm25_emb_dict)
        
    def data_item_to_bm25_embedding(self, data_item):
        # embedding = tokenized string
        if isinstance(data_item, list):
            return data_item
        string_query = self.string_transformation(data_item)
        embed = string_query.split()
        return embed

    def data_item_to_embedding(self, data_item):
        with torch.no_grad():
            embed = self.model.encode(self.string_transformation(
                data_item), convert_to_numpy=True).reshape(1, -1)
        return embed

    def data_item_to_label(self, turn: Turn) -> TurnLabel:
        return f"{turn['ID']}_turn_{turn['turn_id']}"

    def label_to_data_item(self, label: TurnLabel) -> Turn:
        ID, _, turn_id = label.split('_')
        turn_id = int(turn_id)

        for d in self.data_items:
            if d['ID'] == ID and d['turn_id'] == turn_id:
                return d
        raise ValueError(f"label {label} not found. check data items input")

    def item_to_best_examples(self, data_item, k, decoder: AbstractExampleListDecoder = TopKDecoder()):
        # the nearest neighbor is at the end
        query = self.data_item_to_embedding(data_item)
        bm25_query = self.data_item_to_bm25_embedding(data_item)
        if isinstance(k, dict):
            k_sbert = k.get('SBERT', 0)
            k_bm25 = k.get('BM25', 0)
        else:
            k_sbert, k_bm25 = k, k
        try:
            example_generator = (
                (turn_label, score) 
                for turn_label, score in self.retriever.iterate_nearest_dialogs(query, k=k_sbert))
            bm25_example_generator = (
                (turn_label, score) 
                for turn_label, score in self.retriever.bm25_iterate_nearest_dialogs(bm25_query, k=k_bm25))
            return decoder.select_k(k=k, examples=example_generator, bm25_examples=bm25_example_generator, query_id=self.data_item_to_label(data_item))
        except StopIteration as e:
            print("ran out of examples! unable to decode")
            raise e
    
    def label_to_bm25_search_embedding(self, label: TurnLabel) -> NDArray:
        """
        # embedding = tokenized string
        For a known search turn (e.g. a retrieved example), get its embedding from it's label

        :param label: the string label used to identify turns when instantiating the retriever (emb_dict keys)
        :return: that initialized value, which is the search embedding
        """
        if label not in self.retriever.label_to_idx:
            raise KeyError(f"{label} not in search index")
        return self.bm25_emb_dict[label]
    
    def label_to_search_embedding(self, label: TurnLabel) -> NDArray:
        """
        For a known search turn (e.g. a retrieved example), get its embedding from it's label

        :param label: the string label used to identify turns when instantiating the retriever (emb_dict keys)
        :return: that initialized value, which is the search embedding
        """
        if label not in self.retriever.label_to_idx:
            raise KeyError(f"{label} not in search index")
        return self.retriever.emb_values[self.retriever.label_to_idx[label]]

    def get_bm25_scores_for_query(self, query:List, all_considered_examples:List[Tuple[Turn, float]]):
        if isinstance(all_considered_examples[0], tuple):
            example_idx = [self.retriever.label_to_idx[turn_label] for turn_label, score in all_considered_examples]
            scores = self.retriever.bm25.get_scores(query)
            exmaple_score = [scores[idx] for idx in example_idx]
            return np.array(exmaple_score)
        elif isinstance(all_considered_examples[0], list):
            bm25 = BM25Okapi(corpus=all_considered_examples)
            scores = bm25.get_scores(query)  
            return np.array(scores)
        

