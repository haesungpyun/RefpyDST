import random
from typing import Tuple, List, Iterator, Dict, Type, Callable, Union

import numpy as np
import torch
from numpy.typing import NDArray
from refpydst.data_types import Turn
from scipy.spatial import KDTree
from rank_bm25 import BM25Okapi


from refpydst.retriever.abstract_example_retriever import ExampleRetriever
from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
from refpydst.retriever.code.data_management import get_string_transformation_by_type, data_item_to_string
from refpydst.retriever.decoders.top_k import TopKDecoder

TurnLabel = str


class Retriever:
    label_to_idx: Dict[str, int]

    def normalize(self, emb):
        return emb
    
    def __init__(self, emb_dict):

        self.bm25 = BM25Okapi(corpus=list(emb_dict.values()))
        # to query faster, stack all search embeddings and record keys
        self.emb_keys: List[str] = list(emb_dict.keys())

    def iterate_nearest_dialogs(self, query_emb, k=5, selected_idx=None) -> Iterator[Tuple[str, float]]:
        query_emb = self.normalize(query_emb)
        i = 0
        scores = self.bm25.get_scores(query_emb)
        score_idx_dict = [[i,score] for i, score in enumerate(scores)]
        sorted_scores = sorted(score_idx_dict, key=lambda x: x[1], reverse=True)
        query_result = np.array([[i for (i, score) in sorted_scores]])
        sorted_scores = np.array([[score for (i, score) in sorted_scores]])
        while i < len(self.emb_keys):
            if query_result.shape == (1,):
                i += 1
                yield self.emb_keys[query_result.item()], sorted_scores.item()
                if i >= len(self.emb_keys):
                    break
            else:
                for item, score_item in zip(query_result.squeeze(0)[i:], sorted_scores.squeeze(0)[i:]):
                    i += 1
                    if item.item() >= len(self.emb_keys):
                        return  # stop iteration!
                    if selected_idx is not None and item.item() not in selected_idx:
                        continue                 
                    yield self.emb_keys[item.item()], score_item.item()
                if i >= len(self.emb_keys):
                    break

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
    

class TwoStepRetriever(ExampleRetriever):

    def __init__(self, datasets, sampling_method="none", ratio=1.0,
                 full_history=False, retriever_type: Type[Retriever] = Retriever,
                 string_transformation: Union[str, Callable[[Turn], str]] = None, **kwargs):

        # data_items: list of datasets in this notebook. Please include datasets for both search and query
        # sampling method: "random_by_turn", "random_by_dialog", "kmeans_cosine", "pre_assigned"
        # ratio: how much portion is selected

        # embedding : tokenized text
        # string: [CONTEXT]~ [SYS]~ [USER]~
        if kwargs.get('step_1_kwargs'):
            step_1_kwargs = kwargs.get('step_1_kwargs')
        else:
            step_1_kwargs = {}
            input_type = kwargs.get('input_type', 'dialog')
            only_slot = kwargs.get('only_slot', False)
            step_1_kwargs.update({
                'full_history': full_history, 
                'input_type': input_type, 
                'only_slot': only_slot, 
            })

        if kwargs.get('step_2_kwargs'):
            step_2_kwargs = kwargs.get('step_2_kwargs')
        else:    
            step_2_kwargs = {}
            input_type = kwargs.get('input_type', 'gt_delta_slot')
            only_slot = kwargs.get('only_slot', False)
            step_2_kwargs.update({
                'full_history': full_history, 
                'input_type': input_type, 
                'only_slot': only_slot
            })
        
        def step_1_transformation(turn):
            return data_item_to_string(turn, **step_1_kwargs)
        
        def step_2_transformation(turn):
            return data_item_to_string(turn, **step_2_kwargs)

    
        if type(string_transformation) == str:
            # configs can also specify known functions by a string, e.g. 'default'
            self.string_transformation = get_string_transformation_by_type(string_transformation)
        else:
            self.step_1_transformation = string_transformation or step_1_transformation
            self.step_2_transformation = string_transformation or step_2_transformation

        self.data_items = []
        for dataset in datasets:
            self.data_items += dataset
        
        self.step_1_history = [self.step_1_transformation(turn) for turn in self.data_items]
        self.step_2_history = [self.step_2_transformation(turn) for turn in self.data_items]
        
        self.step_1_embedding = [self.step_1_transformation(turn) for turn in self.data_items]
        self.step_2_embedding = [self.step_2_transformation(turn) for turn in self.data_items]

        self.step_1_search_string, self.step_2_search_string = {}, {}
        self.step_1_search_embeddings, self.step_2_search_embeddings = {}, {}
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        for idx, turn in enumerate(self.data_items):
            id_turn_label = self.data_item_to_label(turn)
            self.step_1_search_string.update({id_turn_label: self.step_1_history[idx]})
            self.step_1_search_embeddings.update({id_turn_label: self.step_1_embedding[idx]})

            self.step_2_search_string.update({id_turn_label: self.step_2_history[idx]})
            self.step_2_search_embeddings.update({id_turn_label: self.step_2_embedding[idx]})

            self.label_to_idx.update({id_turn_label: idx})
            self.idx_to_label.update({idx: id_turn_label})

        self.step_1_retriever = retriever_type(self.step_1_search_embeddings)
        self.step_2_retriever = retriever_type(self.step_2_search_embeddings)
        
    def data_item_to_embedding(self, data_item) -> NDArray:
        # embedding = tokenized string
        if isinstance(data_item, list):
            return data_item
        string_query = self.string_transformation(data_item)
        embed = string_query.split()
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

    def item_to_best_examples(self, data_item, k=10, decoder: AbstractExampleListDecoder = TopKDecoder()):
        # the nearest neighbor is at the end
        step_1_query = self.step_1_transformation(data_item)
        step_1_result = [(turn_label, score) 
            for turn_label, score in self.step_1_retriever.iterate_nearest_dialogs(step_1_query, k=k)]
        
        step_1_idx = [self.label_to_idx[turn_label] for (turn_label, _) in step_1_result][:2*k]

        step_2_query = self.step_2_transformation(data_item)
        step_2_result = ((self.label_to_data_item(turn_label), score) 
            for turn_label, score in self.step_2_retriever.iterate_nearest_dialogs(step_2_query, k=k, selected_idx=step_1_idx))

        try:
            return decoder.select_k(k=k, examples=step_2_result)
        except StopIteration as e:
            print("ran out of examples! unable to decode")
            raise e
    
    def label_to_search_embedding(self, label: TurnLabel) -> NDArray:
        """
        # embedding = tokenized string
        For a known search turn (e.g. a retrieved example), get its embedding from it's label

        :param label: the string label used to identify turns when instantiating the retriever (emb_dict keys)
        :return: that initialized value, which is the search embedding
        """
        if label not in self.label_to_idx:
            raise KeyError(f"{label} not in search index")
        return self.search_embeddings[label]

    def get_scores_for_query(self, query:List, all_considered_examples:List[Tuple[Turn, float]]):
        if isinstance(all_considered_examples[0], tuple):
            example_idx = [self.label_to_idx[self.data_item_to_label(data_item)] for data_item, score in all_considered_examples]
            scores = self.retriever.bm25.get_scores(query)
            exmaple_score = [scores[idx] for idx in example_idx]
            return np.array(exmaple_score)
        elif isinstance(all_considered_examples[0], list):
            bm25 = BM25Okapi(corpus=all_considered_examples)
            scores = bm25.get_scores(query)  
            return np.array(scores)
        
