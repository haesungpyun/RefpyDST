from typing import Tuple, Iterator, List, Dict

import numpy as np
from numpy.typing import NDArray
from refpydst.data_types import Turn
from sklearn.metrics.pairwise import cosine_similarity

from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever


class MixedMaxEmbDistinct(AbstractExampleListDecoder):
    """
    An example selection decoder which simply takes the 10 highest scoring examples.
    """
    from_n_possible: int
    discount_factor: float
    retriever: EmbeddingRetriever

    def __init__(self, retriever: EmbeddingRetriever, from_n_possible: int, discount_factor: float = 0.1, **kwargs) -> None:
        self.from_n_possible = from_n_possible
        self.discount_factor = discount_factor
        self.retriever = retriever
        self.operation = kwargs.get('operation', 'sum')
        self.zscore = kwargs.get('zscore', False)

    def select_k(self, examples: Iterator[Tuple[Turn, float]],
                 bm25_examples:Iterator[Tuple[Turn, float]], k: int) -> List[Turn]:
        # Select up to "from N possible" in the iterator
        all_considered_examples: List[Tuple[Turn, float]] = [turn_label_and_score for turn_label_and_score in examples]
        bm25_all_considered_examples: Dict[Tuple[Turn, float]] = {turn_label: score for turn_label, score in bm25_examples}

        if len(all_considered_examples) == 0:
            return []
        
        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result: List[int] = []

        # scores from retriever are euclidean distances, of unit vectors (range 0-2). cos(y, z) = 1-.5*euc(y, z)^2
        # We initialize example_scores with the cosine similarity of each example e to x, the input turn (implicit from
        #  order of argument examples to select_k).
        score = np.array([(1-0.5*(score**2)) for _, score in all_considered_examples])
        bm25_score = np.array(list(bm25_all_considered_examples.values()))

        if self.zscore:
            score = self.standardize(score)
            bm25_score = self.standardize(bm25_score)
        else:
            bm25_score = self.standardize(bm25_score)

        all_considered_examples = [(turn_label, score) for (turn_label, _), score in zip(all_considered_examples, score)]        
        bm25_all_considered_examples = {turn_label: score for turn_label, score in zip(bm25_all_considered_examples.keys(), bm25_score)}

        example_scores = []
        for idx, (turn_label, score) in enumerate(all_considered_examples):
            bm_score = bm25_all_considered_examples[turn_label]
            example_scores += [(turn_label, score+bm_score)] if self.operation == 'sum' else [(turn_label, score*bm_score)]
        
        example_scores = sorted(example_scores, key=lambda x: x[1], reverse=True)

        # shape: (example_idx, embedding size)
        all_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(turn_label)
            for (turn_label, _) in example_scores
        ])
        bm25_all_embeddings: NDArray = {
            turn_label: self.retriever.label_to_bm25_search_embedding(turn_label)
            for turn_label, _ in bm25_all_considered_examples.items()
        }

        tmp = []
        idx_label_dict = {}
        for idx, (turn_label, score) in enumerate(example_scores):
            idx_label_dict.update({idx: turn_label})
            tmp.append(score)
        example_scores = np.array(tmp)
        assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected

        while len(result) < k:
            # find the current best scoring example
            best_idx: int = np.argmax(example_scores).item()
            example_scores[best_idx] = -np.inf
            result.append(best_idx)
            # Update the scores. The worst-case decrease in score is defined by discount_factor.
            best_emb: NDArray = all_embeddings[best_idx]
            best_bm25_emb:NDArray = bm25_all_embeddings[idx_label_dict[best_idx]]
            
            discount = self.standardize(self.retriever.get_bm25_scores_for_query(best_bm25_emb, all_considered_examples))
            if self.operation == "sum":
                discount += self.standardize(cosine_similarity(best_emb[None, :], all_embeddings).squeeze(0))
            else:
                discount *= self.standardize(cosine_similarity(best_emb[None, :], all_embeddings).squeeze(0))
            discount *= self.discount_factor 
            example_scores = example_scores - discount

        return [self.retriever.label_to_data_item(idx_label_dict[i]) for i in result][::-1]
        
    def standardize(self, emb):
        emb = (emb-np.mean(emb))/np.std(emb)
        return 1 / (1+ np.exp(-emb))
            # tf-idf => 35-4 
            # cos sim => -1 1
            # cos sim -> z-score
            # tf-idf -> z-score
