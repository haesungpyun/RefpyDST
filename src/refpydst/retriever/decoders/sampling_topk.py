from typing import Tuple, Iterator, List, Dict

import numpy as np
from numpy.typing import NDArray
from refpydst.data_types import Turn
from sklearn.metrics.pairwise import cosine_similarity

from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever


class SamplingTopK(AbstractExampleListDecoder):
    """
    An example selection decoder which simply takes the 10 highest scoring examples.
    """
    from_n_possible: int
    discount_factor: float
    retriever: EmbeddingRetriever

    def __init__(self, retriever: EmbeddingRetriever,**kwargs) -> None:
        self.retriever = retriever

    def select_k(self, examples: Iterator[Tuple[Turn, float]],
                 bm25_examples:Iterator[Tuple[Turn, float]], k: Dict) -> List[Turn]:
        # Select up to "from N possible" in the iterator
        sbert_all_considered_examples: List[Tuple[Turn, float]] = \
            [turn_label for _, (turn_label, score) in zip(range(k.get('SBERT',0)), examples)]
        
        bm25_all_considered_examples: List[Tuple[Turn, float]] = \
            [turn_label for (turn_label, score) in bm25_examples]
        
        sbert_topk, ids = [], []
        for turn_label in sbert_all_considered_examples:
            sbert_topk.append(self.retriever.label_to_data_item(turn_label))
            ids.append(turn_label)
        
        bm25_topk = []
        k_bm25 = k.get('BM25', 0)
        while k_bm25 > 0:
            turn_label = bm25_all_considered_examples.pop(0)
            if turn_label not in ids:
                bm25_topk.append(self.retriever.label_to_data_item(turn_label))
                k_bm25 -= 1
        
        return {'SBERT': sbert_topk[::-1], 'BM25': bm25_topk[::-1]}
