from typing import Tuple, Iterator, List, Dict

import numpy as np
from numpy.typing import NDArray
from refpydst.data_types import Turn
from sklearn.metrics.pairwise import cosine_similarity

from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever


class SamplingTopK(AbstractExampleListDecoder):
    """
    An example selection decoder which simply takes the 10 highest scoring examples for each retrieving method.

    Input:
    bm25_examples: # of training examples
    sbert_examples: # of k
    BM25: BM25 TF-IDF TOP K
    SBERT: SBERT cos-sim TOP K

    """
    from_n_possible: int
    discount_factor: float
    retriever: EmbeddingRetriever

    def __init__(self, retriever: EmbeddingRetriever,**kwargs) -> None:
        self.retriever = retriever

    def select_k(self, examples: Iterator[Tuple[Turn, float]],
                 bm25_examples:Iterator[Tuple[Turn, float]], k: Dict, **kwargs) -> List[Turn]:
        # Select up to "from N possible" in the iterator
        query_id = kwargs.get('query_id').split('_')[0]

        sbert_all_considered_examples: List[Tuple[Turn, float]] = \
            [turn_label for _, (turn_label, score) in zip(range(k.get('SBERT',0)*2), examples) if query_id not in turn_label]
        
        bm25_all_considered_examples: List[Tuple[Turn, float]] = \
            [turn_label for (turn_label, score) in bm25_examples if query_id not in turn_label]
        
        sbert_topk, ids = [], []
        idx = 0
        while len(sbert_topk) < k.get('SBERT', 0):
            turn_label = sbert_all_considered_examples[idx]
            if turn_label in ids:
                idx += 1
                continue
            sbert_topk.append(self.retriever.label_to_data_item(turn_label))
            ids.append(turn_label)
            idx += 1
        
        bm25_topk = bm25_all_considered_examples[:k.get('BM25', 0)+k.get('SBERT', 0)]
        bm25_topk = list(filter(lambda x: x if x not in ids else None, bm25_topk))[: k.get('BM25', 0)]
        bm25_topk = [self.retriever.label_to_data_item(turn_label) for turn_label in bm25_topk]

        # bm25_topk = []
        # k_bm25 = k.get('BM25', 0)
        # while k_bm25 > 0:
        #     turn_label = bm25_all_considered_examples.pop(0)
        #     if turn_label not in ids:
        #         bm25_topk.append(self.retriever.label_to_data_item(turn_label))
        #         k_bm25 -= 1
        
        return {'SBERT': sbert_topk[::-1], 'BM25': bm25_topk[::-1]}



# from collections import defaultdict
# from typing import Tuple, Iterator, List, Dict

# import numpy as np
# from numpy.typing import NDArray
# from refpydst.data_types import Turn
# from sklearn.metrics.pairwise import cosine_similarity

# from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
# from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever


# class SamplingTopK(AbstractExampleListDecoder):
#     """
#     An example selection decoder which simply takes the 10 highest scoring examples for each retrieving method.

#     Input:
#     bm25_examples: # of training examples
#     sbert_examples: # of k
#     BM25: BM25 TF-IDF TOP K
#     SBERT: SBERT cos-sim TOP K

#     """
#     from_n_possible: int
#     discount_factor: float
#     retriever: EmbeddingRetriever

#     def __init__(self, retriever: EmbeddingRetriever,**kwargs) -> None:
#         self.retriever = retriever
#         self.ground_key = kwargs.get('ground_key')

#     def select_k(self, examples: Iterator[Tuple[Turn, float]], k: Dict) -> List[Turn]:
#         example_by_key = defaultdict(list)
#         if self.ground_key is None:
#             self.ground_key, (data_item, score) = next(examples)
#             example_by_key[self.ground_key] = [data_item]
#         else:
#             example_by_key[self.ground_key] = []

#         for key_dict in examples:
#             for key, (data_item, score) in key_dict.items():
#                 example_by_key[key].append(data_item)
        
#         for key, example_list in example_by_key.items():    
#             if key != self.ground_key:
#                 tmp = []
#                 k_tmp = k.get(key, 10)
#                 while k_tmp > 0:
#                     turn = example_list.pop(0)
#                     if turn not in example_by_key[self.ground_key]:
#                         tmp.append(turn)
#                         k_tmp -= 1
#                 example_by_key[key] = tmp
        
#         for key in example_by_key:
#             example_by_key[key] = example_by_key[key][::-1]

#         return example_by_key
        
