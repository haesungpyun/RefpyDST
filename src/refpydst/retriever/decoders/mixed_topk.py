from typing import Tuple, Iterator, List, Dict

import numpy as np
from numpy.typing import NDArray
from refpydst.data_types import Turn
from sklearn.metrics.pairwise import cosine_similarity

from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever


class MixedTopK(AbstractExampleListDecoder):
    """
    An example selection decoder which simply takes the 10 highest scoring examples.
    """
    from_n_possible: int
    discount_factor: float
    retriever: EmbeddingRetriever

    def __init__(self, retriever: EmbeddingRetriever, **kwargs) -> None:
        self.retriever = retriever
        self.operation = kwargs.get('operation', 'sum')
        self.zscore = kwargs.get('zscore', False)
    
    def select_k(self, examples: Iterator[Tuple[Turn, float]],
                 bm25_examples:Iterator[Tuple[Turn, float]], k: int) -> List[Turn]:
        # Select up to "from N possible" in the iterator
        all_considered_examples: List[Tuple[Turn, float]] = [turn_label_and_score for turn_label_and_score in examples]
        bm25_all_considered_examples: Dict[Tuple[Turn, float]] = {turn_label: score for turn_label, score in bm25_examples}

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

        if self.operation == 'sum':
            example_scores: NDArray = {
            turn_label: score + bm25_all_considered_examples[turn_label] 
            for _, (turn_label, score) in enumerate(all_considered_examples)}
        elif self.operation == 'multiply':
            example_scores: NDArray = {
            turn_label: score * bm25_all_considered_examples[turn_label] 
            for _, (turn_label, score) in enumerate(all_considered_examples)}
        
        example_scores = dict(sorted(example_scores.items(), key=lambda item: item[1], reverse=True))
        assert np.all(np.diff(list(example_scores.values())) <= 0)  # verifies they are decreasing as expected

        return [self.retriever.label_to_data_item(turn_label) 
                for turn_label in list(example_scores.keys())[:k]][::-1]

    def standardize(self, emb):
        emb = (emb-np.mean(emb))/np.std(emb)
        return 1 / (1+ np.exp(-emb))
            # tf-idf => 35-4 
            # cos sim => -1 1
            # cos sim -> z-score
            # tf-idf -> z-score