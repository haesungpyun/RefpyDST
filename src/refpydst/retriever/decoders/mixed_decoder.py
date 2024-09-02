from typing import Tuple, Iterator, List, Dict

import numpy as np
from numpy.typing import NDArray
from refpydst.data_types import Turn
from sklearn.metrics.pairwise import cosine_similarity

from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever


class MixedDecoder(AbstractExampleListDecoder):
    """
    All possible combinations of BM25 and SBERT decoders.
    """
    from_n_possible: int
    discount_factor: float
    retriever: EmbeddingRetriever

    def __init__(self, retriever: EmbeddingRetriever, from_n_possible: int, discount_factor: float = 0.1, **kwargs) -> None:
        self.from_n_possible = from_n_possible
        self.discount_factor = discount_factor
        self.retriever = retriever
        self.operation = kwargs.get('operation', 'multiply')
        self.zscore = kwargs.get('zscore', False)
        self.decoding_logic = kwargs.get('decoding_logit', 'top_k_round_robin')

    def select_k(self, examples: Iterator[Tuple[Turn, float]],
                 bm25_examples:Iterator[Tuple[Turn, float]], k: int) -> List[Turn]:
        # Select up to "from N possible" in the iterator
        sbert_examples: List[Tuple[Turn, float]] = [turn_label_and_score for turn_label_and_score in examples]
        bm25_examples: Dict[Tuple[Turn, float]] = {turn_label: score for turn_label, score in bm25_examples}
        
        # scores from retriever are euclidean distances, of unit vectors (range 0-2). cos(y, z) = 1-.5*euc(y, z)^2
        # We initialize example_scores with the cosine similarity of each example e to x, the input turn (implicit from
        #  order of argument examples to select_k).
        cos_score = np.array([(1-0.5*(score**2)) for _, score in sbert_examples])
        bm25_score = np.array(list(bm25_examples.values()))

        if len(sbert_examples) == 0:
            return []
        
        if self.zscore:
            cos_score = self.standardize(cos_score)
            bm25_score = self.standardize(bm25_score)
        else:
            bm25_score = self.standardize(bm25_score)

        sbert_examples = [(turn_label, score) for (turn_label, _), score in zip(sbert_examples, cos_score)]        
        bm25_examples = {turn_label: score for turn_label, score in zip(bm25_examples.keys(), bm25_score)}

        if self.decoding_logit == 'top_k_round_robin':
            return self.top_k_round_robin(sbert_examples, bm25_examples, k)
        elif self.decoding_logit == 'multiply_top_k':
            return self.multiply_top_k(sbert_examples, bm25_examples, k)
        elif self.decoding_logit == 'round_robin_div_top_k':
            return self.round_robin_div_top_k(sbert_examples, bm25_examples, k)
        elif self.decoding_logit == 'multiply_div_top_k':
            return self.multiply_div_top_k(sbert_examples, bm25_examples, k)
        elif self.decoding_logit == 'div_topk_round_robin':
            return self.div_topk_round_robin(sbert_examples, bm25_examples, k)
        else:
            raise ValueError(f"Unknown decoding logit: {self.decoding_logit}")

    
    def top_k_round_robin(
        self, 
        sbert_examples: List[Tuple[Turn, float]], 
        bm25_examples: List[Tuple[Turn, float]], 
        k: int
    ) -> List[Turn]:
        """
        Selects the top k examples from the round-robin of BM25 and SBERT examples.        
        bm25 -> top k/2  \
                            -> Union
        sbert -> top k/2 / 
        """
        bm25_examples = [turn_label for turn_label, _ in bm25_examples]
        sbert_examples = [turn_label for turn_label, _ in sbert_examples]

        top_bm25 = bm25_examples[:k]
        top_sbert = sbert_examples[:k]

        # Round-robin of the two lists
        result = []
        for i in range(k):
            result.append(top_sbert[i])
            result.append(top_bm25[i])

        return result[::-1] # [low to high]
    
    def multiply_top_k(
        self, 
        sbert_examples: List[Tuple[Turn, float]], 
        bm25_examples: List[Tuple[Turn, float]], 
        k: int
    ) -> List[Turn]:
        """
        Selects the top k examples from the round-robin of BM25 and SBERT examples.
        
        bm25 -> z-score  \
                           -> multiply -> top K
        sbert -> z-score / 
        
        """
        total_examples = {
            turn_label: score * bm25_examples[turn_label] 
            for _, (turn_label, score) in enumerate(sbert_examples)}

        total_examples = dict(sorted(total_examples.items(), key=lambda item: item[1], reverse=True)) # [higt to low]
        assert np.all(np.diff(list(total_examples.values())) <= 0)  # verifies they are decreasing as expected
        
        result = [self.retriever.label_to_data_item(turn_label) for turn_label in list(total_examples.keys())[:k]]  # top k

        return result[::-1] # [low to high]


    def round_robin_div_top_k(
        self, 
        sbert_examples: List[Tuple[Turn, float]], 
        bm25_examples: List[Tuple[Turn, float]], 
        k: int
    ) -> List[Turn]:
        """
        bm25 -> z-score  \
                          -> Union -> diversity -> top K
        sbert -> z-score / 
        """
        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result : List[int] = []
        bm25_examples = [(turn_label, score) for turn_label, score in bm25_examples.items()]
        
        # shape: (example_idx, embedding size)
        sbert_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(turn_label) for (turn_label, _) in example_scores
        ])
        bm25_embeddings = [
            self.retriever.label_to_bm25_search_embedding(turn_label) for turn_label, _ in bm25_examples
        ]
        
        example_scores_embedding = []
        for idx in range(k):
            example_scores_embedding += [(sbert_examples[idx][0], sbert_examples[idx][1], sbert_embeddings[idx])]
            example_scores_embedding += [(bm25_examples[idx][0], bm25_examples[idx][1], bm25_embeddings[idx])]
        
        example_scores_embedding = sorted(example_scores_embedding, key=lambda x: x[1], reverse=True) # [high to low]

        example_scores = []
        idx_label_dict = {}
        embeddings = []
        for idx, (turn_label, score, emb) in enumerate(example_scores_embedding):
            idx_label_dict.update({idx: turn_label})
            example_scores.append(score)
            embeddings.append(emb)
        
        example_scores = np.array(example_scores)
        assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected

        while len(result) < k:
            # find the current best scoring example
            best_idx: int = np.argmax(example_scores).item()
            example_scores[best_idx] = -np.inf
            result.append(best_idx)
            # Update the scores. The worst-case decrease in score is defined by discount_factor.
            embedding = embeddings[best_idx]
            
            if isinstance(embedding, np.ndarray):
                discount = self.discount_factor * cosine_similarity(embedding[None, :], embeddings).squeeze(0)
            else:
                discount = self.discount_factor * self.retriever.get_bm25_scores_for_query(embedding, bm25_embeddings)
            
            example_scores = example_scores - discount

        return [self.retriever.label_to_data_item(idx_label_dict[i]) for i in result][::-1] # [low to high]

    def multiply_div_top_k(
        self, 
        sbert_examples: List[Tuple[Turn, float]], 
        bm25_examples: List[Tuple[Turn, float]], 
        k: int
    ) -> List[Turn]:
        """
        bm25 -> z-score  \
                          -> multiply -> diversity -> top K
        sbert -> z-score / 
        """

        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result : List[int] = []

        example_scores = []
        for idx, (turn_label, cos_score) in enumerate(sbert_examples):
            bm_score = bm25_examples[turn_label]
            example_scores += [(turn_label, cos_score+bm_score)] if self.operation == 'sum' else [(turn_label, cos_score*bm_score)]
        
        example_scores = sorted(example_scores, key=lambda x: x[1], reverse=True)

        # shape: (example_idx, embedding size)
        sbert_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(turn_label) for (turn_label, _) in example_scores
        ])
        bm25_embeddings: NDArray = {
            turn_label: self.retriever.label_to_bm25_search_embedding(turn_label) for turn_label, _ in bm25_examples.items()
        }

        tmp = []
        idx_label_dict = {}
        for idx, (turn_label, cos_score) in enumerate(example_scores):
            idx_label_dict.update({idx: turn_label})
            tmp.append(cos_score)
        example_scores = np.array(tmp)
        assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected

        while len(result) < k:
            # find the current best scoring example
            best_idx: int = np.argmax(example_scores).item()
            example_scores[best_idx] = -np.inf
            result.append(best_idx)
            # Update the scores. The worst-case decrease in score is defined by discount_factor.
            best_emb: NDArray = sbert_embeddings[best_idx]
            best_bm25_emb:NDArray = bm25_embeddings[idx_label_dict[best_idx]]
            
            discount = self.standardize(self.retriever.get_bm25_scores_for_query(best_bm25_emb, sbert_examples))
            if self.operation == "sum":
                discount += self.standardize(cosine_similarity(best_emb[None, :], sbert_embeddings).squeeze(0))
            else:
                discount *= self.standardize(cosine_similarity(best_emb[None, :], sbert_embeddings).squeeze(0))
            discount *= self.discount_factor 
            example_scores = example_scores - discount

        return [self.retriever.label_to_data_item(idx_label_dict[i]) for i in result][::-1] # [low to high]

    def div_topk_round_robin(
        self,
        sbert_examples: List[Tuple[Turn, float]],
        bm25_examples: List[Tuple[Turn, float]],
        k: int
    ) -> List[Turn]:
        """
        bm25 -> diversity -> top K  \
                                     -> Union
        sbert -> diversity -> top k /
        """

        sbert_examples = self.sbert_div(sbert_examples, k)[::-1]
        bm25_examples = self.bm25_div(bm25_examples, k)[::-1]
    
        # Round-robin of the two lists
        result = []
        for i in range(k):
            result.append(sbert_examples[i])
            result.append(bm25_examples[i])

        return result[::-1] # [low to high]
    
    def bm25_div(self, bm25_examples, k = 10):
        """
        BM25 -> diversity
        """
        # shape: (example_idx, embedding size)
        bm25_embeddings = [
            self.retriever.label_to_search_embedding(self.retriever.data_item_to_label(turn))
            for turn, score in bm25_examples
        ]
        if len(bm25_examples) == 0:
            return []
        
        bm25_corpus = bm25_embeddings if 'ret' in self.decoding_pool else  bm25_examples
        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result: List[int] = []

        # scores from retriever are euclidean distances, of unit vectors (range 0-2). cos(y, z) = 1-.5*euc(y, z)^2
        # We initialize example_scores with the cosine similarity of each example e to x, the input turn (implicit from
        #  order of argument examples to select_k).
        example_scores: NDArray = np.asarray([score for turn, score in bm25_examples])
        assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected
        while len(result) < k:
            # find the current best scoring example
            best_idx: int = np.argmax(example_scores).item()
            example_scores[best_idx] = -np.inf
            result.append(best_idx)
            # Update the scores. The worst-case decrease in score is defined by discount_factor.
            best_emb: NDArray = bm25_embeddings[best_idx]
            discount: NDArray = self.discount_factor * self.retriever.get_scores_for_query(best_emb, bm25_corpus)
            example_scores = example_scores - discount
        return [bm25_examples[i][0] for i in result][::-1]  # [low to high]

    def sbert_div(self, examples, k = 10):
        """
        SBERT -> diversity
        """
         # Select up to "from N possible" in the iterator
        examples: List[Tuple[Turn, float]] = \
            [turn_and_score for _, turn_and_score in zip(range(self.from_n_possible), examples)]
        # shape: (example_idx, embedding size)
        sbert_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(self.retriever.data_item_to_label(turn))
            for turn, score in examples
        ])
        if len(examples) == 0:
            return []
        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result: List[int] = []

        # scores from retriever are euclidean distances, of unit vectors (range 0-2). cos(y, z) = 1-.5*euc(y, z)^2
        # We initialize example_scores with the cosine similarity of each example e to x, the input turn (implicit from
        #  order of argument examples to select_k).
        example_scores: NDArray = np.asarray([1 - 0.5*(score**2) for turn, score in examples])
        assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected
        while len(result) < k:
            # find the current best scoring example
            best_idx: int = np.argmax(example_scores).item()
            example_scores[best_idx] = -np.inf
            result.append(best_idx)
            # Update the scores. The worst-case decrease in score is defined by discount_factor.
            best_emb: NDArray = sbert_embeddings[best_idx]
            discount: NDArray = self.discount_factor * cosine_similarity(best_emb[None, :], sbert_embeddings).squeeze(0)
            example_scores = example_scores - discount
        return [examples[i][0] for i in result][::-1]   # [low to high]
    
    def standardize(self, emb):
        emb = (emb-np.mean(emb))/np.std(emb)
        return 1 / (1+ np.exp(-emb))
            # tf-idf => 35-4 
            # cos sim => -1 1
            # cos sim -> z-score
            # tf-idf -> z-score


    
    