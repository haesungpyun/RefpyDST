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
        self.decoding_logic = kwargs.get('decoding_logic', 'top_k_round_robin')

    def select_k(self, examples: Iterator[Tuple[Turn, float]],
                 bm25_examples:Iterator[Tuple[Turn, float]], k: int, **kwargs) -> List[Turn]:
        if isinstance(k, dict):
            self.bm25_k = k.get('BM25', 0)
            self.sbert_k = k.get('SBERT', 0)
        else:
            self.bm25_k = k
            self.sbert_k = k
        
        query_id = kwargs.get('query_id', None)
        query_doc_id = query_id.split('_turn')[0]
        
        # Select up to "from N possible" in the iterator
        sbert_examples: List[Tuple[Turn, float]] = [(turn_label, score) for turn_label, score in examples if turn_label.split('_turn')[0] != query_doc_id]
        bm25_examples: Dict[Tuple[Turn, float]] = {turn_label: score for turn_label, score in bm25_examples if turn_label.split('_turn')[0] != query_doc_id}
        
        # scores from retriever are euclidean distances, of unit vectors (range 0-2). cos(y, z) = 1-.5*euc(y, z)^2
        # We initialize example_scores with the cosine similarity of each example e to x, the input turn (implicit from
        #  order of argument examples to select_k).
        sbert_score = np.array([(1-0.5*(score**2)) for _, score in sbert_examples])
        bm25_score = np.array(list(bm25_examples.values()))
        
        if self.decoding_logic == 'div_topk_round_robin':   # done
            sbert_examples, bm25_examples = self.union_examples_update_score(sbert_examples, bm25_examples, sbert_score, bm25_score)
            return self.div_topk_round_robin(sbert_examples, bm25_examples, self.bm25_k+self.sbert_k)

        elif self.decoding_logic == 'multiply_div_top_k':
            sbert_score, bm25_score = self.transform_distribution(sbert_score, bm25_score)
            sbert_examples, bm25_examples = self.union_examples_update_score(sbert_examples, bm25_examples, sbert_score, bm25_score)
            return self.multiply_div_top_k(sbert_examples, bm25_examples, self.bm25_k+self.sbert_k)
        
        elif self.decoding_logic == 'multiply_top_k':   # done
            sbert_score, bm25_score = self.transform_distribution(sbert_score, bm25_score)
            sbert_examples, bm25_examples = self.union_examples_update_score(sbert_examples, bm25_examples, sbert_score, bm25_score)
            return self.multiply_top_k(sbert_examples, bm25_examples, self.bm25_k+self.sbert_k)       

        elif self.decoding_logic == 'round_robin_div_top_k':    # done
            sbert_score, bm25_score = self.transform_distribution(sbert_score, bm25_score)
            sbert_examples, bm25_examples = self.union_examples_update_score(sbert_examples, bm25_examples, sbert_score, bm25_score)
            return self.round_robin_div_top_k(sbert_examples, bm25_examples, self.bm25_k+self.sbert_k)
        
        elif self.decoding_logic == 'top_k_round_robin':  # done
            sbert_examples, bm25_examples = self.union_examples_update_score(sbert_examples, bm25_examples, sbert_score, bm25_score)
            return self.top_k_round_robin(sbert_examples, bm25_examples, self.bm25_k+self.sbert_k)  
    
    def div_topk_round_robin(
        self,
        sbert_examples: List[Tuple[Turn, float]],
        bm25_examples: Dict[Turn, float],
        k: int=10
    ) -> List[Turn]:
        """
        bm25 -> diversity -> top K  \
                                     -> Union
        sbert -> diversity -> top k /
        """
        sbert_examples = self.sbert_div(sbert_examples, self.from_n_possible)[::-1]
        bm25_examples = self.bm25_div(bm25_examples, self.from_n_possible)[::-1]
    
        # Round-robin of the two lists
        result = {}
        while len(result) < self.bm25_k+self.sbert_k:
            while True:
                try:
                    turn_label = sbert_examples.pop(0)
                    if turn_label not in result:
                        result.update({turn_label: 0})
                        break
                except:
                    break
            while True:
                try:
                    turn_label = bm25_examples.pop(0)
                    if turn_label not in result:
                        result.update({turn_label: 0})
                        break
                except:
                    break
        
        return [self.retriever.label_to_data_item(turn_label) for turn_label in result][::-1] # [low to high]

    def multiply_div_top_k(
        self, 
        sbert_examples: List[Tuple[Turn, float]], 
        bm25_examples: List[Tuple[Turn, float]], 
        k: int=10
    ) -> List[Turn]:
        """
        bm25 -> z-score  \
                          -> multiply -> diversity -> top K
        sbert -> z-score / 
        """

        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result : List[int] = []
        sbert_examples = dict(sbert_examples)
        bm25_examples = dict(bm25_examples)

        union_set = set(sbert_examples).union(set(bm25_examples))

        total_examples = {}
        for turn_label in union_set:
            if self.operation == "sum":
                total_examples[turn_label] = sbert_examples[turn_label] + bm25_examples[turn_label]
            elif self.operation == "multiply":
                total_examples[turn_label] = sbert_examples[turn_label] * bm25_examples[turn_label]
        
        total_examples = sorted(total_examples.items(), key=lambda x: x[1], reverse=True)   # [high to low]
        
        # shape: (example_idx, embedding size)
        sbert_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(turn_label) for turn_label in sbert_examples
        ])
        bm25_embeddings: NDArray = [
            self.retriever.label_to_bm25_search_embedding(turn_label) for turn_label in bm25_examples
        ]

        example_scores = []
        idx_label_dict = {}
        for idx, (turn_label, score) in enumerate(total_examples):
            idx_label_dict.update({idx: turn_label})
            example_scores.append(score)
        
        example_scores = np.array(example_scores)
        assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected

        while len(result) < self.bm25_k+self.sbert_k:
            # find the current best scoring example
            best_idx: int = np.argmax(example_scores).item()
            example_scores[best_idx] = -np.inf
            result.append(best_idx)
            # Update the scores. The worst-case decrease in score is defined by discount_factor.
            best_emb: NDArray = sbert_embeddings[best_idx]
            best_bm25_emb:NDArray = bm25_embeddings[best_idx]
            
            discount = self.standardize(bm25_score=self.retriever.get_bm25_scores_for_query(best_bm25_emb, bm25_embeddings))
            if self.operation == "sum":
                discount += self.standardize(sbert_score=cosine_similarity(best_emb[None, :], sbert_embeddings).squeeze(0))
            else:
                discount *= self.standardize(sbert_score=cosine_similarity(best_emb[None, :], sbert_embeddings).squeeze(0))
            
            discount *= self.discount_factor 
            example_scores = example_scores - discount

        return [self.retriever.label_to_data_item(idx_label_dict[i]) for i in result][::-1] # [low to high]
    
    def multiply_top_k(
        self, 
        sbert_examples: List[Tuple[Turn, float]], 
        bm25_examples: List[Tuple[Turn, float]], 
        k: int=10
    ) -> List[Turn]:
        """
        Selects the top k examples from the round-robin of BM25 and SBERT examples.
        
        bm25 -> z-score  \
                           -> multiply -> top K
        sbert -> z-score / 
        
        """
        sbert_examples = dict(sbert_examples)
        bm25_examples = dict(bm25_examples)

        union_set = set(sbert_examples).union(set(bm25_examples))

        total_examples = {}
        for turn_label in union_set:
            if self.operation == "sum":
                total_examples[turn_label] = sbert_examples[turn_label] + bm25_examples[turn_label]
            elif self.operation == "multiply":
                total_examples[turn_label] = sbert_examples[turn_label] * bm25_examples[turn_label]

        total_examples = sorted(total_examples.items(), key=lambda item: item[1], reverse=True) # [higt to low]
        scores = [score for _, score in total_examples]
        assert np.all(np.diff(np.array(scores)) <= 0)  # verifies they are decreasing as expected
        
        result = [self.retriever.label_to_data_item(turn_label) for turn_label, _ in total_examples[:self.bm25_k+self.sbert_k]]  # top k

        return result[::-1] # return [low to high]

    def round_robin_div_top_k(
        self, 
        sbert_examples: List[Tuple[Turn, float]], 
        bm25_examples: List[Tuple[Turn, float]], 
        k: int=10
    ) -> List[Turn]:
        """
        bm25 -> z-score  \
                          -> Union -> diversity -> top K
        sbert -> z-score / 
        """       
        total_length = len(sbert_examples)
        
        # shape: (example_idx, embedding size)
        sbert_all_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(turn_label) for (turn_label, _) in sbert_examples
        ])
        bm25_all_embeddings = [
            self.retriever.label_to_bm25_search_embedding(turn_label) for (turn_label, _) in bm25_examples
        ]

        # Round-robin of the two lists
        example_scores_embedding = {}
        s_idx, b_idx = 0, 0
        while len(example_scores_embedding) < total_length:
            while s_idx < len(sbert_examples):
                turn_label, s_score = sbert_examples[s_idx]
                if turn_label not in example_scores_embedding:
                    example_scores_embedding.update({turn_label: (s_score, sbert_all_embeddings[s_idx])})
                    s_idx += 1
                    break
                s_idx += 1
                
            while b_idx < len(bm25_examples):
                turn_label, b_score = bm25_examples[b_idx]
                if turn_label not in example_scores_embedding:
                    example_scores_embedding.update({turn_label: (b_score, bm25_all_embeddings[b_idx])})
                    b_idx += 1
                    break
                b_idx += 1
                
        example_scores_embedding = sorted(example_scores_embedding.items(), key=lambda x: x[1][0], reverse=True) # [high to low]
        sbert_all_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(turn_label) for (turn_label, _) in example_scores_embedding
        ])
        bm25_all_embeddings = [
            self.retriever.label_to_bm25_search_embedding(turn_label) for (turn_label, _) in example_scores_embedding
        ]

        idx_label_dict = {}
        example_scores = []
        embeddings = []
        for idx, (turn_label, (score, emb)) in enumerate(example_scores_embedding):
            idx_label_dict.update({idx: turn_label})
            example_scores.append(score)
            embeddings.append(emb)
        
        example_scores = np.array(example_scores)
        assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected

        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result : List[int] = []
        while len(result) < self.bm25_k+self.sbert_k:
            # find the current best scoring example
            best_idx: int = np.argmax(example_scores).item()
            example_scores[best_idx] = -np.inf
            result.append(best_idx)
            # Update the scores. The worst-case decrease in score is defined by discount_factor.
            best_emb = embeddings[best_idx]
            
            if isinstance(best_emb, np.ndarray):
                div_score = cosine_similarity(best_emb[None, :], sbert_all_embeddings).squeeze(0)
                div_score = self.standardize(sbert_score=div_score)
                discount = self.discount_factor * div_score
            else:
                div_score = self.retriever.get_bm25_scores_for_query(best_emb, bm25_all_embeddings)
                div_score = self.standardize(bm25_score=div_score)
                discount = self.discount_factor * div_score
            
            example_scores = example_scores - discount

        return [self.retriever.label_to_data_item(idx_label_dict[i]) for i in result][::-1] # [low to high]
    
    def top_k_round_robin(
        self, 
        sbert_examples: List[Tuple[Turn, float]], 
        bm25_examples: List[Tuple[Turn, float]], 
        k: int = 10
    ) -> List[Turn]:
        """
        Selects the top k examples from the round-robin of BM25 and SBERT examples.        
        bm25 -> top k/2  \
                            -> Union
        sbert -> top k/2 / 
        """
        # Round-robin of the two lists
        result = {}
        s_idx, b_idx = 0, 0
        while len(result) < self.bm25_k+self.sbert_k:
            while s_idx < len(sbert_examples):
                turn_label, _ = sbert_examples[s_idx]
                if turn_label not in result:
                    result.update({turn_label: 0})
                    s_idx += 1
                    break
                s_idx += 1
            while b_idx < len(bm25_examples):
                turn_label, _ = bm25_examples[b_idx]
                if turn_label not in result:
                    result.update({turn_label: 0})
                    b_idx += 1
                    break
                b_idx += 1

        return [self.retriever.label_to_data_item(turn_label) for turn_label in result][::-1] # [low to high]
    
    def bm25_div(self, bm25_examples, k = 10):
        """
        BM25 -> diversity
        """
        # shape: (example_idx, embedding size)
        bm25_embeddings = [
            self.retriever.label_to_bm25_search_embedding(turn_label)
            for turn_label, score in bm25_examples
        ]
        if len(bm25_examples) == 0:
            return []
    
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
            discount: NDArray = self.discount_factor * self.retriever.get_bm25_scores_for_query(best_emb, bm25_embeddings)
            example_scores = example_scores - discount
        return [bm25_examples[i][0] for i in result][::-1]  # [low to high]

    def sbert_div(self, examples, k = 10):
        """
        SBERT -> diversity
        """       
        # shape: (example_idx, embedding size)
        sbert_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(turn_label) 
            for turn_label, score in examples
        ])
        if len(examples) == 0:
            return []
        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result: List[int] = []

        # scores from retriever are euclidean distances, of unit vectors (range 0-2). cos(y, z) = 1-.5*euc(y, z)^2
        # We initialize example_scores with the cosine similarity of each example e to x, the input turn (implicit from
        #  order of argument examples to select_k).
        example_scores: NDArray = np.asarray([score for turn, score in examples])
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
    
    def union_examples_update_score(self, sbert_examples, bm25_examples, cos_score=None, bm25_score=None):
        if cos_score is None:
            cos_score = np.array([score for turn, score in sbert_examples])
        sbert_dict = {
                turn_label: c_score 
            for (turn_label, _), c_score in zip(sbert_examples, cos_score)
        }

        if bm25_score is None:
            bm25_score = np.array([score for turn, score in bm25_examples])
        bm25_dict = {
                turn_label: b_score 
            for (turn_label, _), b_score in zip(bm25_examples.items(), bm25_score)
        }
        
        sbert_set = set(list(sbert_dict.keys())[:self.from_n_possible])
        bm25_set = set(list(bm25_dict.keys())[:self.from_n_possible])
        union_set = sbert_set.union(bm25_set)
        
        sbert_examples = [(turn_label, sbert_dict[turn_label]) for turn_label in union_set if turn_label in sbert_dict]
        bm25_examples = [(turn_label, bm25_dict[turn_label]) for turn_label in union_set if turn_label in bm25_dict]

        sbert_examples = list(sorted(sbert_examples, key=lambda x: x[1], reverse=True)) # [high to low]
        bm25_examples = list(sorted(bm25_examples, key=lambda item: item[1], reverse=True)) # [high to low]

        assert len(sbert_examples) == len(union_set) or len(bm25_examples) == len(union_set)
        return sbert_examples, bm25_examples

    def transform_distribution(self, sbert_score, bm25_score):
        # tf-idf => [35-4] => z-score
        # cos sim => [-1 1] => [0 2] => [0 1] => z-score
        sbert_score = (sbert_score+1)/2     # [0, 1]
        if self.zscore:
            self.sbert_subtrahend, self.sbert_divisor= np.mean(sbert_score), np.std(sbert_score)
            self.bm25_subtrahend, self.bm25_divisor = np.mean(bm25_score), np.std(bm25_score)
        else:
            self.sbert_subtrahend, self.sbert_divisor = np.min(sbert_score), (np.max(sbert_score) - np.min(sbert_score))
            self.bm25_subtrahend, self.bm25_divisor = np.min(sbert_score), (np.max(sbert_score) - np.min(sbert_score))
            
        sbert_score = (sbert_score - self.sbert_subtrahend)/self.sbert_divisor
        bm25_score = (bm25_score-self.bm25_subtrahend)/self.bm25_divisor
        return sbert_score, bm25_score

    def standardize(self, sbert_score=None, bm25_score=None):
        if sbert_score is None and bm25_score is None:
            raise ValueError("Both scores provided to standard")
        if sbert_score is not None: 
            sbert_score = (sbert_score+1)/2     # [0, 1]
            return (sbert_score - self.sbert_subtrahend) / self.sbert_divisor
        elif bm25_score is not None:
            return (bm25_score - self.bm25_subtrahend) / self.bm25_divisor
        else:   
            raise ValueError("No score provided to standard")