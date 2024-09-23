from typing import Tuple, Iterator, List, Dict

import numpy as np
from numpy.typing import NDArray
from refpydst.data_types import Turn
from sklearn.metrics.pairwise import cosine_similarity

from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever

from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering, OPTICS, HDBSCAN

CLUSTERING_ALGORITHMS = {
    'kmeans': KMeans,
    'spectral': SpectralClustering,
    'dbscan': DBSCAN,
    'agglomerative': AgglomerativeClustering,
    'optics': OPTICS,
    'hdbscan': HDBSCAN
}

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
        self.clustering_alogrithm = kwargs.get('clustering', 'kmeans')
        self.clustering_kwargs = kwargs.get('clustering_kwargs', {})

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
        sbert_examples, bm25_examples, sbert_embeddings = self.union_examples_update_score(sbert_examples, bm25_examples, sbert_score, bm25_score)

        self.clustering = CLUSTERING_ALGORITHMS[self.clustering_alogrithm]

        results = self.clustering(**self.clustering_kwargs)

        """
            KMeans: 
                n_clusters, init, n_init, max_iter, tol, verbose, random_state, copy_x, algorithm

            SpectralClustering: 
                n_clusters, eigen_solver, n_components, random_state, n_init, gamma, affinity, n_neighbors, 
                eigen_tol, assign_labels, degree, coef0, kernel_params, n_jobs

            DBSCAN: 
                eps, min_samples, metric, metric_params, algorithm, leaf_size, p, n_jobs

            AgglomerativeClustering: 
                n_clusters, metric, memory, connectivity, compute_full_tree, linkage, distance_threshold

            OPTICS: 
                min_samples, max_eps, metric, p, metric_params, cluster_method, eps, xi, predecessor_correction, min_cluster_size, 
                algorithm, leaf_size, n_jobs

            HDBSCAN: 
                min_cluster_size, min_samples, alpha, cluster_selection_epsilon, metric, p, algorithm, leaf_size, approx_min_span_tree,
                gen_min_span_tree, core_dist_n_jobs, cluster_selection_method, allow_single_cluster, prediction_data, 
                match_reference_implementation, memory, n_jobs
        """

        

    
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

        sbert_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(turn_label) for turn_label in sbert_examples
        ])
        return sbert_examples, bm25_examples, sbert_embeddings
