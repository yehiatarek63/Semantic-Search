from typing import Dict, List, Annotated
import numpy as np
import os
import pickle
from sklearn.cluster import MiniBatchKMeans
from functools import lru_cache

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
NUM_CLUSTERS = 1000
TOP_M_CLUSTERS = 90
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.txt", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self._centroids = None
        self._cluster_index = None
        self._mmap_vectors = None
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    @property
    def mmap_vectors(self):
        """Lazy loading of memory-mapped vectors"""
        if self._mmap_vectors is None:
            num_records = self._get_num_records()
            self._mmap_vectors = np.memmap(
                self.db_path, 
                dtype=np.float32, 
                mode='r', 
                shape=(num_records, DIMENSION)
            )
        return self._mmap_vectors

    @property
    def centroids(self):
        """Lazy loading of centroids"""
        if self._centroids is None:
            centroids_file_path = os.path.splitext(self.index_path)[0] + "_centroids.pkl"
            with open(centroids_file_path, 'rb') as centroids_file:
                self._centroids = pickle.load(centroids_file)
        return self._centroids

    @property
    def cluster_index(self):
        """Lazy loading of cluster index"""
        if self._cluster_index is None:
            cluster_index_file_path = os.path.splitext(self.index_path)[0] + "_cluster_index.pkl"
            with open(cluster_index_file_path, 'rb') as index_file:
                self._cluster_index = pickle.load(index_file)
        return self._cluster_index

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        
        # Reset cached mmap vectors
        self._mmap_vectors = None
        
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()

    @lru_cache(maxsize=1024)
    def get_one_row(self, row_num: int) -> np.ndarray:
        """Cached row retrieval"""
        return np.array(self.mmap_vectors[row_num])

    def get_all_rows(self) -> np.ndarray:
        return np.array(self.mmap_vectors)

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        # Normalize query once
        query = self.normalize_vector(query)
        
        # Use vectorized operations for centroid similarity computation
        similarities = np.dot(self.centroids, query.T).flatten()
        top_centroid_indices = np.argpartition(similarities, -TOP_M_CLUSTERS)[-TOP_M_CLUSTERS:]
        
        # Get all candidate vectors at once
        candidate_indices = []
        for centroid_idx in top_centroid_indices:
            candidate_indices.extend(self.cluster_index.get(centroid_idx, []))
        
        # Batch process candidates
        if candidate_indices:
            candidate_vectors = self.mmap_vectors[candidate_indices]
            # Normalize candidates (can be done in batch)
            norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
            normalized_candidates = candidate_vectors / norms
            # Compute similarities in one operation
            candidate_similarities = np.dot(normalized_candidates, query.T).flatten()
            # Get top k results
            top_k_indices = np.argpartition(candidate_similarities, -top_k)[-top_k:]
            top_k_indices = top_k_indices[np.argsort(candidate_similarities[top_k_indices])[::-1]]
            return [candidate_indices[i] for i in top_k_indices]
        
        return []

    def _build_index(self):
        vectors = self.get_all_rows()
        
        kmeans = MiniBatchKMeans(
            n_clusters=NUM_CLUSTERS,
            batch_size=50000,
            n_init=1,
            random_state=DB_SEED_NUMBER,
            verbose=1
        )
        kmeans.fit(vectors)
        
        # Reset cached properties
        self._centroids = kmeans.cluster_centers_
        self._cluster_index = {i: [] for i in range(NUM_CLUSTERS)}
        
        for vector_id, cluster_id in enumerate(kmeans.labels_):
            self._cluster_index[cluster_id].append(vector_id)
        
        # Save to disk
        centroids_file_path = os.path.splitext(self.index_path)[0] + "_centroids.pkl"
        with open(centroids_file_path, 'wb') as centroids_file:
            pickle.dump(self._centroids, centroids_file)
        
        cluster_index_file_path = os.path.splitext(self.index_path)[0] + "_cluster_index.pkl"
        with open(cluster_index_file_path, 'wb') as index_file:
            pickle.dump(self._cluster_index, index_file)

    @staticmethod
    def normalize_vector(vector):
        return vector / np.linalg.norm(vector)



