from typing import Dict, List, Annotated
import numpy as np
import os
import pickle
import shutil
from memory_profiler import profile
from sklearn.cluster import MiniBatchKMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
NUM_CLUSTERS = 150
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.txt", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        # if new_db:
        #     if db_size is None:
        #         raise ValueError("You need to provide the size of the database")
        #     # delete the old DB file if exists
        #     if os.path.exists(self.db_path):
        #         os.remove(self.db_path)
        #     self.generate_database(db_size)
            
    
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

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        
        centroids_file_path = os.path.splitext(self.index_path)[0] + ".pkl"
        with open(centroids_file_path, 'rb') as centroids_file:
            centroids = pickle.load(centroids_file)
            
        query = self.normalize_vector(query)
        similarities = np.dot(centroids, query.T).flatten() 
        sorted_centroid_indices = np.argsort(similarities)[::-1] 
        results = []
        
        for centroid_index in sorted_centroid_indices:
            cluster_results = self.retrieve_from_certain_cluster(centroid_index)
            results.extend(cluster_results)
            if len(results) >= top_k:
                break
            
        return self.brute_force_retrieve(query, results)[:top_k]

        
    def retrieve_from_certain_cluster(self, cluster_id):
        results = []
        index_file_path = os.path.splitext(self.index_path)[0] + ".txt"
        with open(index_file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num == cluster_id:
                    row_numbers = list(map(int, line.strip().split()))
                    for row_num in row_numbers:
                        results.append(row_num)
        return results
    
    def brute_force_retrieve(self, query, results):
        sorted_results = []
        temp = []
        for result in results:
            vector = self.get_one_row(result)
            score = self._cal_score(query, vector)
            temp.append((result, score))
        sorted_results = [x[0] for x in sorted(temp, key=lambda x: x[1], reverse=True)]
        return sorted_results
        
  
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        # Load all vectors from the database
        vectors = self.get_all_rows()
        
        # Perform k-means clustering
        kmeans = MiniBatchKMeans(
            n_clusters=NUM_CLUSTERS,
            batch_size=int(10000),
            n_init=10, 
            random_state=DB_SEED_NUMBER
        )
        kmeans.fit(vectors)
        
        
        cluster_assignments = kmeans.labels_
        centroids = kmeans.cluster_centers_ 
        
        index_dict = {i: [] for i in range(NUM_CLUSTERS)}
        
        for vector_id, cluster_id in enumerate(cluster_assignments):
            index_dict[cluster_id].append(vector_id)
        
        centroids_file_path = os.path.splitext(self.index_path)[0] + "_centroids.pkl"
        with open(centroids_file_path, 'wb') as centroids_file:
            pickle.dump(centroids, centroids_file)

        

        with open(self.index_path, 'w') as f:
            for cluster_id, vector_indices in sorted(index_dict.items()):
                row_indices_str = ' '.join(map(str, sorted(vector_indices)))
                f.write(f"{row_indices_str}\n")
                
            
    
    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)