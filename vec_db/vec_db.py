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
TOP_M_CLUSTERS = 30
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.txt", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        self.num_records = self._get_num_records()
            
    
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

    def _load_vectors(self, row_indices: List[int]) -> np.ndarray:
        # Efficiently load multiple rows at once
        num_records = self._get_num_records()
        mmap_vectors = np.memmap(
            self.db_path, 
            dtype=np.float32, 
            mode='r', 
            shape=(num_records, DIMENSION)
        )
        return np.array([mmap_vectors[i] for i in row_indices])
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        
        centroids_file_path = os.path.splitext(self.index_path)[0] + ".pkl"
        with open(centroids_file_path, 'rb') as centroids_file:
            centroids_with_offsets = pickle.load(centroids_file)
        
        centroids = np.array([item[0] for item in centroids_with_offsets])
        offsets = [item[1] for item in centroids_with_offsets]
        if self.num_records <= 10**6:
            top_m_clusters = 400
        elif self.num_records <= 10**7:
            top_m_clusters = 160
        elif self.num_records <= 15000000:
            top_m_clusters = 40
        else:
            top_m_clusters = 1
        query = self.normalize_vector(query)
        similarities = np.dot(centroids, query.T).flatten()
        sorted_centroid_indices = np.argsort(similarities)[::-1][:top_m_clusters]
        results = []
        index_file_path = os.path.splitext(self.index_path)[0] + ".txt"
        with open(index_file_path, 'r') as index_file:
            for cluster_id in sorted_centroid_indices:
                index_file.seek(offsets[cluster_id])  # Jump to the byte offset of the cluster
                row_numbers = list(map(int, index_file.readline().strip().split()))
                results.extend(row_numbers)
        
        return self.brute_force_retrieve(query, results)[:top_k]

        
    def retrieve_from_certain_cluster(self, cluster_id):
        results = []
        index_file_path = os.path.splitext(self.index_path)[0] + ".txt"
        with open(index_file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num == cluster_id:
                    row_numbers = list(map(int, line.strip().split()))
                    results.extend(row_numbers)
                    break
        return results
    
    def retrieve_clusters(self, clusters):
        results = []
        index_file_path = os.path.splitext(self.index_path)[0] + ".txt"
        count = 0
        with open(index_file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num in clusters:
                    row_numbers = list(map(int, line.strip().split()))
                    results.extend(row_numbers)
                    count += 1
                    if count >= TOP_M_CLUSTERS:
                        break
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
        faiss.StandardGpuResources()
        assert faiss.get_num_gpus() > 0, "No GPU detected by FAISS!"
        
        kmeans = faiss.Kmeans(DIMENSION, NUM_CLUSTERS, niter=300, verbose=True, gpu=True)
        kmeans.train(vectors)
        
        cluster_assignments = kmeans.index.search(vectors, 1)[1].flatten()
        centroids = kmeans.centroids 
        
        index_dict = {i: [] for i in range(NUM_CLUSTERS)}
        
        for vector_id, cluster_id in enumerate(cluster_assignments):
            index_dict[cluster_id].append(vector_id)
        
        centroids_with_offsets = []
        
        with open(self.index_path, 'w') as f:
            for cluster_id, vector_indices in sorted(index_dict.items()):
                row_indices_str = ' '.join(map(str, sorted(vector_indices)))
                offset = f.tell()  # Record the current byte offset
                centroids_with_offsets.append((centroids[cluster_id], offset))
                f.write(f"{row_indices_str}\n")
        
        # Save centroids and offsets to a file
        centroids_file_path = os.path.splitext(self.index_path)[0] + "_centroids.pkl"
        with open(centroids_file_path, 'wb') as centroids_file:
            pickle.dump(centroids_with_offsets, centroids_file)
                
            
    
    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)