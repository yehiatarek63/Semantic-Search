from typing import Dict, List, Annotated
import numpy as np
import os
import pickle
import shutil
from memory_profiler import profile

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
RANDOM_HYPERPLANS_NUM = 20
HYPERPLANE_SEED = 42

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.txt", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.num_buckets = 0
        self.populated_buckets = []
        self.bucket_set = set()
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
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
        # Step 1: Generate the signature for the query vector
        query_signature = self._generate_signature(query)
        query_signature_str = ''.join(map(str, query_signature))

        # Step 2: Compute Hamming distances to all populated buckets
        bucket_distances = []
        for bucket in self.populated_buckets:
            bucket_signature = f"{bucket:0{len(query_signature_str)}b}"
            hamming_distance = sum(c1 != c2 for c1, c2 in zip(query_signature_str, bucket_signature))
            bucket_distances.append((bucket, hamming_distance))

        # Step 3: Sort buckets by Hamming distance
        bucket_distances.sort(key=lambda x: x[1])

        # Step 4: Retrieve contents from buckets until top_k results are gathered
        results = []
        for bucket, _ in bucket_distances:
            if len(results) >= top_k:
                break

            bucket_values = self.retrieve_from_certain_bucket(bucket)
            results.extend(bucket_values)

        # Step 5: Perform a brute-force search over the collected results to rank them by cosine similarity
        results = self.brute_force_retrieve(query, results)

        return results[:top_k]

        
    def retrieve_from_certain_bucket(self, line_number):
        results = []
        with open(self.index_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num == line_number:
                    row_numbers = list(map(int, line.strip().split()))
                    for row_num in row_numbers:
                        row = self.get_one_row(row_num)
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
        self.hyperplanes_seed = HYPERPLANE_SEED
        hyperplane_rng = np.random.default_rng(self.hyperplanes_seed)

        # Generate normalized random hyperplanes
        random_planes = hyperplane_rng.normal(loc=0, scale=1, size=(RANDOM_HYPERPLANS_NUM, DIMENSION))
        self.hyperplanes = random_planes / np.linalg.norm(random_planes, axis=1, keepdims=True)


        # Initialize an empty dictionary for the index
        index_dict = {i: [] for i in range(2**RANDOM_HYPERPLANS_NUM)}

        vectors = self.get_all_rows()

        for row_num, vector in enumerate(vectors):
            # Generate the signature and compute the bucket number
            signature = self._generate_signature(vector)
            signature_str = ''.join(map(str, signature))
            bucket_number = int(signature_str, 2)

            # Add the row number to the appropriate bucket in the dictionary
            if bucket_number not in index_dict:
                index_dict[bucket_number] = []
            index_dict[bucket_number].append(row_num)
            self.bucket_set.add(bucket_number)

        # Sort the dictionary by bucket keys
        sorted_index = dict(sorted(index_dict.items()))
        self.num_buckets = len(sorted_index)

        # Write the sorted index dictionary to a text file
        with open(self.index_path, 'w') as f:
            for bucket, row_indices in sorted_index.items():
                row_indices_str = ' '.join(map(str, sorted(row_indices)))
                f.write(f"{row_indices_str}\n")
                
        self.populated_buckets = sorted(list(self.bucket_set))
        self.bucket_set = set(self.populated_buckets)

        print("Index built successfully.")
            
    def _generate_signature(self, vector):
        signature = []
        for hyperplane in self.hyperplanes:
            normalized_vector = self.normalize_vector(vector)
            dot_product = np.dot(normalized_vector, hyperplane)
            signature.append(1 if dot_product > 0 else 0)
        return signature
    
    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)