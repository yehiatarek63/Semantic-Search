from typing import Dict, List, Annotated
import numpy as np
import os
import pickle
import shutil

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
RANDOM_HYPERPLANS_NUM = 10
HYPERPLANE_SEED = 42

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.txt", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.signatures = set()
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
        query_signature = self._generate_signature(query)
        bucket_number = int(''.join(map(str, query_signature)), 2)
        results = []
        
        while len(results) < top_k:
            row_indices = self.retrieve_from_certain_bucket(bucket_number)
            results.extend(row_indices)
            bucket_number += 1
            if bucket_number >= 2**len(query_signature):
                break
            
        sorted_results = self.brute_force_retrieve(query, results)
        print(sorted_results[:top_k])
        return sorted_results[:top_k]
            
            
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
        self.hyperplanes = hyperplane_rng.normal(loc=0, scale=1, size=(RANDOM_HYPERPLANS_NUM, DIMENSION))        

        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        open(self.index_path, 'w').close()
        
        vectors = self.get_all_rows()
        
        for row_num, vector in enumerate(vectors): 
            signature = self._generate_signature(vector)
            signature_str = ''.join(map(str, signature))
            bucket_number = int(signature_str, 2)
            
            with open(self.index_path, 'a+') as f:
                f.seek(0)
                lines = f.readlines()
        
            while len(lines) <= bucket_number:
                lines.append("\n")
            
            current_line = lines[bucket_number].strip()
            if current_line:
                lines[bucket_number] = lines[bucket_number].strip() + " " + f"{row_num} " + "\n"
            else:
                lines[bucket_number] = f"{row_num}\n"
            

            with open(self.index_path, 'w') as f:
                f.writelines(lines)
                
        print("Index built successfully.")
            
    def _generate_signature(self, vector):
        signature = []
        for hyperplane in self.hyperplanes:
            dot_product = np.dot(vector, hyperplane)
            signature.append(1 if dot_product > 0 else 0)
        return signature
    