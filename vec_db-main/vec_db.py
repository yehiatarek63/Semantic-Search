from typing import Dict, List, Annotated
import numpy as np
import os
import pickle
import shutil

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
RANDOM_HYPERPLANS_NUM = 70

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index", new_db=True, db_size=None) -> None:
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
        self.hyperplanes = np.random.normal(loc=0, scale=1, size=(RANDOM_HYPERPLANS_NUM, DIMENSION))
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
        results = []
        tried_signatures = set()
        signature = self._generate_signature(query)
        signature_str = ''.join(map(str, signature))
        signatures_to_try = [signature]
        hamming_distance = 0

        while len(results) <= top_k:
            new_signatures = []
            for sig in signatures_to_try:
                sig_str = ''.join(map(str, sig))
                if sig_str in tried_signatures:
                    continue
                tried_signatures.add(sig_str)
                signature_file_path = os.path.join(self.index_path, f"{sig_str}.txt")
                if os.path.exists(signature_file_path):
                    with open(signature_file_path, 'r') as f:
                        for line in f:
                            row_num = int(line.strip())
                            vector = self.get_one_row(row_num)
                            score = self._cal_score(query, vector)
                            results.append((row_num, score))
                            if len(results) >= top_k:
                                break
                if len(results) >= top_k:
                    break
            if len(results) >= top_k:
                break
            hamming_distance += 1
            new_signatures = self._get_signatures_at_hamming_distance(signature, hamming_distance)
            signatures_to_try = new_signatures

        results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        final_results = [result[0] for result in results]
        print(final_results)
        return final_results

    def _get_signatures_at_hamming_distance(self, signature, distance):
        from itertools import combinations

        indices = range(len(signature))
        flip_indices = list(combinations(indices, distance))
        new_signatures = []
        for indices_to_flip in flip_indices:
            new_sig = signature.copy()
            for idx in indices_to_flip:
                new_sig[idx] ^= 1  # Flip bit
            new_signatures.append(new_sig)
        return new_signatures
        
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        # Placeholder for index building logic
        vectors = self.get_all_rows()
        for i, vector in enumerate(vectors):
            signature = self._generate_signature(vector)
            signature_tuple = tuple(signature)
            self.signatures.add(signature_tuple)
            signature_str = ''.join(map(str, signature))
            
            signature_file_path = os.path.join(self.index_path, f"{signature_str}.txt")
            
            self._append_to_file(signature_file_path, i)
        print("Index Built")
        
    def _append_to_file(self, file_path, row_num):
        """Append row number to the file corresponding to a signature."""
        with open(file_path, 'a') as f:
            f.write(f"{row_num}\n")
        
    def _save_index(self, index):
        # Save the index to the file using pickle
        with open(self.index_path, 'wb') as f:
            pickle.dump(index, f)
            
            

    def _generate_signature(self, vector):
        signature = []
        for hyperplane in self.hyperplanes:
            dot_product = np.dot(vector, hyperplane)
            signature.append(1 if dot_product > 0 else 0)
        return signature
            
if __name__ == "__main__":
    db = VecDB(db_size = 10**4)

    # Retrieve similar images for a given query
    query_vector = np.random.rand(1,70) # Query vector of dimension 70
    print(db._get_num_records())