from typing import Dict, List, Annotated
import numpy as np
import os
import pickle
import time
from dataclasses import dataclass
import faiss

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

class VecDB:
    def __init__(
        self,
        database_file_path="saved_db.dat",
        index_path="index.index",
        new_db=True,
        db_size=None,
        M=32,               # Parameter for HNSW: number of neighbors
        ef_construction=200 # Parameter for HNSW: trade-off between speed and accuracy
    ) -> None:
        """
        Initialize the VecDB with HNSW index.

        :param database_file_path: Path to the database file.
        :param index_path: Path to save/load the FAISS index.
        :param new_db: Whether to create a new database.
        :param db_size: Number of vectors in the database.
        :param M: HNSW parameter, number of neighbors.
        :param ef_construction: HNSW parameter, trade-off between speed and accuracy during construction.
        """
        self.db_path = database_file_path
        self.index_path = index_path
        self.index = None  # FAISS index
        self.M = M
        self.ef_construction = ef_construction

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # Delete the old DB file if it exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            self.generate_database(db_size)
        else:
            # Load existing index
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            else:
                raise FileNotFoundError(f"Index file {self.index_path} does not exist.")

    def generate_database(self, size: int) -> None:
        """
        Generate random vectors and build the HNSW index.

        :param size: Number of vectors to generate.
        """
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION)).astype(np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        """
        Write vectors to a memory-mapped file.

        :param vectors: NumPy array of vectors.
        """
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        """
        Get the number of vectors in the database.

        :return: Number of records.
        """
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: np.ndarray):
        """
        Insert new records into the database and update the HNSW index.

        :param rows: NumPy array of new vectors to insert.
        """
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows.astype(np.float32)
        mmap_vectors.flush()
        # Add new records to the FAISS index
        self.index.add(rows.astype(np.float32))
        self._save_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        """
        Retrieve a single vector from the database.

        :param row_num: Index of the row to retrieve.
        :return: NumPy array of the vector.
        """
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(
                self.db_path,
                dtype=np.float32,
                mode='r',
                shape=(1, DIMENSION),
                offset=offset
            )
            return np.array(mmap_vector[0])
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")

    def get_all_rows(self) -> np.ndarray:
        """
        Retrieve all vectors from the database.

        :return: NumPy array of all vectors.
        """
        num_records = self._get_num_records()
        vectors = np.memmap(
            self.db_path,
            dtype=np.float32,
            mode='r',
            shape=(num_records, DIMENSION)
        )
        return np.array(vectors)

    def retrieve(self, query: np.ndarray, top_k=10) -> List[int]:
        """
        Retrieve the top_k nearest neighbors for the given query using HNSW index.

        :param query: NumPy array representing the query vector.
        :param top_k: Number of nearest neighbors to retrieve.
        :return: List of indices of the top_k nearest neighbors.
        """
        if self.index is None:
            raise ValueError("Index has not been built yet.")
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        try:
            distances, indices = self.index.search(query, top_k)
            return indices[0].tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve results: {e}")

    def _cal_score(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors.

        :param vec1: First vector.
        :param vec2: Second vector.
        :return: Cosine similarity score.
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        """
        Build the HNSW index using FAISS.
        """
        if os.path.exists(self.index_path):
            os.remove(self.index_path)

        # Initialize a FAISS HNSW index
        hnsw_index = faiss.IndexHNSWFlat(DIMENSION, self.M)
        hnsw_index.hnsw.efConstruction = self.ef_construction  # Parameters can be tuned
        hnsw_index.hnsw.efSearch = 50  # Tune this parameter based on desired accuracy/speed

        # Add vectors to the index
        vectors = self.get_all_rows()
        hnsw_index.add(vectors)

        self.index = hnsw_index
        self._save_index()

    def _save_index(self):
        """
        Save the FAISS index to disk.
        """
        faiss.write_index(self.index, self.index_path)