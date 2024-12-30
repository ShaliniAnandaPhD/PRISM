import os
import faiss  # Scalable similarity search
import numpy as np
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer  # State-of-the-art embeddings
import pyarrow.csv as csv  # Efficient CSV handling with PyArrow
from pyarrow import Table  # High-performance columnar storage

class QueryConfig:
    """
    Configuration for querying the FAISS index.
    """
    def __init__(self, model_name: str, index_path: str, metadata_path: str):
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path

    def validate_paths(self):
        """
        Ensures the paths to the FAISS index and metadata exist.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")


def load_index(index_path: str) -> faiss.IndexFlatL2:
    """
    Loads the FAISS index from the specified path.

    Args:
        index_path (str): Path to the FAISS index.

    Returns:
        faiss.IndexFlatL2: Loaded FAISS index.
    """
    index = faiss.read_index(index_path)
    print(f"Loaded FAISS index with {index.ntotal} entries.")
    return index


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Loads metadata from the specified CSV file using PyArrow for performance.

    Args:
        metadata_path (str): Path to the metadata file.

    Returns:
        pd.DataFrame: Metadata as a pandas DataFrame.
    """
    table = csv.read_csv(metadata_path)
    metadata = table.to_pandas()
    print(f"Loaded metadata with {len(metadata)} entries.")
    return metadata


def generate_query_embedding(model_name: str, query: str) -> np.ndarray:
    """
    Generates an embedding for the query using Sentence Transformers.

    Args:
        model_name (str): The Hugging Face model to use for embeddings.
        query (str): Query text.

    Returns:
        np.ndarray: Query embedding.
    """
    model = SentenceTransformer(model_name)
    embedding = model.encode([query], convert_to_numpy=True)
    print("Generated query embedding.")
    return embedding


def search_index(index: faiss.IndexFlatL2, query_embedding: np.ndarray, k: int = 5) -> List[int]:
    """
    Searches the FAISS index for the top-k most similar embeddings.

    Args:
        index (faiss.IndexFlatL2): FAISS index to search.
        query_embedding (np.ndarray): Query embedding.
        k (int): Number of top results to return.

    Returns:
        List[int]: Indices of the top-k most similar embeddings.
    """
    distances, indices = index.search(query_embedding, k)
    print(f"Top-{k} results found with distances: {distances.flatten()}")
    return indices.flatten()


def get_results(metadata: pd.DataFrame, indices: List[int]) -> pd.DataFrame:
    """
    Retrieves metadata for the top-k results.

    Args:
        metadata (pd.DataFrame): Metadata DataFrame.
        indices (List[int]): Indices of the top-k results.

    Returns:
        pd.DataFrame: Metadata of the top-k results.
    """
    results = metadata.iloc[indices]
    print(f"Retrieved results for indices: {indices}")
    return results


if __name__ == "__main__":
    """
    Entry point for querying the vector index.
    """
    # Configuration
    config = QueryConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_path="./vector_index/faiss_index.bin",
        metadata_path="./vector_index/metadata.csv",
    )

    # Validate paths
    config.validate_paths()

    # Load FAISS index and metadata
    index = load_index(config.index_path)
    metadata = load_metadata(config.metadata_path)

    # Example query
    query_text = "How to handle legal workflows efficiently?"
    query_embedding = generate_query_embedding(config.model_name, query_text)

    # Search the index
    top_k_indices = search_index(index, query_embedding, k=5)

    # Retrieve results
    results = get_results(metadata, top_k_indices)

    # Display results
    print("\nTop-5 Similar Results:")
    print(results)

# ---------------------------------------------------------------
# What We Did:
# ---------------------------------------------------------------
# - Loaded a FAISS index and associated metadata using PyArrow and pandas.
# - Generated query embeddings with Sentence Transformers.
# - Searched the FAISS index for the top-k most similar embeddings.
# - Retrieved and displayed metadata for the top results.
# ---------------------------------------------------------------
# What's Next:
# ---------------------------------------------------------------
# - Add cosine similarity as an alternative search option.
# - Build a FastAPI-based query service for real-time user interactions.
# - Optimize embeddings for specific domains using fine-tuning.
# ---------------------------------------------------------------
