import os
import faiss  # Scalable similarity search
import numpy as np
import pandas as pd
from typing import List
from pydantic import BaseModel, DirectoryPath, FilePath
from sentence_transformers import SentenceTransformer  # State-of-the-art embeddings

class VectorIndexConfig(BaseModel):
    """
    Configuration for building a vector index.
    """
    input_dir: DirectoryPath  # Directory containing processed CSV files
    output_dir: DirectoryPath  # Directory to save the FAISS index and metadata
    model_name: str  # Hugging Face Sentence Transformer model name

    def ensure_output_dir(self):
        """
        Ensures the output directory exists. If not, it creates one.
        """
        os.makedirs(self.output_dir, exist_ok=True)


def load_processed_files(input_dir: DirectoryPath) -> pd.DataFrame:
    """
    Loads all processed CSV files into a single DataFrame.

    Args:
        input_dir (DirectoryPath): Directory containing processed CSV files.

    Returns:
        pd.DataFrame: DataFrame containing all tokens from the processed files.
    """
    data_frames = []
    for file_name in os.listdir(input_dir):
        if file_name.endswith("_processed.csv"):
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path)
            df["source_file"] = file_name  # Track the source file
            data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    print(f"Loaded {len(data_frames)} files into a single DataFrame.")
    return combined_df


def create_embeddings(model_name: str, sentences: List[str]) -> np.ndarray:
    """
    Generates embeddings for a list of sentences using Sentence Transformers.

    Args:
        model_name (str): The Hugging Face model to use for embeddings.
        sentences (List[str]): List of tokenized sentences.

    Returns:
        np.ndarray: Array of embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=True)
    print(f"Generated embeddings for {len(sentences)} sentences.")
    return np.array(embeddings)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Builds a FAISS index for the provided embeddings.

    Args:
        embeddings (np.ndarray): Embeddings to index.

    Returns:
        faiss.IndexFlatL2: FAISS index for similarity search.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 (Euclidean) similarity
    index.add(embeddings)
    print(f"FAISS index created with {index.ntotal} entries.")
    return index


def save_index(output_dir: DirectoryPath, index: faiss.IndexFlatL2, metadata: pd.DataFrame):
    """
    Saves the FAISS index and metadata to the output directory.

    Args:
        output_dir (DirectoryPath): Directory to save the index and metadata.
        index (faiss.IndexFlatL2): FAISS index.
        metadata (pd.DataFrame): Metadata associated with the embeddings.
    """
    # Save FAISS index
    index_path = os.path.join(output_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata.to_csv(metadata_path, index=False)

    print(f"FAISS index saved to {index_path}.")
    print(f"Metadata saved to {metadata_path}.")


if __name__ == "__main__":
    """
    Entry point for building the vector index.
    """
    # Configuration
    config = VectorIndexConfig(
        input_dir="./processed_data",
        output_dir="./vector_index",
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # Lightweight, fast, and accurate model
    )

    # Ensure output directory exists
    config.ensure_output_dir()

    # Load processed tokens
    combined_data = load_processed_files(config.input_dir)

    # Create embeddings
    embeddings = create_embeddings(config.model_name, combined_data["Tokens"].tolist())

    # Build FAISS index
    faiss_index = build_faiss_index(embeddings)

    # Save index and metadata
    save_index(config.output_dir, faiss_index, combined_data)

# ---------------------------------------------------------------
# What We Did:
# ---------------------------------------------------------------
# - Used Sentence Transformers to generate state-of-the-art sentence embeddings.
# - Built a scalable similarity search index using FAISS for fast retrieval.
# - Saved both the FAISS index and associated metadata for future use.
# ---------------------------------------------------------------
# What's Next:
# ---------------------------------------------------------------
# - Implement an efficient query pipeline for real-time document retrieval.
# - Extend support for additional similarity metrics, like cosine similarity.
# - Build a web API using FastAPI to expose the vector search functionality.
# ---------------------------------------------------------------
