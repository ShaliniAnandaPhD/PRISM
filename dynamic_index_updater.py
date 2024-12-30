import os
import numpy as np
import pandas as pd
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings  # Advanced embedding pipeline
from weaviate import Client, ObjectsBatchRequest  # Scalable vector storage and updates
from pydantic import BaseModel, FilePath, DirectoryPath

# Configuration for Weaviate client
WEAVIATE_URL = "http://localhost:8080"
VECTOR_CLASS_NAME = "DocumentVector"

# Initialize Weaviate client
client = Client(WEAVIATE_URL)
print("Connected to Weaviate server.")

# Define embedding model from LangChain
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Loaded embedding model from LangChain.")


class UpdateConfig(BaseModel):
    """
    Configuration for updating the vector index.
    """
    input_dir: DirectoryPath  # Directory containing new data files
    index_metadata_path: FilePath  # Path to the existing index metadata file


def load_new_data(input_dir: DirectoryPath) -> pd.DataFrame:
    """
    Loads new data files into a DataFrame for processing.

    Args:
        input_dir (DirectoryPath): Directory containing new data files.

    Returns:
        pd.DataFrame: DataFrame containing the new data.
    """
    data_frames = []
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path)
            df["source_file"] = file_name  # Track the source file
            data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    print(f"Loaded {len(data_frames)} new data files.")
    return combined_df


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generates embeddings for a list of texts using the LangChain HuggingFaceEmbeddings pipeline.

    Args:
        texts (List[str]): List of text entries.

    Returns:
        np.ndarray: Embedding vectors.
    """
    embeddings = embedding_model.embed_documents(texts)
    print(f"Generated embeddings for {len(texts)} entries.")
    return np.array(embeddings)


def update_weaviate_index(vectors: np.ndarray, metadata: pd.DataFrame):
    """
    Updates the Weaviate vector index with new vectors and metadata.

    Args:
        vectors (np.ndarray): Embedding vectors to add to the index.
        metadata (pd.DataFrame): Metadata associated with the vectors.
    """
    batch = ObjectsBatchRequest()

    for idx, vector in enumerate(vectors):
        metadata_entry = metadata.iloc[idx].to_dict()
        batch.add_object(
            class_name=VECTOR_CLASS_NAME,
            vector=vector.tolist(),
            properties=metadata_entry,
        )

    client.batch.create(batch)
    print(f"Updated Weaviate index with {len(vectors)} new entries.")


if __name__ == "__main__":
    """
    Entry point for dynamically updating the vector index.
    """
    # Configuration
    config = UpdateConfig(
        input_dir="./new_data",
        index_metadata_path="./vector_index/metadata.csv",
    )

    # Load new data
    new_data = load_new_data(config.input_dir)

    # Generate embeddings for the new data
    embeddings = generate_embeddings(new_data["Tokens"].tolist())

    # Update the Weaviate index with new embeddings and metadata
    update_weaviate_index(embeddings, new_data)

# ---------------------------------------------------------------
# What We Did:
# ---------------------------------------------------------------
# - Used LangChain's HuggingFaceEmbeddings for generating embeddings with state-of-the-art efficiency.
# - Integrated Weaviate for dynamic, scalable updates to the vector index without re-indexing.
# - Designed a modular system to handle new data files and seamlessly add them to the existing index.
# ---------------------------------------------------------------
# What's Next:
# ---------------------------------------------------------------
# - Add support for hybrid search using Weaviate's hybrid (text + vector) capabilities.
# - Implement background processing for continuous updates to the vector index.
# - Build monitoring tools to track the quality of vector search after updates.
# ---------------------------------------------------------------
