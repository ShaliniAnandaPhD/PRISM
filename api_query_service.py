from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import os

# Initialize FastAPI application
app = FastAPI(title="FAISS Vector Search API", version="1.0.0")

# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "./vector_index/faiss_index.bin"
METADATA_PATH = "./vector_index/metadata.csv"

# Load the Sentence Transformer model
model = SentenceTransformer(MODEL_NAME)
print("Loaded embedding model:", MODEL_NAME)

# Load the FAISS index
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
index = faiss.read_index(INDEX_PATH)
print(f"Loaded FAISS index with {index.ntotal} entries.")

# Load metadata
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
metadata = pd.read_csv(METADATA_PATH)
print(f"Loaded metadata with {len(metadata)} entries.")


# Request and Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5  # Default to top 5 results


class QueryResponse(BaseModel):
    results: list


# Helper functions
def generate_query_embedding(query: str) -> np.ndarray:
    """
    Generates an embedding for the input query using the pre-loaded model.

    Args:
        query (str): The query string.

    Returns:
        np.ndarray: Query embedding.
    """
    return model.encode([query], convert_to_numpy=True)


def search_index(embedding: np.ndarray, top_k: int) -> list:
    """
    Searches the FAISS index for the top-k most similar results.

    Args:
        embedding (np.ndarray): Query embedding.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: Indices of the top-k results in the metadata.
    """
    distances, indices = index.search(embedding, top_k)
    return indices.flatten().tolist()


def retrieve_results(indices: list) -> pd.DataFrame:
    """
    Retrieves metadata for the given indices.

    Args:
        indices (list): Indices of the results in the metadata.

    Returns:
        pd.DataFrame: Retrieved metadata rows.
    """
    return metadata.iloc[indices].to_dict(orient="records")


# API Endpoints
@app.post("/query", response_model=QueryResponse)
async def query_vector_index(request: QueryRequest):
    """
    Endpoint to query the FAISS vector index.

    Args:
        request (QueryRequest): Contains the query string and the number of results to retrieve.

    Returns:
        QueryResponse: Contains the top-k results with metadata.
    """
    # Validate the top_k value
    if request.top_k < 1 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100.")

    # Generate query embedding
    embedding = generate_query_embedding(request.query)

    # Search the FAISS index
    indices = search_index(embedding, request.top_k)

    # Retrieve results
    results = retrieve_results(indices)

    return QueryResponse(results=results)


@app.get("/")
async def root():
    """
    Root endpoint to check API status.
    """
    return {"message": "Welcome to the FAISS Vector Search API!"}


# ---------------------------------------------------------------
# What We Did:
# ---------------------------------------------------------------
# - Set up a FastAPI service for querying the FAISS vector index.
# - Exposed an endpoint to accept query strings and return top-k results.
# - Integrated Sentence Transformers for real-time query embedding generation.
# - Returned metadata results with associated indices for easy interpretation.
# ---------------------------------------------------------------
# What's Next:
# ---------------------------------------------------------------
# - Add support for cosine similarity-based searches as an alternative to L2 distance.
# - Implement authentication for API access in production environments.
# - Optimize the FAISS index for incremental updates without full re-indexing.
# ---------------------------------------------------------------
