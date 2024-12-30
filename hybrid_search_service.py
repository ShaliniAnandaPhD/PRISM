from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from weaviate import Client
from typing import List

# Configuration for Weaviate
WEAVIATE_URL = "http://localhost:8080"
VECTOR_CLASS_NAME = "DocumentVector"

# Initialize FastAPI and Weaviate client
app = FastAPI(title="Hybrid Search API", version="1.0.0")
client = Client(WEAVIATE_URL)
print("Connected to Weaviate server.")

# Request and Response Models
class HybridSearchRequest(BaseModel):
    query: str  # Semantic query for vector search
    filters: dict = {}  # Optional filters for exact matches
    top_k: int = 5  # Number of results to retrieve


class HybridSearchResponse(BaseModel):
    results: List[dict]  # Retrieved documents with metadata


# Helper Functions
def hybrid_search(query: str, filters: dict, top_k: int) -> List[dict]:
    """
    Executes a hybrid search in Weaviate using a vector query and optional filters.

    Args:
        query (str): Semantic query for vector search.
        filters (dict): Filters for exact matches (e.g., {"field": "value"}).
        top_k (int): Number of results to retrieve.

    Returns:
        List[dict]: List of retrieved documents with metadata.
    """
    # Construct the hybrid query
    search_query = {
        "hybrid": {
            "query": query,
            "alpha": 0.75,  # Weight between vector and keyword search
        },
        "limit": top_k,
    }

    # Add filters if provided
    if filters:
        search_query["where"] = {"path": list(filters.keys()), "operator": "Equal", "valueString": list(filters.values())[0]}

    # Perform the search
    response = client.query.get(VECTOR_CLASS_NAME, ["*"]).with_hybrid(**search_query).do()
    results = response.get("data", {}).get("Get", {}).get(VECTOR_CLASS_NAME, [])
    print(f"Hybrid search returned {len(results)} results.")
    return results


@app.post("/hybrid-search", response_model=HybridSearchResponse)
async def perform_hybrid_search(request: HybridSearchRequest):
    """
    Endpoint to perform hybrid search combining vector and keyword filters.

    Args:
        request (HybridSearchRequest): Contains the semantic query, filters, and number of results.

    Returns:
        HybridSearchResponse: Retrieved documents with metadata.
    """
    if request.top_k < 1 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100.")

    results = hybrid_search(request.query, request.filters, request.top_k)
    return HybridSearchResponse(results=results)


@app.get("/")
async def root():
    """
    Root endpoint for API status check.
    """
    return {"message": "Welcome to the Hybrid Search API!"}


# ---------------------------------------------------------------
# What We Did:
# ---------------------------------------------------------------
# - Set up a FastAPI service for hybrid search combining vector similarity and keyword filtering.
# - Integrated Weaviate’s hybrid query functionality for flexible and accurate retrieval.
# - Added endpoint validation to ensure proper usage of query parameters.
# ---------------------------------------------------------------
# What's Next:
# ---------------------------------------------------------------
# - Add support for advanced filtering with multiple conditions (e.g., AND/OR logic).
# - Optimize performance for large-scale datasets with batch query execution.
# - Create a UI dashboard for querying the hybrid search API interactively.
# ---------------------------------------------------------------
