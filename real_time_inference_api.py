from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

class InferenceConfig(BaseModel):
    """
    Configuration for the real-time inference API.
    """
    vector_store_path: str  # Path to the FAISS vector store
    embedding_model_name: str  # Name of the embedding model for vector retrieval
    llm_model_name: str  # Name of the language model for generation
    top_k: int = 5  # Number of documents to retrieve

# Initialize FastAPI app
app = FastAPI()

# Global pipeline and configuration
pipeline_config = None
retriever = None
qa_chain = None

@app.on_event("startup")
def initialize_pipeline():
    """
    Initialize the Retrieval-Augmented Generation pipeline during startup.
    """
    global pipeline_config, retriever, qa_chain

    # Load configuration
    pipeline_config = InferenceConfig(
        vector_store_path="./vector_store_faiss",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name="gpt-4",
        top_k=5
    )

    # Load vector store and embeddings
    embeddings = HuggingFaceEmbeddings(model_name=pipeline_config.embedding_model_name)
    vector_store = FAISS.load_local(pipeline_config.vector_store_path, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": pipeline_config.top_k})

    # Load LLM for generation
    llm = OpenAI(model=pipeline_config.llm_model_name)
    qa_chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        chain_type="stuff",
        llm=llm,
        return_source_documents=True
    )

    print("Pipeline initialized successfully.")

class QueryRequest(BaseModel):
    """
    Model for query requests.
    """
    query: str  # The input query from the user

class QueryResponse(BaseModel):
    """
    Model for query responses.
    """
    query: str  # The input query
    response: str  # The generated response
    sources: list  # List of source documents used for the response

@app.post("/inference", response_model=QueryResponse)
def infer(query_request: QueryRequest):
    """
    Endpoint for real-time inference.

    Args:
        query_request (QueryRequest): The input query request.

    Returns:
        QueryResponse: The generated response and source documents.
    """
    global qa_chain

    if not qa_chain:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized.")

    # Run the RAG pipeline
    result = qa_chain({"query": query_request.query})

    return QueryResponse(
        query=query_request.query,
        response=result["result"],
        sources=[doc.page_content for doc in result["source_documents"]]
    )

@app.get("/healthcheck")
def healthcheck():
    """
    Healthcheck endpoint to ensure the service is running.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    """
    Entry point for the real-time inference API.

    What We Did:
    - Initialized a FastAPI app for serving real-time queries.
    - Integrated a Retrieval-Augmented Generation pipeline using LangChain.
    - Created endpoints for query inference and health checks.

    What's Next:
    - Add authentication and rate-limiting for secure API access.
    - Enable logging and monitoring to track API usage and performance.
    - Extend support for batch queries or streaming outputs.
    """
    import uvicorn
    uvicorn.run("real_time_inference_api:app", host="0.0.0.0", port=8000, reload=True)
