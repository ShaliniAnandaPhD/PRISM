import os
import json
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel

class RAGConfig(BaseModel):
    """
    Configuration for the sophisticated Retrieval-Augmented Generation (RAG) pipeline.
    """
    database_path: str  # Path to the document database
    vector_store_path: str  # Path to save/load FAISS vector store
    embedding_model_name: str  # Name of the embedding model for vectorization
    llm_model_name: str  # Name of the language model for generation
    top_k: int = 5  # Number of top documents to retrieve

class SophisticatedRAGPipeline:
    """
    Implements a sophisticated RAG pipeline using LangChain with FAISS and OpenAI.
    """

    def __init__(self, config: RAGConfig):
        """
        Initialize the RAG pipeline with embeddings, vector store, and LLM.

        Args:
            config (RAGConfig): Configuration for the pipeline.
        """
        self.config = config

        # Load embedding model for retrieval
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model_name)
        print(f"Embedding model {config.embedding_model_name} loaded successfully.")

        # Initialize or load FAISS vector store
        if os.path.exists(config.vector_store_path):
            self.vector_store = FAISS.load_local(config.vector_store_path, self.embeddings)
            print("FAISS vector store loaded successfully.")
        else:
            self.vector_store = None

        # Load language model for generation
        self.llm = OpenAI(model=config.llm_model_name)
        print(f"Language model {config.llm_model_name} loaded successfully.")

    def build_vector_store(self):
        """
        Build and save the FAISS vector store from the document database.
        """
        # Load documents
        with open(self.config.database_path, "r") as db_file:
            raw_documents = json.load(db_file)["documents"]

        # Preprocess and split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = []
        for doc in raw_documents:
            loader = TextLoader.from_text(doc)
            split_docs = text_splitter.split_documents(loader.load())
            docs.extend(split_docs)

        # Create and save FAISS vector store
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.vector_store.save_local(self.config.vector_store_path)
        print(f"FAISS vector store built and saved to {self.config.vector_store_path}.")

    def execute_pipeline(self, query: str) -> str:
        """
        Execute the RAG pipeline: retrieve, augment, and generate response.

        Args:
            query (str): User query.

        Returns:
            str: Final generated response.
        """
        # Define retrieval-based QA chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.config.top_k})
        qa_chain = RetrievalQA.from_chain_type(
            retriever=retriever,
            chain_type="stuff",
            llm=self.llm,
            return_source_documents=True
        )

        # Execute query and generate response
        result = qa_chain.run(query)
        print("Generated Response Successfully.")
        return result

if __name__ == "__main__":
    """
    Entry point for the sophisticated RAG pipeline script.

    What We Did:
    - Integrated LangChain for advanced RAG functionality.
    - Used FAISS for scalable and efficient vector search.
    - Applied HuggingFace embeddings and OpenAI LLM for generation.

    What's Next:
    - Add feedback loops to refine vector retrieval and generation results.
    - Evaluate the pipeline on domain-specific datasets for benchmarks.
    - Explore integrating additional chain types for dynamic workflows.
    """
    # Configuration
    config = RAGConfig(
        database_path="./legal_documents.json",
        vector_store_path="./vector_store_faiss",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name="gpt-4",
        top_k=5
    )

    # Initialize RAG pipeline
    rag_pipeline = SophisticatedRAGPipeline(config)

    # Check and build vector store if necessary
    if not rag_pipeline.vector_store:
        print("Building vector store...")
        rag_pipeline.build_vector_store()

    # Test the RAG pipeline
    query = "What are the key clauses in a non-disclosure agreement?"
    response = rag_pipeline.execute_pipeline(query)
    print("Generated Response:", response)
