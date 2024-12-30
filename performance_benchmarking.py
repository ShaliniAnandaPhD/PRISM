import time
import statistics
from typing import List, Dict, Any
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from pydantic import BaseModel

class BenchmarkConfig(BaseModel):
    """
    Configuration for the performance benchmarking of the RAG pipeline.
    """
    vector_store_path: str  # Path to the FAISS vector store
    embedding_model_name: str  # Name of the embedding model for vector retrieval
    llm_model_name: str  # Name of the language model for generation
    top_k: int = 5  # Number of documents to retrieve
    test_queries: List[str]  # List of test queries

class PerformanceBenchmarking:
    """
    Benchmarks the performance of the Retrieval-Augmented Generation (RAG) pipeline.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmarking with retrieval and generation models.

        Args:
            config (BenchmarkConfig): Configuration for the benchmarking process.
        """
        self.config = config

        # Load embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model_name)
        print(f"Embedding model {config.embedding_model_name} loaded successfully.")

        # Load vector store
        self.vector_store = FAISS.load_local(config.vector_store_path, self.embeddings)
        print("FAISS vector store loaded successfully.")

        # Load language model
        self.llm = OpenAI(model=config.llm_model_name)
        print(f"Language model {config.llm_model_name} loaded successfully.")

        # Create the QA pipeline
        self.qa_chain = RetrievalQA.from_chain_type(
            retriever=self.vector_store.as_retriever(search_kwargs={"k": self.config.top_k}),
            chain_type="stuff",
            llm=self.llm,
            return_source_documents=True
        )

    def benchmark_latency(self) -> Dict[str, Any]:
        """
        Measure the latency for each query in the test set.

        Returns:
            Dict[str, Any]: Latency statistics including average, median, and individual latencies.
        """
        latencies = []
        for query in self.config.test_queries:
            start_time = time.time()
            self.qa_chain({"query": query})
            elapsed_time = time.time() - start_time
            latencies.append(elapsed_time)
            print(f"Query: '{query}' took {elapsed_time:.2f} seconds.")

        return {
            "average_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "latencies": latencies
        }

    def evaluate_accuracy(self, expected_answers: List[str]) -> float:
        """
        Evaluate the accuracy of the pipeline against expected answers.

        Args:
            expected_answers (List[str]): List of expected answers corresponding to the test queries.

        Returns:
            float: Accuracy as a percentage of correct responses.
        """
        correct_count = 0
        for query, expected in zip(self.config.test_queries, expected_answers):
            result = self.qa_chain({"query": query})
            generated_answer = result["result"].strip().lower()
            if expected.strip().lower() in generated_answer:
                correct_count += 1

        accuracy = (correct_count / len(expected_answers)) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy

    def measure_consensus_reliability(self) -> Dict[str, Any]:
        """
        Measure consensus reliability by analyzing the overlap of retrieved documents.

        Returns:
            Dict[str, Any]: Consensus statistics including overlap percentage and document count.
        """
        overlaps = []
        for query in self.config.test_queries:
            result = self.qa_chain({"query": query})
            sources = [doc.page_content for doc in result["source_documents"]]
            overlap = len(set(sources)) / len(sources)
            overlaps.append(overlap * 100)

        return {
            "average_overlap": statistics.mean(overlaps),
            "overlaps": overlaps
        }

if __name__ == "__main__":
    """
    Entry point for the performance benchmarking script.

    What We Did:
    - Measured latency for test queries.
    - Evaluated accuracy against expected answers.
    - Analyzed consensus reliability based on document overlaps.

    What's Next:
    - Extend benchmarking to include scalability tests for larger datasets.
    - Visualize benchmarking results with interactive dashboards.
    - Incorporate multi-threaded testing for parallel performance evaluation.
    """
    # Configuration
    config = BenchmarkConfig(
        vector_store_path="./vector_store_faiss",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name="gpt-4",
        top_k=5,
        test_queries=[
            "What are the key elements of a contract?",
            "Explain the legal implications of a breach of contract.",
            "What is the purpose of a non-disclosure agreement?"
        ]
    )

    # Initialize benchmarking
    benchmarking = PerformanceBenchmarking(config)

    # Perform benchmarks
    latency_stats = benchmarking.benchmark_latency()
    print("Latency Stats:", latency_stats)

    # Example expected answers (replace with domain-specific ones)
    expected_answers = [
        "key elements include offer, acceptance, and consideration",
        "a breach of contract can lead to damages or specific performance",
        "the purpose is to protect confidential information"
    ]
    accuracy = benchmarking.evaluate_accuracy(expected_answers)

    consensus_stats = benchmarking.measure_consensus_reliability()
    print("Consensus Stats:", consensus_stats)
