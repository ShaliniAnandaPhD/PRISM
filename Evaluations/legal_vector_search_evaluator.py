"""
CAPABILITIES:
- Evaluates vector-based search performance for legal document retrieval
- Measures response time, accuracy, and relevance of document matches
- Calculates precision, recall, and F1 scores for search quality assessment
- Generates comprehensive evaluation reports in JSON format
- Provides detailed logging for audit compliance and debugging
- Supports batch testing of multiple query vectors
- Compatible with any vector index implementation with standardized interface
"""

import logging
import json
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


# Configure logging for auditability and compliance
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_filename = f"{log_directory}/vector_search_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
logging.getLogger().addHandler(console_handler)


@dataclass
class QueryResult:
    """Data class for storing individual query evaluation results."""
    query_id: str
    passed: bool
    retrieved_document_id: Optional[str] = None
    expected_document_id: Optional[str] = None
    response_time: float = 0.0
    similarity_score: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class EvaluationReport:
    """Data class for storing complete evaluation results."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_response_time: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    query_results: List[QueryResult] = field(default_factory=list)
    
    def calculate_metrics(self) -> None:
        """Calculate evaluation metrics based on query results."""
        if not self.query_results:
            return
            
        # Calculate basic metrics
        self.total_queries = len(self.query_results)
        self.successful_queries = sum(1 for result in self.query_results if result.passed)
        self.failed_queries = self.total_queries - self.successful_queries
        
        # Calculate average response time for successful queries
        response_times = [r.response_time for r in self.query_results if hasattr(r, 'response_time') and r.response_time is not None]
        self.average_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Calculate precision, recall, and F1 score (for basic exact match cases)
        if self.total_queries > 0:
            self.precision = self.successful_queries / self.total_queries
            self.recall = self.successful_queries / self.total_queries  # Same as precision for exact matches
            
            if (self.precision + self.recall) > 0:
                self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        result = asdict(self)
        result['query_results'] = [q.to_dict() for q in self.query_results]
        return result
    
    def save_to_file(self, filename: str) -> None:
        """Save evaluation report to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    def generate_visualizations(self, output_dir: str = 'reports') -> None:
        """Generate visualization charts for the evaluation results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Response time distribution
        plt.figure(figsize=(10, 6))
        response_times = [r.response_time for r in self.query_results if r.response_time is not None]
        plt.hist(response_times, bins=10, alpha=0.7)
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Number of Queries')
        plt.savefig(f"{output_dir}/response_time_distribution.png")
        
        # Success rate pie chart
        plt.figure(figsize=(8, 8))
        plt.pie([self.successful_queries, self.failed_queries], 
                labels=['Successful', 'Failed'],
                autopct='%1.1f%%', 
                colors=['#4CAF50', '#F44336'])
        plt.title('Query Success Rate')
        plt.savefig(f"{output_dir}/success_rate.png")


def calculate_cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    return float(cosine_similarity([vector1], [vector2])[0][0])


def test_query_vector_retrieval(
    vector_index: Callable[[np.ndarray], str],
    query_vectors: List[np.ndarray],
    expected_documents: List[str],
    query_ids: Optional[List[str]] = None,
    document_vectors: Optional[Dict[str, np.ndarray]] = None
) -> EvaluationReport:
    """
    Evaluates how well the vector index retrieves relevant legal documents.
    
    Args:
        vector_index: Function that takes a query vector and returns document ID
        query_vectors: List of query vectors to test
        expected_documents: List of expected document identifiers
        query_ids: Optional list of query identifiers
        document_vectors: Optional dictionary of document vectors for similarity scoring
        
    Returns:
        EvaluationReport with detailed results
    """
    if len(query_vectors) != len(expected_documents):
        raise ValueError("Length of query_vectors must match length of expected_documents")
        
    if query_ids is None:
        query_ids = [f"Query_{i}" for i in range(len(query_vectors))]
        
    report = EvaluationReport()
    
    logging.info(f"Starting evaluation with {len(query_vectors)} queries")
    
    for idx, query_vec in enumerate(query_vectors):
        query_id = query_ids[idx]
        expected_doc = expected_documents[idx]
        
        result = QueryResult(
            query_id=query_id,
            passed=False,
            expected_document_id=expected_doc
        )
        
        try:
            # Measure response time
            start_time = time.perf_counter()
            retrieved_doc = vector_index(query_vec)
            end_time = time.perf_counter()
            
            # Record results
            result.response_time = round(end_time - start_time, 4)
            result.retrieved_document_id = retrieved_doc
            result.passed = retrieved_doc == expected_doc
            
            # Calculate similarity if document vectors are provided
            if document_vectors and retrieved_doc in document_vectors:
                result.similarity_score = calculate_cosine_similarity(
                    query_vec, document_vectors[retrieved_doc]
                )
            
            # Log result
            status = "PASS" if result.passed else "FAIL"
            log_msg = f"{status}: {query_id} | Expected: {expected_doc}, Got: {retrieved_doc}"
            logging.info(log_msg)
            
        except Exception as e:
            # Handle exceptions
            error_msg = str(e)
            result.error = error_msg
            logging.error(f"ERROR: {query_id} | Exception: {error_msg}")
        
        # Add to report
        report.query_results.append(result)
    
    # Calculate overall metrics
    report.calculate_metrics()
    
    logging.info(f"Evaluation complete. Success rate: {report.precision:.2%}")
    
    return report


def evaluate_index_at_scale(
    vector_index: Callable[[np.ndarray], str],
    document_vectors: Dict[str, np.ndarray],
    num_queries: int = 100,
    noise_level: float = 0.05
) -> EvaluationReport:
    """
    Tests vector index performance at scale with noisy variants of document vectors.
    
    Args:
        vector_index: Function that takes a query vector and returns document ID
        document_vectors: Dictionary mapping document IDs to their vector representations
        num_queries: Number of test queries to generate
        noise_level: Amount of noise to add to create query variants
        
    Returns:
        EvaluationReport with detailed results
    """
    query_vectors = []
    expected_documents = []
    query_ids = []
    
    # Generate test queries by adding noise to document vectors
    doc_ids = list(document_vectors.keys())
    samples_per_doc = max(1, num_queries // len(doc_ids))
    
    for doc_id, doc_vector in document_vectors.items():
        for i in range(samples_per_doc):
            # Add random noise to create a similar but not identical vector
            noise = np.random.normal(0, noise_level, size=doc_vector.shape)
            noisy_vector = doc_vector + noise
            
            # Normalize to unit length
            noisy_vector = noisy_vector / np.linalg.norm(noisy_vector)
            
            query_vectors.append(noisy_vector)
            expected_documents.append(doc_id)
            query_ids.append(f"{doc_id}_variant_{i}")
    
    # Run evaluation
    return test_query_vector_retrieval(
        vector_index=vector_index,
        query_vectors=query_vectors[:num_queries],
        expected_documents=expected_documents[:num_queries],
        query_ids=query_ids[:num_queries],
        document_vectors=document_vectors
    )


# Example vector index implementation (replace with actual implementation)
def example_vector_index(query_vector: np.ndarray) -> str:
    """
    Simulated function returning closest matching document based on cosine similarity.
    
    Args:
        query_vector: The query vector to match against the document index
        
    Returns:
        Document ID of the best matching document
    """
    document_vectors = {
        "doc1": np.array([0.1, 0.2, 0.3]),
        "doc2": np.array([0.3, 0.1, 0.4]),
        "doc3": np.array([0.5, 0.5, 0.1]),
        "doc4": np.array([0.2, 0.3, 0.5]),
    }
    
    # Find document with highest cosine similarity
    best_match = max(
        document_vectors.keys(),
        key=lambda doc: calculate_cosine_similarity(query_vector, document_vectors[doc])
    )
    
    return best_match


# Main execution block
if __name__ == "__main__":
    # Define test queries and expected results
    query_vectors = [
        np.array([0.1, 0.2, 0.3]),  # Should match doc1
        np.array([0.3, 0.1, 0.4]),  # Should match doc2
        np.array([0.5, 0.5, 0.1]),  # Should match doc3
        np.array([0.2, 0.3, 0.5]),  # Should match doc4
    ]
    
    expected_documents = ["doc1", "doc2", "doc3", "doc4"]
    
    # Sample document vectors for similarity comparison
    document_vectors = {
        "doc1": np.array([0.1, 0.2, 0.3]),
        "doc2": np.array([0.3, 0.1, 0.4]),
        "doc3": np.array([0.5, 0.5, 0.1]),
        "doc4": np.array([0.2, 0.3, 0.5]),
    }
    
    # Run basic evaluation
    print("Running basic vector index evaluation...")
    basic_report = test_query_vector_retrieval(
        example_vector_index,
        query_vectors,
        expected_documents,
        document_vectors=document_vectors
    )
    
    # Run scale evaluation
    print("Running scale evaluation with noisy queries...")
    scale_report = evaluate_index_at_scale(
        example_vector_index,
        document_vectors,
        num_queries=20,
        noise_level=0.1
    )
    
    # Save reports
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    basic_report.save_to_file(f"{reports_dir}/basic_evaluation_report.json")
    scale_report.save_to_file(f"{reports_dir}/scale_evaluation_report.json")
    
    # Generate visualizations
    basic_report.generate_visualizations(f"{reports_dir}/basic")
    scale_report.generate_visualizations(f"{reports_dir}/scale")
    
    print(f"Evaluation complete. Check '{reports_dir}' directory for reports and visualizations.")
    print(f"Basic evaluation success rate: {basic_report.precision:.2%}")
    print(f"Scale evaluation success rate: {scale_report.precision:.2%}")


"""
TODO:
- Implement real-time performance monitoring capabilities
- Add multi-threading support for parallel query evaluation
- Include advanced metrics like Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG)
- Support evaluation of top-k retrieval performance
- Add benchmarking against baseline retrieval methods
- Implement cross-validation for more robust evaluations
- Extend test suite to handle different vector embedding models
- Add support for evaluating semantic similarity beyond exact matches
"""
