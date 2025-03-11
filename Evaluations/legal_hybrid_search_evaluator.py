"""
CAPABILITIES:
- Evaluates hybrid search systems combining vector and keyword search
- Measures precision, recall, F1-score, and ranking quality across search results
- Conducts load testing with configurable concurrency and query volumes
- Assesses search quality across different legal domains and query types
- Compares performance of hybrid search against keyword-only and vector-only approaches
- Benchmarks against gold standard datasets with known relevant documents
- Visualizes search performance metrics with detailed analytics
- Identifies optimization opportunities and failure modes
- Supports customizable relevance thresholds and evaluation criteria
- Integrates with existing search infrastructure through adaptable interfaces
"""

import logging
import json
import time
import os
import random
import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Set, Callable, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm


# Configure logging with structured output
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{log_dir}/hybrid_search_evaluation_{timestamp}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

# Add console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger("hybrid_search_evaluator")


@dataclass
class SearchResult:
    """Single search result with document ID and relevance score."""
    doc_id: str
    score: float = 1.0
    source: str = "hybrid"  # "hybrid", "vector", "keyword"
    snippet: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Complete result set for a single query."""
    query: str
    retrieved_docs: List[SearchResult]
    expected_docs: List[str]
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    ndcg: float = 0.0
    response_time_sec: float = 0.0
    query_category: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "query": self.query,
            "retrieved_docs": [
                {
                    "doc_id": r.doc_id, 
                    "score": r.score,
                    "source": r.source
                } 
                for r in self.retrieved_docs
            ],
            "expected_docs": self.expected_docs,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "ndcg": self.ndcg,
            "response_time_sec": self.response_time_sec
        }
        
        if self.query_category:
            result["query_category"] = self.query_category
            
        if self.error:
            result["error"] = self.error
            
        return result


@dataclass
class EvaluationSummary:
    """Summary statistics for search evaluation."""
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_f1_score: float = 0.0
    avg_ndcg: float = 0.0
    avg_response_time: float = 0.0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    categories: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Comparison between hybrid, vector, and keyword search."""
    hybrid_metrics: EvaluationSummary
    vector_metrics: Optional[EvaluationSummary] = None
    keyword_metrics: Optional[EvaluationSummary] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "hybrid_metrics": self.hybrid_metrics.to_dict(),
        }
        
        if self.vector_metrics:
            result["vector_metrics"] = self.vector_metrics.to_dict()
            
        if self.keyword_metrics:
            result["keyword_metrics"] = self.keyword_metrics.to_dict()
            
        return result


@dataclass
class LoadTestResult:
    """Results from load testing the search system."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_response_time: float = 0.0
    throughput_qps: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    concurrency_level: int = 1
    test_duration_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class HybridSearchEvaluator:
    """
    Evaluator for hybrid search systems combining vector and keyword search approaches.
    Provides comprehensive evaluation metrics and visualizations.
    """
    
    def __init__(
        self,
        output_dir: str = "evaluation_results",
        save_individual_results: bool = True
    ):
        """
        Initialize the hybrid search evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
            save_individual_results: Whether to save results for each query
        """
        self.output_dir = output_dir
        self.save_individual_results = save_individual_results
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Store results
        self.results: List[QueryResult] = []
    
    def evaluate_search(
        self,
        search_function: Callable[[str], List[Union[str, SearchResult]]],
        queries: List[str],
        expected_results: List[List[str]],
        query_categories: Optional[List[str]] = None,
        relevance_grades: Optional[Dict[str, Dict[str, int]]] = None,
        max_results: int = 10
    ) -> EvaluationSummary:
        """
        Evaluate a search function across a set of test queries.
        
        Args:
            search_function: Function that takes a query string and returns a list of results
            queries: List of test query strings
            expected_results: List of expected relevant document IDs for each query
            query_categories: Optional list of categories for each query
            relevance_grades: Optional dict mapping query to dict of doc_id -> relevance grade
            max_results: Maximum number of results to consider
            
        Returns:
            EvaluationSummary with aggregated metrics
        """
        logger.info(f"Starting evaluation with {len(queries)} queries")
        
        # Reset results
        self.results = []
        
        # Process each query
        for idx, query in enumerate(tqdm(queries, desc="Evaluating queries")):
            expected_docs = expected_results[idx]
            category = query_categories[idx] if query_categories else None
            
            try:
                # Measure search time
                start_time = time.perf_counter()
                retrieved_docs = search_function(query)
                end_time = time.perf_counter()
                response_time = end_time - start_time
                
                # Process results to standardized format
                if retrieved_docs and not isinstance(retrieved_docs[0], SearchResult):
                    # Convert simple doc_id strings to SearchResult objects
                    retrieved_results = [
                        SearchResult(doc_id=doc_id) 
                        for doc_id in retrieved_docs[:max_results]
                    ]
                else:
                    # Already in SearchResult format
                    retrieved_results = retrieved_docs[:max_results]
                
                # Calculate metrics
                retrieved_doc_ids = [r.doc_id for r in retrieved_results]
                retrieved_set = set(retrieved_doc_ids)
                expected_set = set(expected_docs)
                
                # Calculate basic metrics
                precision = len(retrieved_set & expected_set) / len(retrieved_set) if retrieved_set else 0
                recall = len(retrieved_set & expected_set) / len(expected_set) if expected_set else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Calculate NDCG if relevance grades are provided
                ndcg = 0.0
                if relevance_grades and query in relevance_grades:
                    ndcg = self._calculate_ndcg(
                        retrieved_doc_ids,
                        relevance_grades[query],
                        k=len(retrieved_doc_ids)
                    )
                
                # Create query result
                result = QueryResult(
                    query=query,
                    retrieved_docs=retrieved_results,
                    expected_docs=expected_docs,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    ndcg=ndcg,
                    response_time_sec=response_time,
                    query_category=category
                )
                
                # Log result
                log_msg = (
                    f"Query '{query}': "
                    f"Precision={precision:.2f}, Recall={recall:.2f}, "
                    f"F1={f1:.2f}, NDCG={ndcg:.2f}, "
                    f"Time={response_time:.4f}s"
                )
                logger.info(log_msg)
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"ERROR: Query '{query}' | Exception: {error_msg}")
                
                # Create error result
                result = QueryResult(
                    query=query,
                    retrieved_docs=[],
                    expected_docs=expected_docs,
                    query_category=category,
                    error=error_msg
                )
            
            # Store result
            self.results.append(result)
        
        # Calculate summary statistics
        summary = self._calculate_summary()
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def compare_search_methods(
        self,
        hybrid_search: Callable[[str], List[Union[str, SearchResult]]],
        vector_search: Optional[Callable[[str], List[Union[str, SearchResult]]]] = None,
        keyword_search: Optional[Callable[[str], List[Union[str, SearchResult]]]] = None,
        queries: List[str],
        expected_results: List[List[str]],
        query_categories: Optional[List[str]] = None
    ) -> ComparisonResult:
        """
        Compare performance between hybrid, vector-only, and keyword-only search.
        
        Args:
            hybrid_search: Hybrid search function
            vector_search: Vector search function (optional)
            keyword_search: Keyword search function (optional)
            queries: List of test query strings
            expected_results: List of expected relevant document IDs for each query
            query_categories: Optional list of categories for each query
            
        Returns:
            ComparisonResult with metrics for each search method
        """
        logger.info(f"Comparing search methods on {len(queries)} queries")
        
        # Evaluate hybrid search
        logger.info("Evaluating hybrid search...")
        hybrid_metrics = self.evaluate_search(
            hybrid_search, queries, expected_results, query_categories
        )
        
        # Save hybrid results
        hybrid_results = self.results.copy()
        
        # Evaluate vector search if provided
        vector_metrics = None
        if vector_search:
            logger.info("Evaluating vector search...")
            vector_metrics = self.evaluate_search(
                vector_search, queries, expected_results, query_categories
            )
        
        # Save vector results
        vector_results = self.results.copy() if vector_search else []
        
        # Evaluate keyword search if provided
        keyword_metrics = None
        if keyword_search:
            logger.info("Evaluating keyword search...")
            keyword_metrics = self.evaluate_search(
                keyword_search, queries, expected_results, query_categories
            )
        
        # Save keyword results
        keyword_results = self.results.copy() if keyword_search else []
        
        # Create comparison result
        comparison = ComparisonResult(
            hybrid_metrics=hybrid_metrics,
            vector_metrics=vector_metrics,
            keyword_metrics=keyword_metrics
        )
        
        # Generate comparison visualizations
        self._generate_comparison_charts(
            comparison,
            hybrid_results,
            vector_results,
            keyword_results
        )
        
        # Save comparison results
        comparison_path = os.path.join(self.output_dir, "search_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison.to_dict(), f, indent=4)
        
        logger.info(f"Comparison results saved to {comparison_path}")
        
        return comparison
    
    def run_load_test(
        self,
        search_function: Callable[[str], List[Union[str, SearchResult]]],
        queries: List[str],
        concurrency: int = 4,
        query_count: int = 100,
        max_duration_sec: int = 60
    ) -> LoadTestResult:
        """
        Test search performance under load with concurrent queries.
        
        Args:
            search_function: Search function to test
            queries: Pool of queries to select from
            concurrency: Number of concurrent queries
            query_count: Total number of queries to run
            max_duration_sec: Maximum duration in seconds
            
        Returns:
            LoadTestResult with performance metrics
        """
        logger.info(f"Running load test with concurrency={concurrency}, query_count={query_count}")
        
        # Generate query workload (randomly selecting from provided queries)
        workload = [random.choice(queries) for _ in range(query_count)]
        
        # Track results
        successful = 0
        failed = 0
        response_times = []
        
        # Function to execute for each query
        def execute_query(query: str) -> Tuple[bool, float]:
            try:
                start_time = time.perf_counter()
                _ = search_function(query)
                end_time = time.perf_counter()
                return True, end_time - start_time
            except Exception as e:
                logger.error(f"Load test query error: {str(e)}")
                return False, 0
        
        # Run queries with thread pool
        overall_start = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all tasks
            future_to_query = {
                executor.submit(execute_query, query): query 
                for query in workload
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    success, response_time = future.result()
                    if success:
                        successful += 1
                        response_times.append(response_time)
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Load test error: {str(e)}")
                    failed += 1
                
                # Check if we've exceeded max duration
                if time.perf_counter() - overall_start > max_duration_sec:
                    for f in future_to_query:
                        f.cancel()
                    break
        
        overall_end = time.perf_counter()
        test_duration = overall_end - overall_start
        
        # Calculate metrics
        avg_response_time = np.mean(response_times) if response_times else 0
        throughput = len(response_times) / test_duration if test_duration > 0 else 0
        
        # Calculate percentiles
        percentiles = {}
        if response_times:
            percentiles = {
                "p50": np.percentile(response_times, 50),
                "p90": np.percentile(response_times, 90),
                "p95": np.percentile(response_times, 95),
                "p99": np.percentile(response_times, 99)
            }
        else:
            percentiles = {"p50": 0, "p90": 0, "p95": 0, "p99": 0}
        
        # Create result
        result = LoadTestResult(
            total_queries=successful + failed,
            successful_queries=successful,
            failed_queries=failed,
            avg_response_time=avg_response_time,
            throughput_qps=throughput,
            p50_latency=percentiles["p50"],
            p90_latency=percentiles["p90"],
            p95_latency=percentiles["p95"],
            p99_latency=percentiles["p99"],
            concurrency_level=concurrency,
            test_duration_sec=test_duration
        )
        
        # Generate load test visualization
        self._generate_load_test_charts(result, response_times)
        
        # Save load test results
        load_test_path = os.path.join(self.output_dir, "load_test_results.json")
        with open(load_test_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=4)
        
        logger.info(f"Load test results saved to {load_test_path}")
        
        return result
    
    def _calculate_ndcg(
        self, 
        retrieved_docs: List[str], 
        relevance_grades: Dict[str, int],
        k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        Args:
            retrieved_docs: List of retrieved document IDs in order
            relevance_grades: Dict mapping doc_id to relevance grade (higher is better)
            k: Cutoff for calculation
            
        Returns:
            NDCG score
        """
        # Get relevance for retrieved docs
        relevance = [
            relevance_grades.get(doc_id, 0) 
            for doc_id in retrieved_docs[:k]
        ]
        
        # Calculate DCG
        dcg = 0
        for i, rel in enumerate(relevance):
            dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 because i starts at 0
        
        # Calculate ideal DCG
        ideal_order = sorted([
            relevance_grades.get(doc_id, 0) 
            for doc_id in relevance_grades.keys()
        ], reverse=True)
        
        idcg = 0
        for i, rel in enumerate(ideal_order[:k]):
            idcg += (2 ** rel - 1) / np.log2(i + 2)
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0
    
    def _calculate_summary(self) -> EvaluationSummary:
        """Calculate summary statistics from individual query results."""
        if not self.results:
            return EvaluationSummary()
        
        # Basic counts
        total_queries = len(self.results)
        failed_queries = sum(1 for r in self.results if r.error)
        successful_queries = total_queries - failed_queries
        
        # Filter successful results for metric calculations
        successful_results = [r for r in self.results if not r.error]
        
        if not successful_results:
            return EvaluationSummary(
                total_queries=total_queries,
                successful_queries=successful_queries,
                failed_queries=failed_queries
            )
        
        # Calculate averages
        avg_precision = np.mean([r.precision for r in successful_results])
        avg_recall = np.mean([r.recall for r in successful_results])
        avg_f1 = np.mean([r.f1_score for r in successful_results])
        avg_ndcg = np.mean([r.ndcg for r in successful_results])
        avg_time = np.mean([r.response_time_sec for r in successful_results])
        
        # Calculate per-category metrics
        categories = {}
        category_results = defaultdict(list)
        
        for result in successful_results:
            if result.query_category:
                category_results[result.query_category].append(result)
        
        for category, results in category_results.items():
            categories[category] = {
                "avg_precision": np.mean([r.precision for r in results]),
                "avg_recall": np.mean([r.recall for r in results]),
                "avg_f1_score": np.mean([r.f1_score for r in results]),
                "avg_ndcg": np.mean([r.ndcg for r in results]),
                "count": len(results)
            }
        
        # Create summary
        return EvaluationSummary(
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1_score=avg_f1,
            avg_ndcg=avg_ndcg,
            avg_response_time=avg_time,
            total_queries=total_queries,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            categories=categories
        )
    
    def _save_results(self, summary: EvaluationSummary) -> None:
        """Save evaluation results to files."""
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.output_dir, f"evaluation_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Save summary
        summary_path = os.path.join(result_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=4)
        
        # Save individual results if requested
        if self.save_individual_results:
            results_path = os.path.join(result_dir, "query_results.json")
            with open(results_path, 'w') as f:
                results_dict = {
                    "results": [r.to_dict() for r in self.results]
                }
                json.dump(results_dict, f, indent=4)
        
        # Generate visualizations
        self._generate_summary_charts(summary, result_dir)
        
        logger.info(f"Evaluation results saved to {result_dir}")
    
    def _generate_summary_charts(self, summary: EvaluationSummary, output_dir: str) -> None:
        """Generate visualization charts from evaluation summary."""
        # Create a DataFrame with metrics for plotting
        metrics = {
            "Precision": summary.avg_precision,
            "Recall": summary.avg_recall,
            "F1 Score": summary.avg_f1_score,
            "NDCG": summary.avg_ndcg
        }
        
        # Bar chart of metrics
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values(), color='skyblue')
        plt.ylim(0, 1.0)
        plt.title(f"Search Performance Metrics (n={summary.successful_queries} queries)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, (k, v) in enumerate(metrics.items()):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_metrics.png"))
        plt.close()


# Example search functions (replace with actual implementations)
class HybridSearchService:
    """Example implementation of a hybrid search service."""
    
    def __init__(self, keyword_weight: float = 0.3, vector_weight: float = 0.7):
        """
        Initialize the hybrid search service.
        
        Args:
            keyword_weight: Weight to give keyword search results (0-1)
            vector_weight: Weight to give vector search results (0-1)
        """
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        
        # Mock document database with titles and content
        self.documents = {
            "doc1": {
                "title": "Contract Law Fundamentals",
                "content": "This document covers the basics of contract law including formation, consideration, and breach.",
                "vector": np.array([0.1, 0.2, 0.3])  # Mock vector embedding
            },
            "doc2": {
                "title": "Negligence in Personal Injury Cases",
                "content": "Overview of negligence principles in personal injury litigation including duty of care and causation.",
                "vector": np.array([0.3, 0.1, 0.2])
            },
            "doc3": {
                "title": "Commercial Contract Drafting",
                "content": "Best practices for drafting commercial contracts and avoiding common pitfalls.",
                "vector": np.array([0.15, 0.25, 0.35])
            },
            "doc4": {
                "title": "Medical Malpractice and Negligence",
                "content": "Analysis of negligence standards in medical malpractice cases and recent precedents.",
                "vector": np.array([0.35, 0.15, 0.25])
            },
            "doc5": {
                "title": "Property Law Overview",
                "content": "Comprehensive guide to property law including real estate, landlord-tenant, and easements.",
                "vector": np.array([0.4, 0.5, 0.1])
            }
        }
    
    def keyword_search(self, query: str) -> List[SearchResult]:
        """Simple keyword-based search implementation."""
        results = []
        query_terms = query.lower().split()
        
        for doc_id, doc in self.documents.items():
            # Count how many query terms appear in title and content
            title_matches = sum(term in doc["title"].lower() for term in query_terms)
            content_matches = sum(term in doc["content"].lower() for term in query_terms)
            
            # Calculate a simple relevance score
            score = (title_matches * 2 + content_matches) / (len(query_terms) * 3)
            
            if score > 0:
                results.append(SearchResult(
                    doc_id=doc_id,
                    score=score,
                    source="keyword",
                    snippet=doc["title"]
                ))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def vector_search(self, query: str) -> List[SearchResult]:
        """Simple vector-based search implementation."""
        # Mock query vector - in a real system, this would use an embedding model
        query_vector = np.array([
            sum(ord(c) % 3 for c in query) / 100,
            sum(ord(c) % 5 for c in query) / 100,
            sum(ord(c) % 7 for c in query) / 100
        ])
        
        results = []
        
        for doc_id, doc in self.documents.items():
            # Calculate cosine similarity
            doc_vector = doc["vector"]
            similarity = np.dot(query_vector, doc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            )
            
            results.append(SearchResult(
                doc_id=doc_id,
                score=float(similarity),
                source="vector",
                snippet=doc["title"]
            ))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def hybrid_search(self, query: str) -> List[SearchResult]:
        """Combine keyword and vector search results."""
        keyword_results = {r.doc_id: r for r in self.keyword_search(query)}
        vector_results = {r.doc_id: r for r in self.vector_search(query)}
        
        # Combine results
        combined_results = {}
        all_doc_ids = set(keyword_results.keys()) | set(vector_results.keys())
        
        for doc_id in all_doc_ids:
            keyword_score = keyword_results[doc_id].score if doc_id in keyword_results else 0
            vector_score = vector_results[doc_id].score if doc_id in vector_results else 0
            
            # Calculate combined score
            combined_score = (
                self.keyword_weight * keyword_score + 
                self.vector_weight * vector_score
            )
            
            combined_results[doc_id] = SearchResult(
                doc_id=doc_id,
                score=combined_score,
                source="hybrid",
                snippet=self.documents[doc_id]["title"]
            )
        
        # Sort by score
        results = list(combined_results.values())
        results.sort(key=lambda x: x.score, reverse=True)
        return results


def create_test_queries() -> Tuple[List[str], List[List[str]], List[str], Dict[str, Dict[str, int]]]:
    """Create test data for evaluating search."""
    queries = [
        "contract law",
        "negligence case",
        "property dispute",
        "medical malpractice",
        "commercial agreement",
        "landlord tenant issues",
        "personal injury",
        "drafting contracts"
    ]
    
    expected_results = [
        ["doc1", "doc3"],  # contract law
        ["doc2", "doc4"],  # negligence case
        ["doc5"],          # property dispute
        ["doc4"],          # medical malpractice
        ["doc3"],          # commercial agreement
        ["doc5"],          # landlord tenant issues
        ["doc2"],          # personal injury
        ["doc3"]           # drafting contracts
    ]
    
    categories = [
        "contract",
        "tort",
        "property",
        "tort",
        "contract",
        "property",
        "tort",
        "contract"
    ]
    
    # Relevance grades (0-3 scale, higher is more relevant)
    relevance_grades = {
        "contract law": {"doc1": 3, "doc3": 2, "doc5": 1},
        "negligence case": {"doc2": 3, "doc4": 2},
        "property dispute": {"doc5": 3, "doc1": 1},
        "medical malpractice": {"doc4": 3, "doc2": 2},
        "commercial agreement": {"doc3": 3, "doc1": 2},
        "landlord tenant issues": {"doc5": 3},
        "personal injury": {"doc2": 3, "doc4": 1},
        "drafting contracts": {"doc3": 3, "doc1": 2}
    }
    
    return queries, expected_results, categories, relevance_grades


# Main execution
if __name__ == "__main__":
    # Create search service
    search_service = HybridSearchService(keyword_weight=0.3, vector_weight=0.7)
    
    # Create test data
    queries, expected_results, categories, relevance_grades = create_test_queries()
    
    # Create evaluator
    evaluator = HybridSearchEvaluator(output_dir="hybrid_search_evaluation")
    
    # Run evaluation on hybrid search
    print("Evaluating hybrid search...")
    hybrid_metrics = evaluator.evaluate_search(
        search_service.hybrid_search,
        queries,
        expected_results,
        query_categories=categories,
        relevance_grades=relevance_grades
    )
    
    # Compare search methods
    print("\nComparing search methods...")
    comparison = evaluator.compare_search_methods(
        hybrid_search=search_service.hybrid_search,
        vector_search=search_service.vector_search,
        keyword_search=search_service.keyword_search,
        queries=queries,
        expected_results=expected_results,
        query_categories=categories
    )
    
    # Run load test
    print("\nRunning load test...")
    load_test_results = evaluator.run_load_test(
        search_service.hybrid_search,
        queries,
        concurrency=4,
        query_count=50
    )
    
    print("\nEvaluation complete!")
    print(f"Hybrid search F1 score: {hybrid_metrics.avg_f1_score:.3f}")
    print(f"Response time: {hybrid_metrics.avg_response_time:.3f} seconds")
    print(f"Load test throughput: {load_test_results.throughput_qps:.1f} queries/second")
    print(f"\nResults saved to: {evaluator.output_dir}")


"""
TODO:
- Implement industry-standard legal search benchmarks (e.g., COLIEE, legal TREC)
- Add support for relevance feedback and continuous improvement evaluation
- Implement explainability metrics for understanding search result ranking
- Extend comparison with commercial legal search systems
- Add citation verification for legal document references
- Support multi-jurisdictional legal search evaluation
- Incorporate personalization assessment based on user behavior
- Develop specialized metrics for case law similarity and statute relevance
- Integrate with legal ontologies for semantic evaluation
- Add document clustering visualization for search result analysis
"""
        
        # Category comparison if available
        if summary.categories:
            categories = list(summary.categories.keys())
            f1_scores = [summary.categories[cat]["avg_f1_score"] for cat in categories]
            precision = [summary.categories[cat]["avg_precision"] for cat in categories]
            recall = [summary.categories[cat]["avg_recall"] for cat in categories]
            
            plt.figure(figsize=(12, 7))
            x = range(len(categories))
            width = 0.25
            
            plt.bar([i - width for i in x], precision, width, label='Precision', color='skyblue')
            plt.bar(x, recall, width, label='Recall', color='lightgreen')
            plt.bar([i + width for i in x], f1_scores, width, label='F1 Score', color='salmon')
            
            plt.xlabel('Category')
            plt.ylabel('Score')
            plt.title('Performance by Category')
            plt.xticks(x, categories, rotation=45, ha='right')
            plt.ylim(0, 1.0)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, "category_performance.png"))
            plt.close()
    
    def _generate_comparison_charts(
        self,
        comparison: ComparisonResult,
        hybrid_results: List[QueryResult],
        vector_results: List[QueryResult],
        keyword_results: List[QueryResult]
    ) -> None:
        """Generate charts comparing different search methods."""
        # Create output directory
        comparison_dir = os.path.join(self.output_dir, "comparison_charts")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Get metrics for each method
        metrics = ["Precision", "Recall", "F1 Score", "NDCG"]
        methods = []
        values = []
        
        if comparison.hybrid_metrics:
            methods.append("Hybrid")
            values.append([
                comparison.hybrid_metrics.avg_precision,
                comparison.hybrid_metrics.avg_recall,
                comparison.hybrid_metrics.avg_f1_score,
                comparison.hybrid_metrics.avg_ndcg
            ])
        
        if comparison.vector_metrics:
            methods.append("Vector")
            values.append([
                comparison.vector_metrics.avg_precision,
                comparison.vector_metrics.avg_recall,
                comparison.vector_metrics.avg_f1_score,
                comparison.vector_metrics.avg_ndcg
            ])
        
        if comparison.keyword_metrics:
            methods.append("Keyword")
            values.append([
                comparison.keyword_metrics.avg_precision,
                comparison.keyword_metrics.avg_recall,
                comparison.keyword_metrics.avg_f1_score,
                comparison.keyword_metrics.avg_ndcg
            ])
        
        # Create grouped bar chart
        x = np.arange(len(metrics))
        width = 0.25
        
        plt.figure(figsize=(12, 7))
        
        for i, (method, vals) in enumerate(zip(methods, values)):
            offset = (i - len(methods) / 2 + 0.5) * width
            plt.bar(x + offset, vals, width, label=method)
        
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Search Method Comparison')
        plt.xticks(x, metrics)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "method_comparison.png"))
        plt.close()
        
        # Response time comparison
        response_times = []
        
        if hybrid_results:
            hybrid_times = [r.response_time_sec for r in hybrid_results if not r.error]
            response_times.append(("Hybrid", hybrid_times))
        
        if vector_results:
            vector_times = [r.response_time_sec for r in vector_results if not r.error]
            response_times.append(("Vector", vector_times))
        
        if keyword_results:
            keyword_times = [r.response_time_sec for r in keyword_results if not r.error]
            response_times.append(("Keyword", keyword_times))
        
        if response_times:
            plt.figure(figsize=(10, 6))
            plt.boxplot([times for _, times in response_times])
            plt.xlabel('Search Method')
            plt.ylabel('Response Time (seconds)')
            plt.title('Response Time Comparison')
            plt.xticks(range(1, len(response_times) + 1), [method for method, _ in response_times])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, "response_time_comparison.png"))
            plt.close()
    
    def _generate_load_test_charts(self, result: LoadTestResult, response_times: List[float]) -> None:
        """Generate charts for load test results."""
        # Create output directory
        load_test_dir = os.path.join(self.output_dir, "load_test_charts")
        os.makedirs(load_test_dir, exist_ok=True)
        
        # Response time histogram
        if response_times:
            plt.figure(figsize=(10, 6))
            plt.hist(response_times, bins=20, alpha=0.7, color='skyblue')
            plt.axvline(result.avg_response_time, color='red', linestyle='dashed', linewidth=2)
            plt.text(
                result.avg_response_time, 
                plt.ylim()[1] * 0.9, 
                f"Avg: {result.avg_response_time:.3f}s", 
                verticalalignment='top',
                horizontalalignment='right'
            )
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Count')
            plt.title(f'Response Time Distribution (n={result.successful_queries} queries)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(load_test_dir, "response_time_distribution.png"))
            plt.close()
        
        # Latency percentiles
        percentiles = {
            "p50": result.p50_latency,
            "p90": result.p90_latency,
            "p95": result.p95_latency,
            "p99": result.p99_latency
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(percentiles.keys(), percentiles.values(), color='lightgreen')
        plt.xlabel('Percentile')
        plt.ylabel('Response Time (seconds)')
        plt.title('Response Time Percentiles')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, (k, v) in enumerate(percentiles.items()):
            plt.text(i, v + 0.002, f"{v:.3f}s", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(load_test_dir, "latency_percentiles.png"))
        plt.close()
