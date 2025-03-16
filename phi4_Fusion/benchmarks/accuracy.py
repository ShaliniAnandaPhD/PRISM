"""
accuracy.py - Accuracy benchmarking for Phi-4 + PRISM fusion model

This module provides accuracy benchmarking utilities to evaluate the
model's performance on legal tasks including citation accuracy,
hallucination detection, and question answering.

"""

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from fusion import FusionModel


class AccuracyBenchmark:
    """Comprehensive accuracy benchmarking for the fusion model."""
    
    def __init__(self, model: FusionModel, benchmark_path: str = "benchmarks/data"):
        """
        Initialize the accuracy benchmark.
        
        Args:
            model: FusionModel instance to evaluate
            benchmark_path: Path to benchmark data directory
        """
        self.model = model
        self.benchmark_path = benchmark_path
        self.fusion_ratio = model.fusion_ratio
        self.results = {}
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
    
    def load_benchmark_data(self, benchmark_type: str) -> Any:
        """
        Load benchmark data from file.
        
        Args:
            benchmark_type: Type of benchmark data to load
            
        Returns:
            Loaded benchmark data or empty container
        """
        filename_map = {
            "citation": "citation_benchmark.json",
            "hallucination": "hallucination_benchmark.json",
            "qa": "qa_benchmark.json",
            "domain": "domain_benchmark.json",
        }
        
        if benchmark_type not in filename_map:
            self.logger.warning(f"Unknown benchmark type: {benchmark_type}")
            return [] if benchmark_type != "domain" else {}
        
        file_path = os.path.join(self.benchmark_path, filename_map[benchmark_type])
        
        if not os.path.exists(file_path):
            self.logger.warning(f"Benchmark file not found: {file_path}")
            return [] if benchmark_type != "domain" else {}
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return data
        except Exception as e:
            self.logger.error(f"Error loading benchmark data: {e}")
            return [] if benchmark_type != "domain" else {}
    
    def evaluate_citation_accuracy(self) -> Dict[str, Any]:
        """
        Evaluate citation accuracy of the model.
        
        Returns:
            Dictionary with citation accuracy metrics
        """
        self.logger.info("Evaluating citation accuracy")
        
        # Load benchmark data
        benchmark_data = self.load_benchmark_data("citation")
        
        if not benchmark_data:
            self.logger.warning("No citation benchmark data available")
            return {"f1_score": 0.0, "note": "No benchmark data available"}
        
        self.logger.info(f"Running citation evaluation with {len(benchmark_data)} test cases")
        
        # Track metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Track per-document metrics
        documents_evaluated = 0
        document_f1_scores = []
        
        for test_case in benchmark_data:
            documents_evaluated += 1
            query = test_case["query"]
            document = test_case["document"]
            expected_citations = test_case["expected_citations"]
            
            # Generate response
            result = self.model.generate(query, document=document)
            
            # Extract citations from response
            if "citations" in result:
                actual_citations = result["citations"]
                
                # Extract citation texts for comparison
                citation_texts = [c.get("snippet", "") for c in actual_citations]
                
                # Track metrics for this document
                doc_true_positives = 0
                doc_false_positives = 0
                doc_false_negatives = 0
                
                # Check for matches between expected and actual citations
                for exp_citation in expected_citations:
                    found = False
                    for act_citation in citation_texts:
                        # Simple string matching (could use more sophisticated methods)
                        if self._citation_match(exp_citation, act_citation):
                            doc_true_positives += 1
                            true_positives += 1
                            found = True
                            break
                    
                    if not found:
                        doc_false_negatives += 1
                        false_negatives += 1
                
                # Check for incorrect citations
                for act_citation in citation_texts:
                    found = False
                    for exp_citation in expected_citations:
                        if self._citation_match(exp_citation, act_citation):
                            found = True
                            break
                    
                    if not found:
                        doc_false_positives += 1
                        false_positives += 1
                
                # Calculate F1 score for this document
                if doc_true_positives + doc_false_positives > 0:
                    doc_precision = doc_true_positives / (doc_true_positives + doc_false_positives)
                else:
                    doc_precision = 0.0
                    
                if doc_true_positives + doc_false_negatives > 0:
                    doc_recall = doc_true_positives / (doc_true_positives + doc_false_negatives)
                else:
                    doc_recall = 0.0
                    
                if doc_precision + doc_recall > 0:
                    doc_f1 = 2 * doc_precision * doc_recall / (doc_precision + doc_recall)
                else:
                    doc_f1 = 0.0
                
                document_f1_scores.append(doc_f1)
        
        # Calculate overall metrics
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
            
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0
            
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        # Calculate average document F1 score
        if document_f1_scores:
            avg_doc_f1 = sum(document_f1_scores) / len(document_f1_scores)
        else:
            avg_doc_f1 = 0.0
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "avg_document_f1": float(avg_doc_f1),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "documents_evaluated": documents_evaluated
        }
    
    def _citation_match(self, citation1: str, citation2: str) -> bool:
        """
        Check if two citation strings match.
        
        Args:
            citation1: First citation string
            citation2: Second citation string
            
        Returns:
            True if citations match, False otherwise
        """
        # Simple containment check - could be more sophisticated
        return citation1 in citation2 or citation2 in citation1
    
    def evaluate_hallucination_rate(self) -> Dict[str, Any]:
        """
        Evaluate hallucination rate of the model.
        
        Returns:
            Dictionary with hallucination metrics
        """
        self.logger.info("Evaluating hallucination rate")
        
        # Load benchmark data
        benchmark_data = self.load_benchmark_data("hallucination")
        
        if not benchmark_data:
            self.logger.warning("No hallucination benchmark data available")
            return {"hallucination_rate": 1.0, "note": "No benchmark data available"}
        
        self.logger.info(f"Running hallucination evaluation with {len(benchmark_data)} test cases")
        
        # Track metrics
        total_statements = 0
        hallucinated_statements = 0
        
        # Track per-category metrics
        categories = defaultdict(lambda: {"total": 0, "hallucinated": 0})
        
        for test_case in benchmark_data:
            query = test_case["query"]
            document = test_case["document"]
            factual_statements = test_case["factual_statements"]
            
            # Generate response
            result = self.model.generate(query, document=document)
            response_text = result["response"]
            
            # Check each factual statement
            for statement in factual_statements:
                fact = statement["text"]
                is_true = statement["is_true"]
                category = statement.get("category", "general")
                
                # Check if statement appears in response
                if self._text_contains_statement(response_text, fact):
                    total_statements += 1
                    categories[category]["total"] += 1
                    
                    # If statement is false but included, it's a hallucination
                    if not is_true:
                        hallucinated_statements += 1
                        categories[category]["hallucinated"] += 1
        
        # Calculate overall hallucination rate
        if total_statements > 0:
            hallucination_rate = hallucinated_statements / total_statements
        else:
            hallucination_rate = 1.0  # Default to worst case if no statements found
        
        # Calculate per-category hallucination rates
        category_rates = {}
        for category, counts in categories.items():
            if counts["total"] > 0:
                rate = counts["hallucinated"] / counts["total"]
            else:
                rate = 1.0
            category_rates[category] = float(rate)
        
        return {
            "hallucination_rate": float(hallucination_rate),
            "hallucinated_statements": hallucinated_statements,
            "total_statements": total_statements,
            "category_rates": category_rates
        }
    
    def _text_contains_statement(self, text: str, statement: str) -> bool:
        """
        Check if text contains a statement.
        
        Args:
            text: Text to check
            statement: Statement to look for
            
        Returns:
            True if statement is found, False otherwise
        """
        # Normalize text for comparison
        text_norm = re.sub(r'\s+', ' ', text.lower())
        statement_norm = re.sub(r'\s+', ' ', statement.lower())
        
        return statement_norm in text_norm
    
    def evaluate_legal_qa_accuracy(self) -> Dict[str, Any]:
        """
        Evaluate legal question answering accuracy.
        
        Returns:
            Dictionary with QA accuracy metrics
        """
        self.logger.info("Evaluating legal QA accuracy")
        
        # Load benchmark data
        benchmark_data = self.load_benchmark_data("qa")
        
        if not benchmark_data:
            self.logger.warning("No QA benchmark data available")
            return {"accuracy": 0.0, "note": "No benchmark data available"}
        
        self.logger.info(f"Running QA evaluation with {len(benchmark_data)} questions")
        
        # Track metrics
        correct_answers = 0
        total_questions = len(benchmark_data)
        
        # Track by question type
        question_types = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for test_case in benchmark_data:
            query = test_case["question"]
            document = test_case.get("document")  # Optional
            expected_answer = test_case["answer"]
            question_type = test_case.get("type", "general")
            
            # Update question type counter
            question_types[question_type]["total"] += 1
            
            # Generate response
            result = self.model.generate(query, document=document)
            response_text = result["response"]
            
            # Evaluate answer
            is_correct = self._evaluate_answer(test_case, response_text)
            
            if is_correct:
                correct_answers += 1
                question_types[question_type]["correct"] += 1
        
        # Calculate overall accuracy
        if total_questions > 0:
            accuracy = correct_answers / total_questions
        else:
            accuracy = 0.0
        
        # Calculate per-type accuracy
        type_accuracy = {}
        for q_type, counts in question_types.items():
            if counts["total"] > 0:
                type_accuracy[q_type] = float(counts["correct"] / counts["total"])
            else:
                type_accuracy[q_type] = 0.0
        
        return {
            "accuracy": float(accuracy),
            "correct_answers": correct_answers,
            "total_questions": total_questions,
            "type_accuracy": type_accuracy
        }
    
    def _evaluate_answer(self, test_case: Dict[str, Any], response_text: str) -> bool:
        """
        Evaluate if an answer is correct.
        
        Args:
            test_case: Test case with expected answer
            response_text: Model's response
            
        Returns:
            True if answer is correct, False otherwise
        """
        expected_answer = test_case["answer"]
        
        # Multiple choice questions
        if "multiple_choice" in test_case and test_case["multiple_choice"]:
            options = test_case["options"]
            for option_letter, option_text in options.items():
                if option_letter == expected_answer and self._contains_option_selection(response_text, option_letter):
                    return True
            return False
        
        # Free-form answers - check for key phrases
        key_phrases = test_case.get("key_phrases", [expected_answer])
        
        # If any key phrase is found, answer is correct
        for phrase in key_phrases:
            if phrase.lower() in response_text.lower():
                return True
        
        return False
    
    def _contains_option_selection(self, text: str, option_letter: str) -> bool:
        """
        Check if text contains a clear option selection.
        
        Args:
            text: Response text
            option_letter: Expected option letter (A, B, C, etc.)
            
        Returns:
            True if text indicates selection of the option
        """
        # Look for patterns like "Option A", "The answer is A", "A is correct"
        patterns = [
            rf"\b[Oo]ption\s+{option_letter}\b",
            rf"\b[Tt]he\s+answer\s+is\s+{option_letter}\b",
            rf"\b{option_letter}\s+is\s+(?:the\s+)?(?:correct|right)\b",
            rf"\b[Cc]hoose\s+{option_letter}\b",
            rf"\b[Ss]elect\s+{option_letter}\b",
            rf"^\s*{option_letter}\.?\s*$",  # Just the letter at the start of a line
            rf"^\s*[Aa]nswer:\s*{option_letter}\b"
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def evaluate_domain_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate performance across legal domains.
        
        Returns:
            Dictionary with domain-specific performance metrics
        """
        self.logger.info("Evaluating domain performance")
        
        # Load benchmark data
        benchmark_data = self.load_benchmark_data("domain")
        
        if not benchmark_data:
            self.logger.warning("No domain benchmark data available")
            return {"overall": {"accuracy": 0.0, "note": "No benchmark data available"}}
        
        domain_results = {}
        total_correct = 0
        total_cases = 0
        
        for domain, test_cases in benchmark_data.items():
            self.logger.info(f"Evaluating domain: {domain} with {len(test_cases)} test cases")
            
            # Track domain-specific metrics
            correct = 0
            domain_total = len(test_cases)
            
            for test_case in test_cases:
                query = test_case["query"]
                document = test_case.get("document")  # Optional
                expected_result = test_case["expected_result"]
                
                # Generate response
                result = self.model.generate(query, document=document)
                response_text = result["response"]
                
                # Evaluate response
                if self._evaluate_domain_response(response_text, expected_result):
                    correct += 1
            
            # Calculate domain-specific accuracy
            if domain_total > 0:
                accuracy = correct / domain_total
            else:
                accuracy = 0.0
            
            domain_results[domain] = {
                "accuracy": float(accuracy),
                "correct": correct,
                "total": domain_total
            }
            
            # Update overall counts
            total_correct += correct
            total_cases += domain_total
        
        # Calculate overall accuracy
        if total_cases > 0:
            overall_accuracy = total_correct / total_cases
        else:
            overall_accuracy = 0.0
        
        # Add overall results
        domain_results["overall"] = {
            "accuracy": float(overall_accuracy),
            "correct": total_correct,
            "total": total_cases
        }
        
        return domain_results
    
    def _evaluate_domain_response(self, response_text: str, expected_result: Union[str, List[str]]) -> bool:
        """
        Evaluate if a domain-specific response is correct.
        
        Args:
            response_text: Model's response
            expected_result: Expected result or list of acceptable results
            
        Returns:
            True if response matches expected result, False otherwise
        """
        if isinstance(expected_result, list):
            # Multiple acceptable results
            for expected in expected_result:
                if expected.lower() in response_text.lower():
                    return True
            return False
        else:
            # Single expected result
            return expected_result.lower() in response_text.lower()
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all accuracy benchmarks.
        
        Returns:
            Dictionary with all benchmark results
        """
        self.logger.info(f"Running all accuracy benchmarks with fusion ratio {self.fusion_ratio}")
        
        # Run individual benchmarks
        citation_results = self.evaluate_citation_accuracy()
        hallucination_results = self.evaluate_hallucination_rate()
        qa_results = self.evaluate_legal_qa_accuracy()
        domain_results = self.evaluate_domain_performance()
        
        # Combine results
        results = {
            "citation": citation_results,
            "hallucination": hallucination_results,
            "qa": qa_results,
            "domain": domain_results,
            "fusion_ratio": self.fusion_ratio,
            "timestamp": self._get_timestamp()
        }
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results)
        results["overall"] = overall_score
        
        self.logger.info(f"Accuracy benchmarks completed. Overall score: {overall_score['score']:.4f}")
        
        return results
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate an overall score from all benchmarks.
        
        Args:
            results: Dictionary with all benchmark results
            
        Returns:
            Dictionary with overall score and components
        """
        # Extract key metrics
        f1_score = results["citation"].get("f1_score", 0)
        hallucination_rate = results["hallucination"].get("hallucination_rate", 1.0)
        qa_accuracy = results["qa"].get("accuracy", 0)
        domain_accuracy = results["domain"]["overall"].get("accuracy", 0)
        
        # Calculate weighted score
        # This is a simplified scoring method - could be more sophisticated
        overall_score = (
            f1_score * 0.3 +                    # 30% weight to citation accuracy
            (1 - hallucination_rate) * 0.3 +    # 30% weight to factual accuracy
            qa_accuracy * 0.2 +                 # 20% weight to QA performance
            domain_accuracy * 0.2               # 20% weight to domain performance
        )
        
        return {
            "score": float(overall_score),
            "f1_score": float(f1_score),
            "factual_accuracy": float(1 - hallucination_rate),
            "qa_accuracy": float(qa_accuracy),
            "domain_accuracy": float(domain_accuracy),
            "weights": {
                "citation": 0.3,
                "factual": 0.3,
                "qa": 0.2,
                "domain": 0.2
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_accuracy_benchmark(model: FusionModel, benchmark_path: str = "benchmarks/data") -> Dict[str, Any]:
    """
    Run comprehensive accuracy benchmark.
    
    Args:
        model: FusionModel instance
        benchmark_path: Path to benchmark data directory
        
    Returns:
        Dictionary with benchmark results
    """
    benchmark = AccuracyBenchmark(model, benchmark_path)
    return benchmark.run_all_benchmarks()


"""
SUMMARY:
- Provides comprehensive accuracy benchmarking for the fusion model
- Evaluates citation accuracy, hallucination rate, and QA performance
- Measures performance across different legal domains
- Uses realistic evaluation metrics for legal understanding
- Produces detailed breakdowns by category and question type
- Calculates an overall weighted score for model comparison

TODO:
- Implement human evaluation pipeline for subjective assessments
- Add support for more sophisticated answer matching using embeddings
- Integrate with standard legal benchmarks (LexGLUE, LegalBench)
- Develop continuous benchmark tracking system
- Implement reference model comparisons
- Add statistical significance testing for model improvements
- Support for customizable evaluation weights
"""
