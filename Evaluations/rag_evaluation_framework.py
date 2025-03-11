"""
RAG PIPELINE EVALUATION SCRIPT
This script evaluates the functionality, accuracy, performance, and auditability of the RAG 
(Retrieval-Augmented Generation) pipeline.

Features:
1. Comprehensive evaluation of retrieval quality
2. Response accuracy assessment with semantic similarity
3. Performance benchmarking with detailed metrics
4. Audit trail for evaluation runs
5. Document coverage analysis
6. Detailed reporting with visualizations

The framework is designed to meet industry standards for evaluating LLM-based 
retrieval-augmented generation systems.
"""

import logging
import logging.handlers
import json
import time
import os
import sys
import io
import re
import uuid
import socket
import traceback
import hashlib
import hmac
import base64
import zlib
import gzip
import threading
import queue
import signal
import datetime
import argparse
import sqlite3
import ssl
import secrets
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
from functools import wraps

# Try importing optional dependencies
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    NLP_METRICS_AVAILABLE = True
except ImportError:
    NLP_METRICS_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.metrics import edit_distance
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    import requests
    REMOTE_AVAILABLE = True
except ImportError:
    REMOTE_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Constants
DEFAULT_LOG_DIR = "rag_logs"
DEFAULT_RESULTS_DIR = "rag_results"
DEFAULT_DB_FILE = "rag_evaluation.db"
DEFAULT_CONFIG_FILE = "rag_evaluation_config.json"
MAX_RESPONSE_LENGTH = 10000
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DASHBOARD_PORT = 5001
MAX_QUERY_BATCH_SIZE = 100

# Set up logger for the framework itself
evaluation_logger = logging.getLogger("rag_evaluation")
evaluation_logger.setLevel(logging.INFO)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
evaluation_logger.addHandler(_console_handler)

#-----------------------------------------------------------------------------
# EVALUATION CONTEXT & RESULTS STORAGE
#-----------------------------------------------------------------------------

class EvaluationContext:
    """
    Manages an evaluation run with context and audit trail.
    
    Features:
    - Unique ID for each evaluation run
    - Metadata tracking for evaluation context
    - Timestamping and auditing
    """
    def __init__(self, name: str, description: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None,
                 log_dir: str = DEFAULT_LOG_DIR):
        """
        Initialize evaluation context.
        
        Args:
            name: Name of the evaluation run
            description: Description of the evaluation
            metadata: Additional metadata for the evaluation
            log_dir: Directory to store logs
        """
        self.eval_id = str(uuid.uuid4())
        self.name = name
        self.description = description or f"RAG evaluation run: {name}"
        self.metadata = metadata or {}
        self.log_dir = log_dir
        self.start_time = datetime.datetime.now()
        self.end_time = None
        self.status = "running"
        
        # Add some default metadata
        self.metadata.update({
            "hostname": socket.gethostname(),
            "username": os.environ.get('USER', os.environ.get('USERNAME', 'unknown')),
            "start_time_utc": self.start_time.isoformat(),
            "python_version": sys.version,
        })
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up specialized logger for this evaluation
        self.logger = self._setup_logger()
        self.logger.info(f"Starting evaluation: {name} (ID: {self.eval_id})")
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a specialized logger for this evaluation run."""
        logger = logging.getLogger(f"rag_eval_{self.eval_id}")
        logger.setLevel(logging.DEBUG)
        
        # Create file handler with rotation
        log_file = os.path.join(self.log_dir, f"eval_{self.name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def finish(self, status: str = "completed"):
        """
        Mark the evaluation as finished.
        
        Args:
            status: Final status (completed, failed, etc.)
        """
        self.end_time = datetime.datetime.now()
        self.status = status
        self.metadata["end_time_utc"] = self.end_time.isoformat()
        self.metadata["duration_seconds"] = (self.end_time - self.start_time).total_seconds()
        
        self.logger.info(f"Evaluation finished: {self.name} (ID: {self.eval_id})")
        self.logger.info(f"Status: {status}")
        self.logger.info(f"Duration: {self.metadata['duration_seconds']:.2f} seconds")
    
    def log_query_evaluation(self, query: str, response: str, expected: Optional[str] = None,
                           metrics: Optional[Dict[str, Any]] = None):
        """
        Log a query evaluation.
        
        Args:
            query: The query that was evaluated
            response: The response from the RAG system
            expected: Expected response (if available)
            metrics: Metrics for this evaluation
        """
        metrics = metrics or {}
        
        # Truncate response and expected for logging if too long
        truncated_response = (response[:MAX_RESPONSE_LENGTH] + "...") if len(response) > MAX_RESPONSE_LENGTH else response
        
        if expected:
            truncated_expected = (expected[:MAX_RESPONSE_LENGTH] + "...") if len(expected) > MAX_RESPONSE_LENGTH else expected
            self.logger.info(f"Query: {query}")
            self.logger.info(f"Response: {truncated_response}")
            self.logger.info(f"Expected: {truncated_expected}")
        else:
            self.logger.info(f"Query: {query}")
            self.logger.info(f"Response: {truncated_response}")
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"Metric - {metric_name}: {metric_value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation context to dictionary for serialization."""
        return {
            "eval_id": self.eval_id,
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "metadata": self.metadata
        }

class EvaluationDatabase:
    """
    Database for storing and retrieving evaluation results.
    
    Features:
    - Persistent storage of evaluation results
    - Query and response tracking
    - Metrics and performance data
    """
    def __init__(self, db_file: str = DEFAULT_DB_FILE):
        """
        Initialize evaluation database.
        
        Args:
            db_file: Path to the SQLite database file
        """
        self.db_file = db_file
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create evaluations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                eval_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT,
                metadata TEXT
            )
        ''')
        
        # Create queries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                eval_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT,
                expected TEXT,
                similarity_score REAL,
                retrieval_score REAL,
                response_time REAL,
                timestamp TEXT NOT NULL,
                metrics TEXT,
                FOREIGN KEY (eval_id) REFERENCES evaluations (eval_id)
            )
        ''')
        
        # Create aggregate metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS aggregate_metrics (
                eval_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metric_type TEXT,
                PRIMARY KEY (eval_id, metric_name),
                FOREIGN KEY (eval_id) REFERENCES evaluations (eval_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Get a connection to the SQLite database."""
        return sqlite3.connect(self.db_file)
    
    def store_evaluation(self, context: EvaluationContext):
        """
        Store evaluation context in the database.
        
        Args:
            context: The evaluation context to store
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT OR REPLACE INTO evaluations (eval_id, name, description, start_time, end_time, status, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                context.eval_id,
                context.name,
                context.description,
                context.start_time.isoformat(),
                context.end_time.isoformat() if context.end_time else None,
                context.status,
                json.dumps(context.metadata)
            )
        )
        
        conn.commit()
        conn.close()
    
    def store_query_evaluation(self, eval_id: str, query: str, response: str, 
                             expected: Optional[str] = None, similarity_score: Optional[float] = None,
                             retrieval_score: Optional[float] = None, response_time: Optional[float] = None,
                             metrics: Optional[Dict[str, Any]] = None):
        """
        Store a query evaluation result.
        
        Args:
            eval_id: Evaluation ID
            query: The query that was evaluated
            response: The response from the RAG system
            expected: Expected response (if available)
            similarity_score: Similarity score between response and expected
            retrieval_score: Score for retrieval quality
            response_time: Response time in seconds
            metrics: Additional metrics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO query_evaluations (eval_id, query, response, expected, similarity_score, "
            "retrieval_score, response_time, timestamp, metrics) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                eval_id,
                query,
                response,
                expected,
                similarity_score,
                retrieval_score,
                response_time,
                datetime.datetime.now().isoformat(),
                json.dumps(metrics) if metrics else None
            )
        )
        
        conn.commit()
        conn.close()
    
    def store_aggregate_metrics(self, eval_id: str, metrics: Dict[str, Any]):
        """
        Store aggregate metrics for an evaluation.
        
        Args:
            eval_id: Evaluation ID
            metrics: Dictionary of metric names and values
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        for metric_name, metric_value in metrics.items():
            metric_type = type(metric_value).__name__
            
            # Convert non-scalar metrics to JSON
            if not isinstance(metric_value, (int, float, bool)):
                metric_value = json.dumps(metric_value)
                metric_type = "json"
            
            cursor.execute(
                "INSERT OR REPLACE INTO aggregate_metrics (eval_id, metric_name, metric_value, metric_type) "
                "VALUES (?, ?, ?, ?)",
                (eval_id, metric_name, metric_value, metric_type)
            )
        
        conn.commit()
        conn.close()
    
    def get_evaluation(self, eval_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an evaluation by ID.
        
        Args:
            eval_id: Evaluation ID
            
        Returns:
            Evaluation data or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM evaluations WHERE eval_id = ?", (eval_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Extract column names
        columns = [col[0] for col in cursor.description]
        evaluation = dict(zip(columns, row))
        
        # Parse metadata JSON
        if evaluation.get('metadata'):
            evaluation['metadata'] = json.loads(evaluation['metadata'])
        
        # Get query evaluations
        cursor.execute("SELECT * FROM query_evaluations WHERE eval_id = ?", (eval_id,))
        query_rows = cursor.fetchall()
        
        query_evaluations = []
        for q_row in query_rows:
            q_columns = [col[0] for col in cursor.description]
            q_eval = dict(zip(q_columns, q_row))
            
            # Parse metrics JSON
            if q_eval.get('metrics'):
                q_eval['metrics'] = json.loads(q_eval['metrics'])
            
            query_evaluations.append(q_eval)
        
        evaluation['query_evaluations'] = query_evaluations
        
        # Get aggregate metrics
        cursor.execute("SELECT metric_name, metric_value, metric_type FROM aggregate_metrics WHERE eval_id = ?", (eval_id,))
        metric_rows = cursor.fetchall()
        
        metrics = {}
        for metric_name, metric_value, metric_type in metric_rows:
            # Parse JSON metrics
            if metric_type == 'json':
                metrics[metric_name] = json.loads(metric_value)
            else:
                metrics[metric_name] = metric_value
        
        evaluation['aggregate_metrics'] = metrics
        
        conn.close()
        return evaluation
    
    def list_evaluations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List recent evaluations.
        
        Args:
            limit: Maximum number of evaluations to return
            
        Returns:
            List of evaluation data
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT eval_id, name, description, start_time, end_time, status FROM evaluations "
            "ORDER BY start_time DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        
        evaluations = []
        for row in rows:
            evaluations.append({
                "eval_id": row[0],
                "name": row[1],
                "description": row[2],
                "start_time": row[3],
                "end_time": row[4],
                "status": row[5]
            })
        
        conn.close()
        return evaluations

#-----------------------------------------------------------------------------
# RAG EVALUATION METRICS
#-----------------------------------------------------------------------------

class RagMetrics:
    """
    Computes metrics for RAG evaluation.
    
    Features:
    - Response similarity metrics
    - Retrieval quality metrics
    - Response time and performance metrics
    - Document relevance scoring
    """
    def __init__(self):
        """Initialize RAG metrics calculator."""
        self.has_nlp_metrics = NLP_METRICS_AVAILABLE
        self.has_nltk = NLTK_AVAILABLE
        self.has_transformers = TRANSFORMERS_AVAILABLE
        
        # Initialize text similarity models if available
        self.transformer_model = None
        self.transformer_tokenizer = None
        self.tfidf_vectorizer = None
        
        if self.has_transformers:
            try:
                # Use a sentence-transformers model for better similarity
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.transformer_model = AutoModel.from_pretrained(model_name)
                evaluation_logger.info(f"Loaded transformer model: {model_name}")
            except Exception as e:
                evaluation_logger.warning(f"Failed to load transformer model: {e}")
                self.has_transformers = False
        
        if self.has_nlp_metrics:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    def compute_similarity(self, response: str, expected: str) -> Dict[str, float]:
        """
        Compute similarity between response and expected response.
        
        Args:
            response: Generated response
            expected: Expected response
            
        Returns:
            Dictionary of similarity metrics
        """
        metrics = {}
        
        # Don't try to compute if either string is empty
        if not response or not expected:
            return {"text_similarity": 0.0}
        
        # First try transformer-based similarity if available
        if self.has_transformers and self.transformer_model and self.transformer_tokenizer:
            try:
                # Use mean pooling to get sentence embeddings
                def mean_pooling(model_output, attention_mask):
                    token_embeddings = model_output[0]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Encode both texts
                inputs1 = self.transformer_tokenizer(response, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs2 = self.transformer_tokenizer(expected, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs1 = self.transformer_model(**inputs1)
                    outputs2 = self.transformer_model(**inputs2)
                
                embeddings1 = mean_pooling(outputs1, inputs1["attention_mask"])
                embeddings2 = mean_pooling(outputs2, inputs2["attention_mask"])
                
                # Normalize embeddings
                embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
                embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
                
                # Compute cosine similarity
                similarity = torch.mm(embeddings1, embeddings2.transpose(0, 1)).item()
                metrics["semantic_similarity"] = float(similarity)
            except Exception as e:
                evaluation_logger.warning(f"Error computing transformer similarity: {e}")
        
        # Try TFIDF similarity if available
        if self.has_nlp_metrics:
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([response, expected])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                metrics["tfidf_similarity"] = float(similarity)
            except Exception as e:
                evaluation_logger.warning(f"Error computing TFIDF similarity: {e}")
        
        # Compute BLEU score if NLTK is available
        if self.has_nltk:
            try:
                reference = [expected.lower().split()]
                candidate = response.lower().split()
                smoothing = SmoothingFunction().method1
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
                metrics["bleu_score"] = float(bleu_score)
                
                # Normalized edit distance (Levenshtein)
                edit_dist = edit_distance(response.lower(), expected.lower())
                # Normalize by the length of the longer string
                norm_edit_dist = 1.0 - (edit_dist / max(len(response), len(expected)))
                metrics["edit_similarity"] = float(norm_edit_dist)
            except Exception as e:
                evaluation_logger.warning(f"Error computing NLTK metrics: {e}")
        
        # Fallback to simple token overlap
        try:
            response_tokens = set(response.lower().split())
            expected_tokens = set(expected.lower().split())
            
            if not response_tokens or not expected_tokens:
                overlap = 0.0
            else:
                # Jaccard similarity
                overlap = len(response_tokens.intersection(expected_tokens)) / len(response_tokens.union(expected_tokens))
            
            metrics["token_overlap"] = float(overlap)
        except Exception as e:
            evaluation_logger.warning(f"Error computing token overlap: {e}")
            metrics["token_overlap"] = 0.0
        
        # Compute an aggregate text similarity score
        if "semantic_similarity" in metrics:
            metrics["text_similarity"] = metrics["semantic_similarity"]
        elif "tfidf_similarity" in metrics:
            metrics["text_similarity"] = metrics["tfidf_similarity"]
        elif "token_overlap" in metrics:
            metrics["text_similarity"] = metrics["token_overlap"]
        else:
            metrics["text_similarity"] = 0.0
        
        return metrics
    
    def evaluate_relevance(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the relevance of retrieved documents to the query.
        
        Args:
            query: The query
            retrieved_docs: List of retrieved documents with text and metadata
            
        Returns:
            Dictionary of relevance metrics
        """
        metrics = {}
        
        if not retrieved_docs:
            return {"retrieval_precision": 0.0, "avg_relevance_score": 0.0}
        
        # Calculate relevance scores if possible
        relevance_scores = []
        
        if self.has_transformers and self.transformer_model and self.transformer_tokenizer:
            try:
                # Use the transformer model to calculate query-document relevance
                def mean_pooling(model_output, attention_mask):
                    token_embeddings = model_output[0]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Encode query
                query_inputs = self.transformer_tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                with torch.no_grad():
                    query_outputs = self.transformer_model(**query_inputs)
                
                query_embedding = mean_pooling(query_outputs, query_inputs["attention_mask"])
                query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
                
                # Encode and score each document
                for doc in retrieved_docs:
                    doc_text = doc.get('text', '')
                    if not doc_text:
                        continue
                        
                    doc_inputs = self.transformer_tokenizer(doc_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    
                    with torch.no_grad():
                        doc_outputs = self.transformer_model(**doc_inputs)
                    
                    doc_embedding = mean_pooling(doc_outputs, doc_inputs["attention_mask"])
                    doc_embedding = torch.nn.functional.normalize(doc_embedding, p=2, dim=1)
                    
                    # Calculate similarity
                    score = torch.mm(query_embedding, doc_embedding.transpose(0, 1)).item()
                    relevance_scores.append(score)
            except Exception as e:
                evaluation_logger.warning(f"Error computing transformer relevance: {e}")
        
        # Fallback to TF-IDF if transformers not available
        elif self.has_nlp_metrics:
            try:
                # Create corpus with query and all documents
                corpus = [query]
                for doc in retrieved_docs:
                    doc_text = doc.get('text', '')
                    if doc_text:
                        corpus.append(doc_text)
                
                # Fit and transform
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
                
                # Compute similarity between query and each document
                query_vector = tfidf_matrix[0:1]
                for i in range(1, len(corpus)):
                    doc_vector = tfidf_matrix[i:i+1]
                    similarity = cosine_similarity(query_vector, doc_vector)[0][0]
                    relevance_scores.append(similarity)
            except Exception as e:
                evaluation_logger.warning(f"Error computing TF-IDF relevance: {e}")
        
        # Compute metrics based on relevance scores
        if relevance_scores:
            metrics["avg_relevance_score"] = float(sum(relevance_scores) / len(relevance_scores))
            
            # Calculate precision based on relevance threshold
            relevant_count = sum(1 for score in relevance_scores if score >= DEFAULT_SIMILARITY_THRESHOLD)
            metrics["retrieval_precision"] = float(relevant_count / len(relevance_scores))
        else:
            # Without advanced metrics, assume all retrievals have basic relevance
            metrics["avg_relevance_score"] = 0.5
            metrics["retrieval_precision"] = 0.5
        
        return metrics
    
    def calculate_overall_metrics(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall metrics for a set of query results.
        
        Args:
            query_results: List of query evaluation results
            
        Returns:
            Dictionary of aggregate metrics
        """
        if not query_results:
            return {}
        
        metrics = {}
        
        # Extract arrays of metrics
        similarity_scores = [r.get('similarity_score', 0) for r in query_results if r.get('similarity_score') is not None]
        retrieval_scores = [r.get('retrieval_score', 0) for r in query_results if r.get('retrieval_score') is not None]
        response_times = [r.get('response_time', 0) for r in query_results if r.get('response_time') is not None]
        
        # Calculate standard statistics
        if similarity_scores:
            metrics["avg_similarity"] = float(sum(similarity_scores) / len(similarity_scores))
            metrics["min_similarity"] = float(min(similarity_scores))
            metrics["max_similarity"] = float(max(similarity_scores))
        
        if retrieval_scores:
            metrics["avg_retrieval_score"] = float(sum(retrieval_scores) / len(retrieval_scores))
            metrics["min_retrieval_score"] = float(min(retrieval_scores))
            metrics["max_retrieval_score"] = float(max(retrieval_scores))
        
        if response_times:
            metrics["avg_response_time"] = float(sum(response_times) / len(response_times))
            metrics["min_response_time"] = float(min(response_times))
            metrics["max_response_time"] = float(max(response_times))
            metrics["p90_response_time"] = float(sorted(response_times)[int(len(response_times) * 0.9)])
            metrics["p95_response_time"] = float(sorted(response_times)[int(len(response_times) * 0.95)])
        
        # Calculate accuracy based on threshold
        if similarity_scores:
            correct_count = sum(1 for score in similarity_scores if score >= DEFAULT_SIMILARITY_THRESHOLD)
            metrics["accuracy"] = float(correct_count / len(similarity_scores))
        
        # Calculate overall score (balanced between similarity, retrieval, and response time)
        overall_score_components = []
        
        if similarity_scores:
            overall_score_components.append(sum(similarity_scores) / len(similarity_scores))
        
        if retrieval_scores:
            overall_score_components.append(sum(retrieval_scores) / len(retrieval_scores))
        
        if response_times and metrics.get("avg_response_time"):
            # Convert response time to a score (lower is better)
            # Assuming 5 seconds is acceptable (score 0.5)
            time_score = min(1.0, 2.5 / metrics["avg_response_time"])
            overall_score_components.append(time_score)
        
        if overall_score_components:
            metrics["overall_score"] = float(sum(overall_score_components) / len(overall_score_components))
        
        return metrics

#-----------------------------------------------------------------------------
# RAG EVALUATION CORE
#-----------------------------------------------------------------------------

class RagEvaluator:
    """
    Main class for evaluating RAG pipelines.
    
    Features:
    - Comprehensive evaluation of RAG performance
    - Support for different evaluation types
    - Result aggregation and reporting
    - Parallel evaluation for efficiency
    """
    def __init__(self, rag_pipeline: Callable, eval_db: Optional[EvaluationDatabase] = None,
                metrics: Optional[RagMetrics] = None, parallel: bool = True, 
                max_workers: int = 4, results_dir: str = DEFAULT_RESULTS_DIR):
        """
        Initialize RAG evaluator.
        
        Args:
            rag_pipeline: Function that accepts a query and returns a response
            eval_db: Evaluation database for storing results
            metrics: Metrics calculator
            parallel: Whether to run evaluations in parallel
            max_workers: Maximum number of parallel workers
            results_dir: Directory to store evaluation results
        """
        self.rag_pipeline = rag_pipeline
        self.eval_db = eval_db or EvaluationDatabase()
        self.metrics = metrics or RagMetrics()
        self.parallel = parallel
        self.max_workers = max_workers
        self.results_dir = results_dir
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        evaluation_logger.info(f"Initialized RAG evaluator. Parallel: {parallel}, Max workers: {max_workers}")
    
    def evaluate_query(self, query: str, expected_response: Optional[str] = None, 
                     context: Optional[EvaluationContext] = None,
                     include_retrieved_docs: bool = False) -> Dict[str, Any]:
        """
        Evaluate a single query against the RAG pipeline.
        
        Args:
            query: The query to evaluate
            expected_response: Expected response (if available)
            context: Evaluation context
            include_retrieved_docs: Whether to include retrieved documents in results
            
        Returns:
            Dictionary with evaluation results
        """
        evaluation_logger.info(f"Evaluating query: {query}")
        result = {
            "query": query,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Measure response time
        start_time = time.perf_counter()
        
        try:
            # Some RAG systems return both response and retrieved docs
            rag_response = self.rag_pipeline(query)
            
            if isinstance(rag_response, tuple) and len(rag_response) >= 2:
                # Assume (response, retrieved_docs) format
                response = rag_response[0]
                retrieved_docs = rag_response[1]
            elif isinstance(rag_response, dict) and 'response' in rag_response:
                # Dictionary format with response and retrieved_docs keys
                response = rag_response.get('response', '')
                retrieved_docs = rag_response.get('retrieved_docs', [])
            else:
                # Just a response string
                response = rag_response
                retrieved_docs = []
            
            end_time = time.perf_counter()
            response_time = end_time - start_time
            
            result["response"] = response
            result["response_time"] = response_time
            
            # Calculate similarity if expected response is available
            if expected_response:
                result["expected"] = expected_response
                similarity_metrics = self.metrics.compute_similarity(response, expected_response)
                result.update(similarity_metrics)
                result["similarity_score"] = similarity_metrics.get("text_similarity", 0.0)
            
            # Calculate retrieval metrics if docs are available
            if retrieved_docs:
                if include_retrieved_docs:
                    result["retrieved_docs"] = retrieved_docs
                
                retrieval_metrics = self.metrics.evaluate_relevance(query, retrieved_docs)
                result.update(retrieval_metrics)
                result["retrieval_score"] = retrieval_metrics.get("avg_relevance_score", 0.0)
            
            # Log to evaluation context if provided
            if context:
                metrics = {
                    "response_time": response_time,
                    "similarity_score": result.get("similarity_score"),
                    "retrieval_score": result.get("retrieval_score")
                }
                context.log_query_evaluation(query, response, expected_response, metrics)
                
                # Store in database
                self.eval_db.store_query_evaluation(
                    context.eval_id,
                    query,
                    response,
                    expected_response,
                    result.get("similarity_score"),
                    result.get("retrieval_score"),
                    response_time,
                    {k: v for k, v in result.items() if k not in [
                        "query", "response", "expected", "timestamp", "retrieved_docs"
                    ]}
                )
            
            evaluation_logger.info(f"Evaluation complete: {query}")
            if expected_response and "similarity_score" in result:
                evaluation_logger.info(f"Similarity score: {result['similarity_score']:.4f}")
            
            return result
        
        except Exception as e:
            end_time = time.perf_counter()
            response_time = end_time - start_time
            
            error_msg = f"Error evaluating query: {str(e)}"
            traceback_str = traceback.format_exc()
            
            evaluation_logger.error(error_msg)
            evaluation_logger.debug(traceback_str)
            
            result["error"] = error_msg
            result["traceback"] = traceback_str
            result["response_time"] = response_time
            
            # Log to evaluation context if provided
            if context:
                context.log_query_evaluation(
                    query, 
                    f"ERROR: {error_msg}", 
                    expected_response,
                    {"response_time": response_time, "error": True}
                )
            
            return result
    
    def evaluate_dataset(self, queries: List[str], expected_responses: Optional[List[str]] = None,
                       name: str = "dataset_evaluation", description: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a dataset of queries.
        
        Args:
            queries: List of queries to evaluate
            expected_responses: List of expected responses (optional)
            name: Name for this evaluation
            description: Description of this evaluation
            metadata: Additional metadata
            
        Returns:
            Dictionary with evaluation results
        """
        # Create evaluation context
        context = EvaluationContext(name, description, metadata)
        context.logger.info(f"Starting dataset evaluation with {len(queries)} queries")
        
        # Store evaluation context
        self.eval_db.store_evaluation(context)
        
        # Prepare queries and expected responses
        if expected_responses and len(expected_responses) != len(queries):
            context.logger.warning(
                f"Number of expected responses ({len(expected_responses)}) doesn't match "
                f"number of queries ({len(queries)}). Some queries will be evaluated without expectations."
            )
            
            # Extend expected_responses with None to match queries length
            expected_responses.extend([None] * (len(queries) - len(expected_responses)))
        
        expected_responses = expected_responses or [None] * len(queries)
        
        # Evaluate queries
        results = []
        
        if self.parallel and len(queries) > 1:
            context.logger.info(f"Running parallel evaluation with {self.max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create evaluation tasks
                eval_tasks = []
                for query, expected in zip(queries, expected_responses):
                    eval_tasks.append(
                        executor.submit(
                            self.evaluate_query, query, expected, context
                        )
                    )
                
                # Collect results
                for task in eval_tasks:
                    try:
                        result = task.result()
                        results.append(result)
                    except Exception as e:
                        context.logger.error(f"Error in evaluation task: {str(e)}")
        else:
            context.logger.info("Running sequential evaluation")
            
            for query, expected in zip(queries, expected_responses):
                result = self.evaluate_query(query, expected, context)
                results.append(result)
        
        # Calculate overall metrics
        overall_metrics = self.metrics.calculate_overall_metrics(results)
        context.logger.info(f"Overall metrics: {json.dumps(overall_metrics, indent=2)}")
        
        # Store overall metrics
        self.eval_db.store_aggregate_metrics(context.eval_id, overall_metrics)
        
        # Generate result report
        report_file = self._generate_report(context, results, overall_metrics)
        context.logger.info(f"Evaluation report saved to: {report_file}")
        
        # Generate visualizations
        viz_files = self._generate_visualizations(context, results, overall_metrics)
        if viz_files:
            context.logger.info(f"Generated {len(viz_files)} visualization files")
        
        # Mark evaluation as completed
        context.finish("completed")
        self.eval_db.store_evaluation(context)
        
        return {
            "eval_id": context.eval_id,
            "name": name,
            "num_queries": len(queries),
            "metrics": overall_metrics,
            "results": results,
            "report_file": report_file,
            "visualization_files": viz_files
        }
    
    def _generate_report(self, context: EvaluationContext, results: List[Dict[str, Any]],
                       overall_metrics: Dict[str, Any]) -> str:
        """
        Generate evaluation report.
        
        Args:
            context: Evaluation context
            results: List of query results
            overall_metrics: Overall metrics
            
        Returns:
            Path to the generated report file
        """
        # Create report data
        report = {
            "evaluation_id": context.eval_id,
            "name": context.name,
            "description": context.description,
            "timestamp": datetime.datetime.now().isoformat(),
            "duration_seconds": context.metadata.get("duration_seconds"),
            "num_queries": len(results),
            "overall_metrics": overall_metrics,
            "query_results": results,
            "metadata": context.metadata
        }
        
        # Save report to file
        report_file = os.path.join(
            self.results_dir, 
            f"report_{context.name}_{context.eval_id}.json"
        )
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Also generate a text summary
        summary_file = os.path.join(
            self.results_dir, 
            f"summary_{context.name}_{context.eval_id}.txt"
        )
        
        with open(summary_file, 'w') as f:
            f.write(f"RAG EVALUATION SUMMARY: {context.name}\n")
            f.write(f"Evaluation ID: {context.eval_id}\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Description: {context.description}\n\n")
            
            f.write("OVERALL METRICS:\n")
            for metric, value in overall_metrics.items():
                f.write(f"  - {metric}: {value}\n")
            
            f.write(f"\nTotal Queries: {len(results)}\n")
            
            if "accuracy" in overall_metrics:
                f.write(f"Accuracy: {overall_metrics['accuracy'] * 100:.2f}%\n")
            
            if "avg_response_time" in overall_metrics:
                f.write(f"Average Response Time: {overall_metrics['avg_response_time']:.4f} seconds\n")
            
            f.write("\n5 EXAMPLE QUERIES:\n")
            for i, result in enumerate(results[:5]):
                f.write(f"Query {i+1}: {result['query']}\n")
                if "error" in result:
                    f.write(f"  Error: {result['error']}\n")
                else:
                    f.write(f"  Response: {result['response'][:100]}...\n")
                    if "expected" in result:
                        f.write(f"  Expected: {result['expected'][:100]}...\n")
                    if "similarity_score" in result:
                        f.write(f"  Similarity: {result['similarity_score']:.4f}\n")
                f.write("\n")
        
        return report_file
    
    def _generate_visualizations(self, context: EvaluationContext, results: List[Dict[str, Any]],
                               overall_metrics: Dict[str, Any]) -> List[str]:
        """
        Generate visualizations of evaluation results.
        
        Args:
            context: Evaluation context
            results: List of query results
            overall_metrics: Overall metrics
            
        Returns:
            List of paths to generated visualization files
        """
        if not VISUALIZATION_AVAILABLE:
            context.logger.warning("Visualization libraries not available. Skipping visualization generation.")
            return []
        
        viz_files = []
        viz_dir = os.path.join(self.results_dir, f"viz_{context.name}_{context.eval_id}")
        os.makedirs(viz_dir, exist_ok=True)
        
        try:
            # Extract metrics for visualization
            similarity_scores = [r.get("similarity_score", 0) for r in results if "similarity_score" in r]
            retrieval_scores = [r.get("retrieval_score", 0) for r in results if "retrieval_score" in r]
            response_times = [r.get("response_time", 0) for r in results if "response_time" in r]
            
            # Create histogram of similarity scores
            if similarity_scores:
                plt.figure(figsize=(10, 6))
                plt.hist(similarity_scores, bins=20, alpha=0.7, color='blue')
                plt.title('Distribution of Similarity Scores')
                plt.xlabel('Similarity Score')
                plt.ylabel('Count')
                plt.grid(axis='y', alpha=0.3)
                plt.axvline(x=DEFAULT_SIMILARITY_THRESHOLD, color='red', linestyle='--', 
                           label=f'Threshold ({DEFAULT_SIMILARITY_THRESHOLD})')
                plt.legend()
                
                file_path = os.path.join(viz_dir, 'similarity_distribution.png')
                plt.savefig(file_path)
                plt.close()
                viz_files.append(file_path)
            
            # Create histogram of response times
            if response_times:
                plt.figure(figsize=(10, 6))
                plt.hist(response_times, bins=20, alpha=0.7, color='green')
                plt.title('Distribution of Response Times')
                plt.xlabel('Response Time (seconds)')
                plt.ylabel('Count')
                plt.grid(axis='y', alpha=0.3)
                
                file_path = os.path.join(viz_dir, 'response_time_distribution.png')
                plt.savefig(file_path)
                plt.close()
                viz_files.append(file_path)
            
            # Create scatter plot of similarity vs retrieval scores
            if similarity_scores and retrieval_scores and len(similarity_scores) == len(retrieval_scores):
                plt.figure(figsize=(10, 6))
                plt.scatter(retrieval_scores, similarity_scores, alpha=0.7)
                plt.title('Similarity Score vs Retrieval Score')
                plt.xlabel('Retrieval Score')
                plt.ylabel('Similarity Score')
                plt.grid(alpha=0.3)
                
                # Add correlation coefficient
                correlation = np.corrcoef(retrieval_scores, similarity_scores)[0, 1]
                plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                        transform=plt.gca().transAxes, fontsize=12)
                
                file_path = os.path.join(viz_dir, 'similarity_vs_retrieval.png')
                plt.savefig(file_path)
                plt.close()
                viz_files.append(file_path)
            
            # Create scatter plot of response time vs similarity
            if response_times and similarity_scores and len(response_times) == len(similarity_scores):
                plt.figure(figsize=(10, 6))
                plt.scatter(response_times, similarity_scores, alpha=0.7, color='orange')
                plt.title('Response Time vs Similarity Score')
                plt.xlabel('Response Time (seconds)')
                plt.ylabel('Similarity Score')
                plt.grid(alpha=0.3)
                
                file_path = os.path.join(viz_dir, 'response_time_vs_similarity.png')
                plt.savefig(file_path)
                plt.close()
                viz_files.append(file_path)
            
            # Create summary bar chart of key metrics
            metrics_to_plot = {k: v for k, v in overall_metrics.items() 
                             if k in ['accuracy', 'avg_similarity', 'avg_retrieval_score', 'overall_score']}
            
            if metrics_to_plot:
                plt.figure(figsize=(10, 6))
                plt.bar(metrics_to_plot.keys(), metrics_to_plot.values(), alpha=0.7, color='purple')
                plt.title('Summary Metrics')
                plt.ylabel('Score')
                plt.ylim(0, 1.0)
                
                # Add value labels
                for i, (k, v) in enumerate(metrics_to_plot.items()):
                    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
                
                file_path = os.path.join(viz_dir, 'summary_metrics.png')
                plt.savefig(file_path)
                plt.close()
                viz_files.append(file_path)
            
            return viz_files
            
        except Exception as e:
            context.logger.error(f"Error generating visualizations: {str(e)}")
            return []
    
    def compare_evaluations(self, eval_ids: List[str], output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple evaluation runs.
        
        Args:
            eval_ids: List of evaluation IDs to compare
            output_file: File to write comparison results to
            
        Returns:
            Dictionary with comparison results
        """
        if not eval_ids or len(eval_ids) < 2:
            evaluation_logger.error("Need at least two evaluation IDs to compare")
            return {"error": "Need at least two evaluation IDs to compare"}
        
        evaluation_logger.info(f"Comparing evaluations: {', '.join(eval_ids)}")
        
        # Get evaluation data
        evaluations = []
        for eval_id in eval_ids:
            eval_data = self.eval_db.get_evaluation(eval_id)
            if eval_data:
                evaluations.append(eval_data)
            else:
                evaluation_logger.warning(f"Evaluation not found: {eval_id}")
        
        if len(evaluations) < 2:
            evaluation_logger.error("Need at least two valid evaluations to compare")
            return {"error": "Need at least two valid evaluations to compare"}
        
        # Extract metrics for comparison
        comparison = {
            "timestamp": datetime.datetime.now().isoformat(),
            "evaluation_ids": eval_ids,
            "evaluations": [],
            "metric_comparison": {}
        }
        
        for eval_data in evaluations:
            eval_summary = {
                "eval_id": eval_data["eval_id"],
                "name": eval_data["name"],
                "timestamp": eval_data["start_time"],
                "metrics": eval_data.get("aggregate_metrics", {})
            }
            comparison["evaluations"].append(eval_summary)
        
        # Compare metrics
        metric_keys = set()
        for eval_summary in comparison["evaluations"]:
            metric_keys.update(eval_summary["metrics"].keys())
        
        for metric in metric_keys:
            metric_values = []
            for eval_summary in comparison["evaluations"]:
                if metric in eval_summary["metrics"]:
                    metric_values.append({
                        "eval_id": eval_summary["eval_id"],
                        "name": eval_summary["name"],
                        "value": eval_summary["metrics"][metric]
                    })
            
            comparison["metric_comparison"][metric] = metric_values
        
        # Generate comparison visualizations
        if VISUALIZATION_AVAILABLE:
            viz_files = self._generate_comparison_visualizations(comparison)
            comparison["visualization_files"] = viz_files
        
        # Write to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(comparison, f, indent=4)
            evaluation_logger.info(f"Comparison saved to: {output_file}")
        
        return comparison
    
    def _generate_comparison_visualizations(self, comparison: Dict[str, Any]) -> List[str]:
        """
        Generate visualizations comparing multiple evaluations.
        
        Args:
            comparison: Comparison data
            
        Returns:
            List of paths to generated visualization files
        """
        viz_files = []
        viz_dir = os.path.join(self.results_dir, f"comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(viz_dir, exist_ok=True)
        
        try:
            # Create bar charts for key metrics
            key_metrics = ['accuracy', 'avg_similarity', 'avg_retrieval_score', 
                         'avg_response_time', 'overall_score']
            
            for metric in key_metrics:
                if metric in comparison["metric_comparison"]:
                    metric_data = comparison["metric_comparison"][metric]
                    if not metric_data:
                        continue
                    
                    # Sort by name for consistency
                    metric_data.sort(key=lambda x: x["name"])
                    
                    names = [item["name"] for item in metric_data]
                    values = [item["value"] for item in metric_data]
                    
                    plt.figure(figsize=(12, 6))
                    bars = plt.bar(names, values, alpha=0.7)
                    plt.title(f'Comparison of {metric}')
                    plt.ylabel(metric)
                    
                    # Add value labels
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                f'{value:.4f}', ha='center', va='bottom')
                    
                    # Adjust for response time (lower is better)
                    if metric == 'avg_response_time':
                        plt.gca().invert_yaxis()
                    
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    
                    file_path = os.path.join(viz_dir, f'comparison_{metric}.png')
                    plt.savefig(file_path)
                    plt.close()
                    viz_files.append(file_path)
            
            # Create radar chart for overall comparison
            if len(comparison["evaluations"]) >= 2:
                # Get metrics that are present in all evaluations
                common_metrics = []
                for metric in ['accuracy', 'avg_similarity', 'avg_retrieval_score', 'overall_score']:
                    if metric in comparison["metric_comparison"] and len(comparison["metric_comparison"][metric]) == len(comparison["evaluations"]):
                        common_metrics.append(metric)
                
                if common_metrics:
                    # Create radar chart
                    num_metrics = len(common_metrics)
                    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
                    angles += angles[:1]  # Close the polygon
                    
                    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                    
                    for eval_idx, eval_summary in enumerate(comparison["evaluations"]):
                        values = []
                        for metric in common_metrics:
                            metric_data = comparison["metric_comparison"][metric]
                            for item in metric_data:
                                if item["eval_id"] == eval_summary["eval_id"]:
                                    values.append(item["value"])
                                    break
                        
                        values += values[:1]  # Close the polygon
                        ax.plot(angles, values, linewidth=2, label=eval_summary["name"])
                        ax.fill(angles, values, alpha=0.1)
                    
                    # Set labels and title
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(common_metrics)
                    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
                    plt.title('Evaluation Comparison')
                    plt.legend(loc='upper right')
                    
                    file_path = os.path.join(viz_dir, 'radar_comparison.png')
                    plt.savefig(file_path)
                    plt.close()
                    viz_files.append(file_path)
            
            return viz_files
            
        except Exception as e:
            evaluation_logger.error(f"Error generating comparison visualizations: {str(e)}")
            return []

#-----------------------------------------------------------------------------
# EVALUATION DASHBOARD
#-----------------------------------------------------------------------------

def start_evaluation_dashboard(evaluator: RagEvaluator, port: int = DASHBOARD_PORT):
    """
    Start a web-based dashboard for browsing RAG evaluation results.
    
    Args:
        evaluator: RAG evaluator instance
        port: Port to run the dashboard on
        
    Returns:
        Flask app instance (or None if Flask is not available)
    """
    if not REMOTE_AVAILABLE:
        evaluation_logger.error("Flask not available. Cannot start evaluation dashboard.")
        return None
    
    # Create Flask app
    app = flask.Flask(__name__)
    
    @app.route('/')
    def index():
        """Dashboard homepage."""
        # Get recent evaluations
        evaluations = evaluator.eval_db.list_evaluations(limit=20)
        
        return flask.render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 20px; }
                .tabs { display: flex; border-bottom: 1px solid #ddd; margin-bottom: 20px; }
                .tab { padding: 10px 15px; cursor: pointer; }
                .tab.active { border-bottom: 2px solid #007bff; font-weight: bold; }
                .tab-content { display: none; }
                .tab-content.active { display: block; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .refresh-btn { background-color: #007bff; color: white; border: none; padding: 5px 10px;
                                border-radius: 4px; cursor: pointer; }
                .metric-box { 
                    display: inline-block; 
                    width: 150px; 
                    height: 100px; 
                    margin: 10px;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                    vertical-align: top;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .metric-box h3 { margin-top: 0; font-size: 14px; }
                .metric-box .value { 
                    font-size: 24px; 
                    font-weight: bold; 
                    margin: 15px 0; 
                }
                .metric-box.blue { background-color: #e3f2fd; }
                .metric-box.green { background-color: #e8f5e9; }
                .metric-box.orange { background-color: #fff3e0; }
                .metric-box.purple { background-color: #f3e5f5; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>RAG Evaluation Dashboard</h1>
                
                <div class="tabs">
                    <div class="tab active" data-tab="evaluations">Evaluations</div>
                    <div class="tab" data-tab="comparison">Comparison</div>
                    <div class="tab" data-tab="new-evaluation">New Evaluation</div>
                </div>
                
                <div id="evaluations" class="tab-content active">
                    <div class="card">
                        <h2>Recent Evaluations</h2>
                        <button class="refresh-btn" onclick="location.reload()">Refresh</button>
                        
                        <table>
                            <tr>
                                <th>Name</th>
                                <th>Description</th>
                                <th>Start Time</th>
                                <th>Status</th>
                                <th>Actions</th>
                            </tr>
                            {% for eval in evaluations %}
                            <tr>
                                <td>{{ eval.name }}</td>
                                <td>{{ eval.description or 'N/A' }}</td>
                                <td>{{ eval.start_time }}</td>
                                <td>{{ eval.status }}</td>
                                <td>
                                    <a href="/evaluation/{{ eval.eval_id }}">View Details</a>
                                </td>
                            </tr>
                            {% endfor %}
                        </table>
                    </div>
                </div>
                
                <div id="comparison" class="tab-content">
                    <div class="card">
                        <h2>Compare Evaluations</h2>
                        
                        <form id="compare-form">
                            <p>Select evaluations to compare:</p>
                            <div id="eval-checkboxes">
                                {% for eval in evaluations %}
                                <div>
                                    <input type="checkbox" name="eval_ids" value="{{ eval.eval_id }}" id="eval_{{ eval.eval_id }}">
                                    <label for="eval_{{ eval.eval_id }}">{{ eval.name }} ({{ eval.start_time }})</label>
                                </div>
                                {% endfor %}
                            </div>
                            <button class="refresh-btn" type="button" onclick="compareEvaluations()">Compare</button>
                        </form>
                        
                        <div id="comparison-results"></div>
                    </div>
                </div>
                
                <div id="new-evaluation" class="tab-content">
                    <div class="card">
                        <h2>Run New Evaluation</h2>
                        
                        <form id="eval-form">
                            <div>
                                <label for="eval-name">Evaluation Name:</label>
                                <input type="text" id="eval-name" name="name" required>
                            </div>
                            <div>
                                <label for="eval-description">Description:</label>
                                <textarea id="eval-description" name="description"></textarea>
                            </div>
                            <div>
                                <label for="eval-queries">Queries (one per line):</label>
                                <textarea id="eval-queries" name="queries" rows="10" required></textarea>
                            </div>
                            <div>
                                <label for="eval-expected">Expected Responses (one per line, optional):</label>
                                <textarea id="eval-expected" name="expected" rows="10"></textarea>
                            </div>
                            <button class="refresh-btn" type="button" onclick="runEvaluation()">Run Evaluation</button>
                        </form>
                        
                        <div id="eval-status"></div>
                    </div>
                </div>
            </div>
            
            <script>
                // Tab switching
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.addEventListener('click', () => {
                        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                        
                        tab.classList.add('active');
                        document.getElementById(tab.dataset.tab).classList.add('active');
                    });
                });
                
                // Compare evaluations
                function compareEvaluations() {
                    const form = document.getElementById('compare-form');
                    const selected = Array.from(form.querySelectorAll('input[name="eval_ids"]:checked'))
                        .map(cb => cb.value);
                    
                    if (selected.length < 2) {
                        alert('Please select at least two evaluations to compare.');
                        return;
                    }
                    
                    // Create query string
                    const queryString = selected.map(id => `eval_id=${id}`).join('&');
                    
                    // Redirect to comparison page
                    window.location.href = `/compare?${queryString}`;
                }
                
                // Run new evaluation
                function runEvaluation() {
                    const form = document.getElementById('eval-form');
                    const name = form.querySelector('#eval-name').value;
                    const description = form.querySelector('#eval-description').value;
                    const queries = form.querySelector('#eval-queries').value;
                    const expected = form.querySelector('#eval-expected').value;
                    
                    if (!name || !queries) {
                        alert('Please provide a name and queries.');
                        return;
                    }
                    
                    // Disable form
                    form.querySelectorAll('input, textarea, button').forEach(el => {
                        el.disabled = true;
                    });
                    
                    const statusDiv = document.getElementById('eval-status');
                    statusDiv.innerHTML = '<p>Starting evaluation...</p>';
                    
                    // Submit evaluation request
                    fetch('/api/evaluate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            name,
                            description,
                            queries: queries.split('\\n').filter(q => q.trim()),
                            expected: expected ? expected.split('\\n').filter(e => e.trim()) : []
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            statusDiv.innerHTML = `<p>Error: ${data.error}</p>`;
                        } else {
                            statusDiv.innerHTML = `
                                <p>Evaluation started!</p>
                                <p>Evaluation ID: ${data.eval_id}</p>
                                <p><a href="/evaluation/${data.eval_id}">View Results</a></p>
                            `;
                        }
                    })
                    .catch(error => {
                        statusDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                    })
                    .finally(() => {
                        // Re-enable form
                        form.querySelectorAll('input, textarea, button').forEach(el => {
                            el.disabled = false;
                        });
                    });
                }
            </script>
        </body>
        </html>
        ''', evaluations=evaluations)
    
    @app.route('/evaluation/<eval_id>')
    def view_evaluation(eval_id):
        """View details of a specific evaluation."""
        evaluation = evaluator.eval_db.get_evaluation(eval_id)
        
        if not evaluation:
            return "Evaluation not found", 404
        
        return flask.render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Details - {{ evaluation.name }}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 20px; }
                .tabs { display: flex; border-bottom: 1px solid #ddd; margin-bottom: 20px; }
                .tab { padding: 10px 15px; cursor: pointer; }
                .tab.active { border-bottom: 2px solid #007bff; font-weight: bold; }
                .tab-content { display: none; }
                .tab-content.active { display: block; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow: auto; }
                .metric-box { 
                    display: inline-block; 
                    width: 150px; 
                    height: 100px; 
                    margin: 10px;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                    vertical-align: top;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .metric-box h3 { margin-top: 0; font-size: 14px; }
                .metric-box .value { 
                    font-size: 24px; 
                    font-weight: bold; 
                    margin: 15px 0; 
                }
                .metric-box.blue { background-color: #e3f2fd; }
                .metric-box.green { background-color: #e8f5e9; }
                .metric-box.orange { background-color: #fff3e0; }
                .metric-box.purple { background-color: #f3e5f5; }
                .visualization-container {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 20px;
                    margin-top: 20px;
                }
                .visualization-container img {
                    max-width: 500px;
                    max-height: 400px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Evaluation: {{ evaluation.name }}</h1>
                <p><a href="/">&laquo; Back to Dashboard</a></p>
                
                <div class="card">
                    <h2>Overview</h2>
                    <table>
                        <tr>
                            <th>Evaluation ID</th>
                            <td>{{ evaluation.eval_id }}</td>
                        </tr>
                        <tr>
                            <th>Description</th>
                            <td>{{ evaluation.description }}</td>
                        </tr>
                        <tr>
                            <th>Start Time</th>
                            <td>{{ evaluation.start_time }}</td>
                        </tr>
                        <tr>
                            <th>End Time</th>
                            <td>{{ evaluation.end_time or 'N/A' }}</td>
                        </tr>
                        <tr>
                            <th>Status</th>
                            <td>{{ evaluation.status }}</td>
                        </tr>
                        <tr>
                            <th>Queries</th>
                            <td>{{ evaluation.query_evaluations|length }}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="card">
                    <h2>Key Metrics</h2>
                    <div>
                        {% if evaluation.aggregate_metrics %}
                            <div class="metric-box blue">
                                <h3>Overall Score</h3>
                                <div class="value">{{ "%.2f"|format(evaluation.aggregate_metrics.overall_score|float) }}</div>
                            </div>
                            
                            <div class="metric-box green">
                                <h3>Accuracy</h3>
                                <div class="value">{{ "%.2f"|format(evaluation.aggregate_metrics.accuracy|float * 100) }}%</div>
                            </div>
                            
                            <div class="metric-box orange">
                                <h3>Avg Response Time</h3>
                                <div class="value">{{ "%.2f"|format(evaluation.aggregate_metrics.avg_response_time|float) }}s</div>
                            </div>
                            
                            <div class="metric-box purple">
                                <h3>Avg Similarity</h3>
                                <div class="value">{{ "%.2f"|format(evaluation.aggregate_metrics.avg_similarity|float) }}</div>
                            </div>
                        {% else %}
                            <p>No metrics available</p>
                        {% endif %}
                    </div>
                </div>
                
                <div class="tabs">
                    <div class="tab active" data-tab="results">Query Results</div>
                    <div class="tab" data-tab="metrics">All Metrics</div>
                    <div class="tab" data-tab="visualizations">Visualizations</div>
                    <div class="tab" data-tab="metadata">Metadata</div>
                </div>
                
                <div id="results" class="tab-content active">
                    <div class="card">
                        <h2>Query Results</h2>
                        <table>
                            <tr>
                                <th>#</th>
                                <th>Query</th>
                                <th>Response</th>
                                <th>Expected</th>
                                <th>Similarity</th>
                                <th>Time (s)</th>
                            </tr>
                            {% for result in evaluation.query_evaluations %}
                            <tr>
                                <td>{{ loop.index }}</td>
                                <td>{{ result.query }}</td>
                                <td>{{ result.response[:100] + '...' if result.response|length > 100 else result.response }}</td>
                                <td>{{ result.expected[:100] + '...' if result.expected and result.expected|length > 100 else result.expected or 'N/A' }}</td>
                                <td>{{ "%.2f"|format(result.similarity_score) if result.similarity_score else 'N/A' }}</td>
                                <td>{{ "%.2f"|format(result.response_time) if result.response_time else 'N/A' }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                    </div>
                </div>
                
                <div id="metrics" class="tab-content">
                    <div class="card">
                        <h2>All Metrics</h2>
                        {% if evaluation.aggregate_metrics %}
                        <pre>{{ evaluation.aggregate_metrics|tojson(indent=2) }}</pre>
                        {% else %}
                        <p>No metrics available</p>
                        {% endif %}
                    </div>
                </div>
                
                <div id="visualizations" class="tab-content">
                    <div class="card">
                        <h2>Visualizations</h2>
                        <div class="visualization-container">
                            <!-- Example visualizations - would need to be generated and served -->
                            <p>Visualizations not available in this demo.</p>
                        </div>
                    </div>
                </div>
                
                <div id="metadata" class="tab-content">
                    <div class="card">
                        <h2>Metadata</h2>
                        {% if evaluation.metadata %}
                        <pre>{{ evaluation.metadata|tojson(indent=2) }}</pre>
                        {% else %}
                        <p>No metadata available</p>
                        {% endif %}
                    </div>
                </div>
            </div>
            
            <script>
                // Tab switching
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.addEventListener('click', () => {
                        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                        
                        tab.classList.add('active');
                        document.getElementById(tab.dataset.tab).classList.add('active');
                    });
                });
            </script>
        </body>
        </html>
        ''', evaluation=evaluation)
    
    @app.route('/compare')
    def compare():
        """Compare multiple evaluations."""
        eval_ids = flask.request.args.getlist('eval_id')
        
        if not eval_ids or len(eval_ids) < 2:
            return "Need at least two evaluation IDs to compare", 400
        
        # Run comparison
        comparison = evaluator.compare_evaluations(eval_ids)
        
        return flask.render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Comparison</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 20px; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow: auto; }
                .visualization-container {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 20px;
                    margin-top: 20px;
                }
                .visualization-container img {
                    max-width: 500px;
                    max-height: 400px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Evaluation Comparison</h1>
                <p><a href="/">&laquo; Back to Dashboard</a></p>
                
                <div class="card">
                    <h2>Evaluations Being Compared</h2>
                    <table>
                        <tr>
                            <th>Name</th>
                            <th>Timestamp</th>
                            <th>Actions</th>
                        </tr>
                        {% for eval in comparison.evaluations %}
                        <tr>
                            <td>{{ eval.name }}</td>
                            <td>{{ eval.timestamp }}</td>
                            <td><a href="/evaluation/{{ eval.eval_id }}">View Details</a></td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                <div class="card">
                    <h2>Metric Comparison</h2>
                    
                    {% for metric, values in comparison.metric_comparison.items() %}
                    {% if metric in ['overall_score', 'accuracy', 'avg_similarity', 'avg_retrieval_score', 'avg_response_time'] %}
                    <h3>{{ metric }}</h3>
                    <table>
                        <tr>
                            <th>Evaluation</th>
                            <th>Value</th>
                        </tr>
                        {% for item in values %}
                        <tr>
                            <td>{{ item.name }}</td>
                            <td>{{ "%.4f"|format(item.value) }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                    {% endif %}
                    {% endfor %}
                </div>
                
                <div class="card">
                    <h2>Visualizations</h2>
                    <div class="visualization-container">
                        <!-- Example visualizations - would need to be generated and served -->
                        <p>Visualizations not available in this demo.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        ''', comparison=comparison)
    
    @app.route('/api/evaluate', methods=['POST'])
    def api_evaluate():
        """API endpoint for starting a new evaluation."""
        data = flask.request.json
        
        if not data:
            return flask.jsonify({"error": "No data provided"}), 400
        
        name = data.get('name')
        description = data.get('description')
        queries = data.get('queries', [])
        expected = data.get('expected', [])
        
        if not name or not queries:
            return flask.jsonify({"error": "Name and queries are required"}), 400
        
        # Start evaluation in a background thread
        def run_evaluation():
            try:
                evaluator.evaluate_dataset(
                    queries=queries,
                    expected_responses=expected,
                    name=name,
                    description=description
                )
            except Exception as e:
                evaluation_logger.error(f"Error running evaluation: {str(e)}")
        
        # Create context first to get the eval_id
        context = EvaluationContext(name, description)
        eval_id = context.eval_id
        
        # Store initial context
        evaluator.eval_db.store_evaluation(context)
        
        # Start evaluation thread
        thread = threading.Thread(target=run_evaluation)
        thread.daemon = True
        thread.start()
        
        return flask.jsonify({"success": True, "eval_id": eval_id})
    
    # Start the dashboard in a background thread
    def run_dashboard():
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    
    evaluation_logger.info(f"Evaluation dashboard started at http://localhost:{port}")
    return app

#-----------------------------------------------------------------------------
# COMMAND LINE INTERFACE
#-----------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Evaluation Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG_FILE,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--results-dir", "-d",
        default=DEFAULT_RESULTS_DIR,
        help="Directory for evaluation results"
    )
    
    parser.add_argument(
        "--log-dir", "-l",
        default=DEFAULT_LOG_DIR,
        help="Directory for logs"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start evaluation dashboard"
    )
    
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=DASHBOARD_PORT,
        help="Port for evaluation dashboard"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation"
    )
    
    parser.add_argument(
        "--queries-file",
        help="File containing evaluation queries (one per line)"
    )
    
    parser.add_argument(
        "--expected-file",
        help="File containing expected responses (one per line)"
    )
    
    parser.add_argument(
        "--name",
        default="cli_evaluation",
        help="Name for the evaluation"
    )
    
    parser.add_argument(
        "--description",
        help="Description for the evaluation"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run evaluation in parallel"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers"
    )
    
    parser.add_argument(
        "--compare",
        nargs='+',
        help="Compare evaluations by their IDs"
    )
    
    return parser.parse_args()

def configure_from_file(config_file: str) -> Dict[str, Any]:
    """
    Configure the evaluation framework from a JSON or YAML file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary with configuration settings
    """
    # Load configuration
    if not os.path.exists(config_file):
        evaluation_logger.error(f"Configuration file not found: {config_file}")
        return {}
    
    config = {}
    
    try:
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    evaluation_logger.error("YAML configuration requires PyYAML package")
                    return {}
            else:
                config = json.load(f)
    except Exception as e:
        evaluation_logger.error(f"Error loading configuration file: {e}")
        return {}
    
    return config

#-----------------------------------------------------------------------------
# EXAMPLE USAGE
#-----------------------------------------------------------------------------

# Example RAG pipeline function (to be replaced with actual implementation)
def example_rag_pipeline(query: str) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
    """
    Example RAG pipeline function for testing.
    
    Args:
        query: The query to process
        
    Returns:
        Either just the response or a tuple of (response, retrieved_docs)
    """
    # Simple mock responses
    responses = {
        "What is contract law?": "Contract law governs agreements between parties.",
        "Define negligence in legal terms.": "Negligence is the failure to exercise reasonable care.",
        "What are the elements of a valid contract?": "The elements of a valid contract include offer, acceptance, consideration, capacity, and legal purpose.",
        "Explain the doctrine of precedent.": "The doctrine of precedent is a legal principle that requires courts to follow previous judicial decisions when the same issues arise again."
    }
    
    # Mock retrieved documents
    retrieved_docs = []
    
    # If query is about contract law, return mock documents
    if "contract" in query.lower():
        retrieved_docs = [
            {
                "id": "doc1",
                "text": "Contract law is a body of law that governs agreements between parties. It forms the foundation of business relationships.",
                "metadata": {"source": "Legal Textbook", "relevance": 0.95}
            },
            {
                "id": "doc2",
                "text": "A contract requires offer, acceptance, consideration, capacity, and legal purpose to be valid and enforceable.",
                "metadata": {"source": "Case Study", "relevance": 0.85}
            }
        ]
    elif "negligence" in query.lower():
        retrieved_docs = [
            {
                "id": "doc3",
                "text": "Negligence is the failure to exercise reasonable care, which results in harm to another party.",
                "metadata": {"source": "Legal Dictionary", "relevance": 0.9}
            },
            {
                "id": "doc4",
                "text": "The elements of negligence include duty, breach, causation, and damages.",
                "metadata": {"source": "Case Law", "relevance": 0.8}
            }
        ]
    
    # Get response with 1-second delay to simulate processing time
    time.sleep(1)
    response = responses.get(query, "I don't have specific information on that legal topic.")
    
    # Return both response and retrieved documents
    return response, retrieved_docs

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Configure logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"rag_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    evaluation_logger.addHandler(file_handler)
    
    evaluation_logger.info("Starting RAG Pipeline Evaluation Framework")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        config = configure_from_file(args.config)
        evaluation_logger.info(f"Loaded configuration from {args.config}")
    
    # Initialize evaluator with example pipeline
    # In a real implementation, you would replace this with your actual RAG pipeline
    evaluator = RagEvaluator(
        rag_pipeline=example_rag_pipeline,
        parallel=args.parallel,
        max_workers=args.max_workers,
        results_dir=args.results_dir
    )
    
    # Compare evaluations if requested
    if args.compare:
        evaluation_logger.info(f"Comparing evaluations: {', '.join(args.compare)}")
        comparison_result = evaluator.compare_evaluations(args.compare)
        
        # Save comparison report
        comparison_file = os.path.join(
            args.results_dir, 
            f"comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_result, f, indent=4)
        
        evaluation_logger.info(f"Comparison results saved to {comparison_file}")
        print(f"Comparison results saved to {comparison_file}")
    
    # Run evaluation if requested
    if args.evaluate:
        if not args.queries_file:
            evaluation_logger.error("--queries-file is required for evaluation")
            print("Error: --queries-file is required for evaluation")
            return
        
        # Load queries
        try:
            with open(args.queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
        except Exception as e:
            evaluation_logger.error(f"Error loading queries file: {e}")
            print(f"Error loading queries file: {e}")
            return
        
        # Load expected responses if provided
        expected_responses = None
        if args.expected_file:
            try:
                with open(args.expected_file, 'r') as f:
                    expected_responses = [line.strip() for line in f if line.strip()]
            except Exception as e:
                evaluation_logger.error(f"Error loading expected responses file: {e}")
                print(f"Error loading expected responses file: {e}")
                return
        
        evaluation_logger.info(f"Running evaluation with {len(queries)} queries")
        print(f"Running evaluation with {len(queries)} queries...")
        
        # Run evaluation
        result = evaluator.evaluate_dataset(
            queries=queries,
            expected_responses=expected_responses,
            name=args.name,
            description=args.description
        )
        
        print(f"\nEvaluation complete: {result['name']} (ID: {result['eval_id']})")
        print(f"Results saved to: {result['report_file']}")
        
        # Print summary metrics
        if 'metrics' in result:
            print("\nSummary Metrics:")
            for metric, value in result['metrics'].items():
                if metric in ['overall_score', 'accuracy', 'avg_similarity', 'avg_response_time']:
                    if metric == 'accuracy':
                        print(f"  - {metric}: {value * 100:.2f}%")
                    else:
                        print(f"  - {metric}: {value:.4f}")
    
    # Start dashboard if requested
    if args.dashboard:
        evaluation_logger.info(f"Starting evaluation dashboard on port {args.dashboard_port}")
        print(f"Starting evaluation dashboard at http://localhost:{args.dashboard_port}")
        
        dashboard = start_evaluation_dashboard(
            evaluator=evaluator,
            port=args.dashboard_port
        )
        
        if dashboard:
            print("Dashboard started. Press Ctrl+C to stop...")
            try:
                # Keep main thread alive
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
        else:
            print("Failed to start dashboard. Make sure Flask is installed.")
    
    evaluation_logger.info("RAG Pipeline Evaluation Framework finished")

if __name__ == "__main__":
    main()

# Example configuration file (rag_evaluation_config.json):
"""
{
    "evaluation": {
        "results_dir": "rag_results",
        "log_dir": "rag_logs",
        "parallel": true,
        "max_workers": 4
    },
    "database": {
        "db_file": "rag_evaluation.db"
    },
    "metrics": {
        "similarity_threshold": 0.7,
        "use_transformers": true,
        "transformer_model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "dashboard": {
        "enabled": true,
        "port": 5001
    },
    "default_datasets": [
        {
            "name": "legal_questions",
            "description": "Basic legal questions for evaluation",
            "queries_file": "datasets/legal_questions.txt",
            "expected_file": "datasets/legal_answers.txt"
        },
        {
            "name": "contract_law",
            "description": "Contract law specific questions",
            "queries_file": "datasets/contract_questions.txt",
            "expected_file": "datasets/contract_answers.txt"
        }
    ]
}
"""

"""
USAGE EXAMPLES:

# Run evaluation with a dataset
python rag_evaluation.py --evaluate --queries-file queries.txt --expected-file expected.txt --name "my_evaluation"

# Start the evaluation dashboard
python rag_evaluation.py --dashboard

# Compare multiple evaluations
python rag_evaluation.py --compare eval_id_1 eval_id_2 eval_id_3

# Use a configuration file
python rag_evaluation.py --config rag_evaluation_config.json --evaluate

# Running a comprehensive evaluation with all options
python rag_evaluation.py --config rag_evaluation_config.json --evaluate --queries-file queries.txt --expected-file expected.txt --name "comprehensive_eval" --description "Comprehensive evaluation of RAG pipeline" --parallel --max-workers 8 --dashboard
"""


#-----------------------------------------------------------------------------
# EVALUATION DASHBOARD
#-----------------------------------------------------------------------------

def start_evaluation_dashboard(evaluator: RagEvaluator, port: int = DASHBOARD_PORT):
    """
    Start a web-based dashboard for browsing RAG evaluation results.
    
    Args:
        evaluator: RAG evaluator instance
        port: Port to run the dashboard on
        
    Returns:
        Flask app instance (or None if Flask is not available)
    """
    if not REMOTE_AVAILABLE:
        evaluation_logger.error("Flask not available. Cannot start evaluation dashboard.")
        return None
    
    # Create Flask app
    app = flask.Flask(__name__)
    
    @app.route('/')
    def index():
        """Dashboard homepage."""
        # Get recent evaluations
        evaluations = evaluator.eval_db.list_evaluations(limit=20)
        
        return flask.render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 20px; }
                .tabs { display: flex; border-bottom: 1px solid #ddd; margin-bottom: 20px; }
                .tab { padding: 10px 15px; cursor: pointer; }
                .tab.active { border-bottom: 2px solid #007bff; font-weight: bold; }
                .tab-content { display: none; }
                .tab-content.active { display: block; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .refresh-btn { background-color: #007bff; color: white; border: none; padding: 5px 10px;
                                border-radius: 4px; cursor: pointer; }
                .metric-box { 
                    display: inline-block; 
                    width: 150px; 
                    height: 100px; 
                    margin: 10px;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                    vertical-align: top;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .metric-box h3 { margin-top: 0; font-size: 14px; }
                .metric-box .value { 
                    font-size: 24px; 
                    font-weight: bold; 
                    margin: 15px 0; 
                }
                .metric-box.blue { background-color: #e3f2fd; }
                .metric-box.green { background-color: #e8f5e9; }
                .metric-box.orange { background-color: #fff3e0; }
                .metric-box.purple { background-color: #f3e5f5; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>RAG Evaluation Dashboard</h1>
                
                <div class="tabs">
                    <div class="tab active" data-tab="evaluations">Evaluations</div>
                    <div class="
