"""
logger.py - Logging configuration for the Phi-4 + PRISM system

This module provides logging setup and utilities for the fusion model system.
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logger(level: int = logging.INFO, 
                log_file: Optional[str] = None,
                log_to_console: bool = True,
                max_file_size: int = 10 * 1024 * 1024,  # 10 MB
                backup_count: int = 5) -> logging.Logger:
    """
    Set up the logger for the application.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (default: None)
        log_to_console: Whether to log to console (default: True)
        max_file_size: Maximum size of log file before rotation (default: 10 MB)
        backup_count: Number of backup files to keep (default: 5)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_default_log_file() -> str:
    """
    Get default log file path based on current date and time.
    
    Returns:
        Path to default log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"phi4_prism_{timestamp}.log")
    
    return log_file


class MetricsLogger:
    """Logger for performance and accuracy metrics."""
    
    def __init__(self, metrics_file: str):
        """
        Initialize metrics logger.
        
        Args:
            metrics_file: Path to metrics log file
        """
        self.metrics_file = metrics_file
        
        # Create directory if it doesn't exist
        metrics_dir = os.path.dirname(metrics_file)
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)
        
        # Initialize metrics file with header if it doesn't exist
        if not os.path.exists(metrics_file):
            with open(metrics_file, 'w') as f:
                f.write("timestamp,model,fusion_ratio,metric_name,metric_value\n")
    
    def log_metric(self, 
                  model: str,
                  fusion_ratio: str,
                  metric_name: str,
                  metric_value: float):
        """
        Log a metric value.
        
        Args:
            model: Model name
            fusion_ratio: Fusion ratio (e.g., "0.7/0.3")
            metric_name: Name of the metric
            metric_value: Value of the metric
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.metrics_file, 'a') as f:
            f.write(f"{timestamp},{model},{fusion_ratio},{metric_name},{metric_value}\n")
    
    def log_multiple_metrics(self,
                            model: str,
                            fusion_ratio: str,
                            metrics: dict):
        """
        Log multiple metrics at once.
        
        Args:
            model: Model name
            fusion_ratio: Fusion ratio (e.g., "0.7/0.3")
            metrics: Dictionary of metric_name: metric_value
        """
        for metric_name, metric_value in metrics.items():
            self.log_metric(model, fusion_ratio, metric_name, metric_value)


class QueryLogger:
    """Logger for queries and responses."""
    
    def __init__(self, query_log_file: str):
        """
        Initialize query logger.
        
        Args:
            query_log_file: Path to query log file
        """
        self.query_log_file = query_log_file
        
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(query_log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def log_query(self,
                 query: str,
                 response: str,
                 model: str,
                 fusion_ratio: str,
                 processing_time: float,
                 metadata: Optional[dict] = None):
        """
        Log a query and its response.
        
        Args:
            query: User query
            response: Model response
            model: Model name
            fusion_ratio: Fusion ratio (e.g., "0.7/0.3")
            processing_time: Query processing time in seconds
            metadata: Optional additional metadata
        """
        import json
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            "timestamp": timestamp,
            "model": model,
            "fusion_ratio": fusion_ratio,
            "query": query,
            "response": response,
            "processing_time": processing_time
        }
        
        # Add metadata if provided
        if metadata:
            log_entry["metadata"] = metadata
        
        with open(self.query_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")


"""
SUMMARY:
- Provides logging configuration for the Phi-4 + PRISM system
- Supports console and file logging with rotation
- Includes specialized loggers for metrics and queries
- Implements proper formatting for different log types
- Generates timestamped log files by default

TODO:
- Add support for structured logging formats (JSON)
- Implement log aggregation and analysis utilities
- Add support for remote logging services
- Enhance metric visualization capabilities
- Implement query performance trending
- Add support for log filtering and searching
- Implement secure logging for sensitive information
"""
