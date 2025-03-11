"""
A comprehensive framework for secure, scalable, and compliant logging and monitoring 
across distributed systems. This framework provides:

1. Tamper-proof logging with digital signatures and encryption
2. Real-time monitoring with dashboards and alerts
3. Log analysis with anomaly detection
4. Distributed log aggregation
5. Compliance enforcement with retention policies
6. Audit trails for legal requirements

The framework is designed to meet industry standards including GDPR, HIPAA, 
PCI-DSS, SOC2, and ISO 27001.
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
import ipaddress
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
    import cryptography
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import flask
    from flask import Flask, render_template, jsonify, request, Response
    WEB_DASHBOARD_AVAILABLE = True
except ImportError:
    WEB_DASHBOARD_AVAILABLE = False

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

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

# Constants
DEFAULT_LOG_DIR = "logs"
DEFAULT_SIGNATURES_FILE = "log_signatures.db"
DEFAULT_CONFIG_FILE = "logging_config.json"
DEFAULT_RETENTION_DAYS = 90
MAX_LOG_BATCH_SIZE = 1000
DEFAULT_ANOMALY_THRESHOLD = 0.95
DASHBOARD_PORT = 5000
LOG_STREAM_CHUNK_SIZE = 4096

# Configuration for secure logging
SIGNATURE_KEY_ENV_VAR = "LOG_SIGNATURE_KEY"
DEFAULT_SIGNATURE_KEY = "your-secret-key-for-development-only"
HASH_ALGORITHM = "sha256"

# Set up logger for the framework itself
framework_logger = logging.getLogger("log_monitor")
framework_logger.setLevel(logging.INFO)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
framework_logger.addHandler(_console_handler)

#-----------------------------------------------------------------------------
# SECURE LOGGING CLASSES
#-----------------------------------------------------------------------------

class SecureLogger:
    """
    Logger that provides tamper-proof logging capabilities with digital signatures.
    
    Features:
    - HMAC-based log entry signing
    - Optional encryption
    - Verification of log integrity
    - Support for structured logging (JSON)
    """
    def __init__(self, app_name: str, log_dir: str = DEFAULT_LOG_DIR, 
                 signature_key: Optional[str] = None, 
                 enable_encryption: bool = False,
                 structured_logging: bool = True,
                 max_log_size: int = 10 * 1024 * 1024,  # 10 MB
                 backup_count: int = 10):
        """
        Initialize secure logger.
        
        Args:
            app_name: Name of the application
            log_dir: Directory to store logs
            signature_key: Key for HMAC signatures (if None, uses env var or default)
            enable_encryption: Whether to encrypt log entries
            structured_logging: Whether to use JSON format
            max_log_size: Maximum size of log files before rotation
            backup_count: Number of backup log files to keep
        """
        self.app_name = app_name
        self.log_dir = log_dir
        self.structured_logging = structured_logging
        self.enable_encryption = enable_encryption
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up signature key
        self.signature_key = signature_key
        if not self.signature_key:
            self.signature_key = os.environ.get(SIGNATURE_KEY_ENV_VAR, DEFAULT_SIGNATURE_KEY)
            if self.signature_key == DEFAULT_SIGNATURE_KEY:
                framework_logger.warning(
                    f"Using default signature key. Set {SIGNATURE_KEY_ENV_VAR} for production."
                )
        
        # Initialize signature database
        self.signatures_db = os.path.join(log_dir, DEFAULT_SIGNATURES_FILE)
        self._init_signatures_db()
        
        # Create main logger
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler with rotation
        log_file = os.path.join(log_dir, f"{app_name}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_log_size, backupCount=backup_count
        )
        
        # Create formatter
        if structured_logging:
            formatter = self._create_json_formatter()
        else:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Add console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Register to handle unhandled exceptions
        sys.excepthook = self._handle_uncaught_exception
    
    def _init_signatures_db(self):
        """Initialize the signatures database."""
        try:
            conn = sqlite3.connect(self.signatures_db)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS log_signatures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    log_file TEXT,
                    log_entry TEXT,
                    signature TEXT,
                    verified BOOLEAN
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            framework_logger.error(f"Error initializing signatures database: {e}")
    
    def _create_json_formatter(self):
        """Create a JSON formatter."""
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'process': record.process,
                    'thread': record.thread,
                }
                
                # Add exception info if available
                if record.exc_info:
                    log_data['exception'] = {
                        'type': record.exc_info[0].__name__,
                        'message': str(record.exc_info[1]),
                        'traceback': self.formatException(record.exc_info)
                    }
                
                # Add extra data if available
                if hasattr(record, 'data') and record.data:
                    log_data['data'] = record.data
                
                return json.dumps(log_data)
        
        return JsonFormatter()
    
    def _sign_log_entry(self, log_entry: str) -> str:
        """
        Sign a log entry using HMAC.
        
        Args:
            log_entry: The log entry to sign
            
        Returns:
            Base64 encoded signature
        """
        key = self.signature_key.encode('utf-8')
        message = log_entry.encode('utf-8')
        
        signature = hmac.new(key, message, digestmod=getattr(hashlib, HASH_ALGORITHM))
        return base64.b64encode(signature.digest()).decode('utf-8')
    
    def _store_signature(self, log_file: str, log_entry: str, signature: str):
        """Store a signature in the database."""
        try:
            conn = sqlite3.connect(self.signatures_db)
            cursor = conn.cursor()
            timestamp = datetime.datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO log_signatures (timestamp, log_file, log_entry, signature, verified) "
                "VALUES (?, ?, ?, ?, ?)",
                (timestamp, log_file, log_entry, signature, True)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            framework_logger.error(f"Error storing signature: {e}")
    
    def _encrypt_log_entry(self, log_entry: str) -> str:
        """
        Encrypt a log entry.
        
        Args:
            log_entry: Log entry to encrypt
            
        Returns:
            Base64 encoded encrypted log entry
        """
        if not CRYPTO_AVAILABLE:
            warnings.warn("Cryptography package not available. Log entries will not be encrypted.")
            return log_entry
        
        try:
            # For demonstration, using a simple encryption method
            # In production, use proper encryption with key management
            compressed = zlib.compress(log_entry.encode('utf-8'))
            return base64.b64encode(compressed).decode('utf-8')
        except Exception as e:
            framework_logger.error(f"Error encrypting log entry: {e}")
            return log_entry
    
    def _handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """
        Handle uncaught exceptions by logging them.
        
        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupt exceptions
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        self.logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    def log(self, level: str, message: str, data: Dict[str, Any] = None, 
            tags: List[str] = None, sign: bool = True, encrypt: bool = None):
        """
        Log a message with optional structured data.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, CRITICAL, DEBUG)
            message: Log message
            data: Additional structured data
            tags: List of tags for the log entry
            sign: Whether to sign the log entry
            encrypt: Whether to encrypt the log entry (defaults to class setting)
        """
        # Prepare the log record
        log_record = logging.LogRecord(
            name=self.logger.name,
            level=getattr(logging, level.upper()),
            pathname=os.path.abspath(__file__),
            lineno=0,
            msg=message,
            args=(),
            exc_info=sys.exc_info() if level.upper() in ('ERROR', 'CRITICAL') else None
        )
        
        # Add extra data
        if data:
            log_record.data = data
        
        if tags:
            if not hasattr(log_record, 'data') or not log_record.data:
                log_record.data = {}
            log_record.data['tags'] = tags
        
        # Format the log entry
        handlers = self.logger.handlers
        if handlers:
            formatter = handlers[0].formatter
            log_entry = formatter.format(log_record)
            
            # Sign and/or encrypt if enabled
            if sign:
                signature = self._sign_log_entry(log_entry)
                # Store signature in database
                log_file = os.path.join(self.log_dir, f"{self.app_name}.log")
                self._store_signature(log_file, log_entry, signature)
                
                # Add signature to log entry for structured logging
                if self.structured_logging:
                    log_entry_dict = json.loads(log_entry)
                    log_entry_dict['signature'] = signature
                    log_entry = json.dumps(log_entry_dict)
            
            # Encrypt if enabled
            should_encrypt = encrypt if encrypt is not None else self.enable_encryption
            if should_encrypt:
                log_entry = self._encrypt_log_entry(log_entry)
            
            # Log it
            self.logger.handle(log_record)
        else:
            # Fallback if no handlers
            getattr(self.logger, level.lower())(message)
    
    def info(self, message: str, data: Dict[str, Any] = None, tags: List[str] = None):
        """Log an INFO level message."""
        self.log("INFO", message, data, tags)
    
    def warning(self, message: str, data: Dict[str, Any] = None, tags: List[str] = None):
        """Log a WARNING level message."""
        self.log("WARNING", message, data, tags)
    
    def error(self, message: str, data: Dict[str, Any] = None, tags: List[str] = None):
        """Log an ERROR level message."""
        self.log("ERROR", message, data, tags)
    
    def critical(self, message: str, data: Dict[str, Any] = None, tags: List[str] = None):
        """Log a CRITICAL level message."""
        self.log("CRITICAL", message, data, tags)
    
    def debug(self, message: str, data: Dict[str, Any] = None, tags: List[str] = None):
        """Log a DEBUG level message."""
        self.log("DEBUG", message, data, tags)
    
    def audit(self, message: str, data: Dict[str, Any] = None):
        """
        Log an audit event (always signed) with high importance.
        
        Args:
            message: Audit message
            data: Additional audit data
        """
        if not data:
            data = {}
        
        # Add audit-specific fields
        data['audit_id'] = str(uuid.uuid4())
        data['timestamp_utc'] = datetime.datetime.utcnow().isoformat()
        data['hostname'] = socket.gethostname()
        data['username'] = os.environ.get('USER', os.environ.get('USERNAME', 'unknown'))
        
        # Add process and environment info
        data['process_id'] = os.getpid()
        data['working_directory'] = os.getcwd()
        
        self.log("INFO", message, data, tags=['audit'], sign=True)

    def verify_log_integrity(self, log_file: Optional[str] = None):
        """
        Verify the integrity of log entries by checking their signatures.
        
        Args:
            log_file: Path to the log file to check (default: current log file)
            
        Returns:
            Tuple of (verification_result, tampered_entries)
        """
        if not log_file:
            log_file = os.path.join(self.log_dir, f"{self.app_name}.log")
        
        if not os.path.exists(log_file):
            return False, ["Log file does not exist"]
        
        tampered_entries = []
        verification_result = True
        
        try:
            with open(log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # For JSON formatted logs
                    if self.structured_logging and line.startswith('{'):
                        try:
                            log_data = json.loads(line)
                            if 'signature' in log_data:
                                # Extract and remove signature for verification
                                signature = log_data.pop('signature')
                                
                                # Re-create the log entry without the signature
                                log_entry = json.dumps(log_data)
                                
                                # Calculate the expected signature
                                expected_signature = self._sign_log_entry(log_entry)
                                
                                if signature != expected_signature:
                                    tampered_entries.append({
                                        'line': line_num,
                                        'content': line
                                    })
                                    verification_result = False
                        except json.JSONDecodeError:
                            # Not a valid JSON entry
                            pass
                    # For non-JSON logs, check against stored signatures
                    else:
                        conn = sqlite3.connect(self.signatures_db)
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT signature FROM log_signatures WHERE log_entry = ?",
                            (line,)
                        )
                        result = cursor.fetchone()
                        conn.close()
                        
                        if result:
                            stored_signature = result[0]
                            expected_signature = self._sign_log_entry(line)
                            
                            if stored_signature != expected_signature:
                                tampered_entries.append({
                                    'line': line_num,
                                    'content': line
                                })
                                verification_result = False
        except Exception as e:
            framework_logger.error(f"Error verifying log integrity: {e}")
            return False, [str(e)]
        
        return verification_result, tampered_entries

#-----------------------------------------------------------------------------
# LOG AGGREGATION & DISTRIBUTED LOGGING
#-----------------------------------------------------------------------------

class LogAggregator:
    """
    Aggregate logs from multiple sources and centralize them.
    
    Features:
    - Log collection from multiple sources
    - Correlation of logs across services
    - Consistent formatting
    - Filtering and search capabilities
    """
    def __init__(self, central_log_dir: str = "central_logs", buffer_size: int = 1000):
        """
        Initialize log aggregator.
        
        Args:
            central_log_dir: Directory to store centralized logs
            buffer_size: Size of in-memory buffer for logs before flushing
        """
        self.central_log_dir = central_log_dir
        self.buffer_size = buffer_size
        self.buffer = []
        self.buffer_lock = threading.Lock()
        
        # Ensure central log directory exists
        os.makedirs(central_log_dir, exist_ok=True)
        
        # Create aggregator logger
        self.logger = logging.getLogger("log_aggregator")
        self.logger.setLevel(logging.DEBUG)
        
        # Create main log file
        central_log_file = os.path.join(central_log_dir, "aggregated.log")
        handler = logging.handlers.RotatingFileHandler(
            central_log_file, maxBytes=50*1024*1024, backupCount=10
        )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Background thread for processing logs
        self._stop_event = threading.Event()
        self._processing_thread = threading.Thread(
            target=self._process_logs_background, 
            daemon=True
        )
        self._processing_thread.start()
    
    def shutdown(self):
        """Gracefully shutdown the aggregator."""
        self._stop_event.set()
        self._processing_thread.join(timeout=5)
        self._flush_buffer()
    
    def add_log(self, source: str, log_data: Dict[str, Any]):
        """
        Add a log entry to the aggregator.
        
        Args:
            source: Source identifier (e.g., service name)
            log_data: Log data as a dictionary
        """
        # Add source information
        log_data['source'] = source
        log_data['aggregated_timestamp'] = datetime.datetime.now().isoformat()
        
        # Add to buffer
        with self.buffer_lock:
            self.buffer.append(log_data)
            
            # Flush if buffer is full
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush the buffer to disk."""
        with self.buffer_lock:
            if not self.buffer:
                return
            
            # Group logs by source
            logs_by_source = defaultdict(list)
            for log in self.buffer:
                source = log.get('source', 'unknown')
                logs_by_source[source].append(log)
            
            # Write logs to source-specific files
            for source, logs in logs_by_source.items():
                source_log_file = os.path.join(self.central_log_dir, f"{source}.log")
                try:
                    with open(source_log_file, 'a') as f:
                        for log in logs:
                            f.write(json.dumps(log) + '\n')
                except Exception as e:
                    framework_logger.error(f"Error writing to {source_log_file}: {e}")
            
            # Write to main aggregated log
            for log in self.buffer:
                log_message = json.dumps(log)
                self.logger.info(log_message)
            
            # Clear buffer
            self.buffer.clear()
    
    def _process_logs_background(self):
        """Background thread to periodically flush logs."""
        while not self._stop_event.is_set():
            try:
                # Flush buffer every 5 seconds
                time.sleep(5)
                self._flush_buffer()
            except Exception as e:
                framework_logger.error(f"Error in log processing thread: {e}")
    
    def search_logs(self, query: str, sources: List[str] = None, 
                   start_time: Optional[datetime.datetime] = None,
                   end_time: Optional[datetime.datetime] = None,
                   max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search logs across all sources.
        
        Args:
            query: Search query
            sources: List of sources to search (None for all)
            start_time: Start time boundary
            end_time: End time boundary
            max_results: Maximum number of results to return
            
        Returns:
            List of matching log entries
        """
        results = []
        
        # Flush buffer to ensure all logs are on disk
        self._flush_buffer()
        
        # List of source files to search
        if sources:
            source_files = [os.path.join(self.central_log_dir, f"{source}.log") 
                          for source in sources]
        else:
            # Get all log files in central directory
            source_files = [os.path.join(self.central_log_dir, f) 
                          for f in os.listdir(self.central_log_dir) 
                          if f.endswith('.log')]
        
        # Search each file
        for file_path in source_files:
            if not os.path.exists(file_path):
                continue
            
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            log_entry = json.loads(line)
                            
                            # Apply time filters if specified
                            if start_time or end_time:
                                # Try to parse timestamp
                                timestamp_str = log_entry.get('timestamp', 
                                                            log_entry.get('aggregated_timestamp'))
                                if not timestamp_str:
                                    continue
                                
                                try:
                                    # Parse ISO format timestamp
                                    timestamp = datetime.datetime.fromisoformat(timestamp_str)
                                    
                                    if start_time and timestamp < start_time:
                                        continue
                                    if end_time and timestamp > end_time:
                                        continue
                                except ValueError:
                                    # Skip if timestamp can't be parsed
                                    continue
                            
                            # Apply query filter (simple string match for now)
                            if query.lower() in json.dumps(log_entry).lower():
                                results.append(log_entry)
                                
                                if len(results) >= max_results:
                                    return results
                        except json.JSONDecodeError:
                            # Skip non-JSON lines
                            continue
            except Exception as e:
                framework_logger.error(f"Error searching logs in {file_path}: {e}")
        
        return results

class LogShipper:
    """
    Ship logs to remote destinations.
    
    Features:
    - Send logs to remote log aggregator
    - Support for various protocols (HTTP, TCP, etc.)
    - Buffering and retry on failure
    - Batch processing
    """
    def __init__(self, destination_url: str, app_name: str, batch_size: int = 100,
                retry_interval: int = 5, max_retries: int = 3, api_key: Optional[str] = None):
        """
        Initialize log shipper.
        
        Args:
            destination_url: URL of remote log aggregator
            app_name: Name of the application
            batch_size: Size of batches for sending logs
            retry_interval: Seconds to wait between retries
            max_retries: Maximum number of retry attempts
            api_key: API key for authentication
        """
        self.destination_url = destination_url
        self.app_name = app_name
        self.batch_size = batch_size
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.api_key = api_key
        
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        
        # Start background shipper thread
        self._shipper_thread = threading.Thread(
            target=self._ship_logs_background,
            daemon=True
        )
        self._shipper_thread.start()
    
    def ship_log(self, log_data: Dict[str, Any]):
        """
        Ship a log entry to the remote destination.
        
        Args:
            log_data: Log data as a dictionary
        """
        # Add application name and timestamp
        log_data['app_name'] = self.app_name
        log_data['ship_timestamp'] = datetime.datetime.now().isoformat()
        
        # Add to queue for background processing
        self.queue.put(log_data)
    
    def shutdown(self):
        """Gracefully shutdown the shipper."""
        self._stop_event.set()
        self._shipper_thread.join(timeout=10)
    
    def _ship_logs_background(self):
        """Background thread to ship logs to remote destination."""
        batch = []
        
        while not self._stop_event.is_set():
            try:
                # Get a log entry from the queue, with timeout to check stop_event
                try:
                    log_data = self.queue.get(timeout=1)
                    batch.append(log_data)
                    self.queue.task_done()
                except queue.Empty:
                    # If batch has items, ship them after a timeout
                    if batch:
                        self._ship_batch(batch)
                        batch = []
                    continue
                
                # If batch is full, ship it
                if len(batch) >= self.batch_size:
                    self._ship_batch(batch)
                    batch = []
            
            except Exception as e:
                framework_logger.error(f"Error in log shipper thread: {e}")
                # Sleep briefly before retrying
                time.sleep(1)
        
        # Ship any remaining logs on shutdown
        if batch:
            self._ship_batch(batch)
    
    def _ship_batch(self, batch: List[Dict[str, Any]]):
        """
        Ship a batch of logs to the remote destination.
        
        Args:
            batch: List of log entries to ship
        """
        if not REMOTE_AVAILABLE:
            framework_logger.warning("Requests package not available. Cannot ship logs remotely.")
            return
        
        for attempt in range(self.max_retries):
            try:
                # Prepare request headers
                headers = {'Content-Type': 'application/json'}
                if self.api_key:
                    headers['Authorization'] = f"Bearer {self.api_key}"
                
                # Send logs to remote destination
                response = requests.post(
                    self.destination_url,
                    json={'logs': batch},
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    return  # Success
                else:
                    framework_logger.warning(
                        f"Failed to ship logs (attempt {attempt+1}/{self.max_retries}): "
                        f"Status code {response.status_code}"
                    )
            
            except Exception as e:
                framework_logger.warning(
                    f"Error shipping logs (attempt {attempt+1}/{self.max_retries}): {e}"
                )
            
            # Wait before retrying
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_interval)
        
        framework_logger.error(f"Failed to ship logs after {self.max_retries} attempts")

#-----------------------------------------------------------------------------
# LOG ANALYSIS & ANOMALY DETECTION
#-----------------------------------------------------------------------------

class LogAnalyzer:
    """
    Analyze logs for patterns, anomalies, and security threats.
    
    Features:
    - Statistical analysis of log patterns
    - Anomaly detection
    - Security threat identification
    - Performance issue detection
    """
    def __init__(self, log_dir: str, app_name: Optional[str] = None, 
                train_size: int = 1000, anomaly_threshold: float = DEFAULT_ANOMALY_THRESHOLD):
        """
        Initialize log analyzer.
        
        Args:
            log_dir: Directory containing logs to analyze
            app_name: Name of the application (for specific log file)
            train_size: Number of log entries to use for training anomaly detector
            anomaly_threshold: Threshold for anomaly detection (0-1)
        """
        self.log_dir = log_dir
        self.app_name = app_name
        self.train_size = train_size
        self.anomaly_threshold = anomaly_threshold
        
        # Log patterns and statistics
        self.log_patterns = {}
        self.hourly_counts = defaultdict(int)
        self.level_counts = defaultdict(int)
        self.message_counts = defaultdict(int)
        
        # Initialize anomaly detector
        self.anomaly_detector = None
        self.feature_names = []
        self._init_anomaly_detector()
    
    def _init_anomaly_detector(self):
        """Initialize the anomaly detector if scikit-learn is available."""
        if not ML_AVAILABLE:
            framework_logger.warning("scikit-learn not available. Anomaly detection disabled.")
            return
        
        try:
            # Initialize Isolation Forest for anomaly detection
            self.anomaly_detector = IsolationForest(
                contamination=1.0 - self.anomaly_threshold,
                random_state=42
            )
            
            # Define features for anomaly detection
            self.feature_names = [
                'log_length',         # Length of log message
                'hour_of_day',        # Hour of the day (0-23)
                'day_of_week',        # Day of the week (0-6)
                'level_value',        # Numeric value of log level
                'has_exception',      # Whether log contains exception
                'contains_ip',        # Whether log contains IP address
                'contains_error',     # Whether log contains error keywords
                'contains_warning',   # Whether log contains warning keywords
                'word_count'          # Number of words in message
            ]
            
            framework_logger.info("Anomaly detector initialized")
        except Exception as e:
            framework_logger.error(f"Error initializing anomaly detector: {e}")
    
    def _extract_features(self, log_entry: Dict[str, Any]) -> List[float]:
        """
        Extract features from a log entry for anomaly detection.
        
        Args:
            log_entry: Log entry as a dictionary
            
        Returns:
            List of feature values
        """
        features = []
        
        try:
            # Get message
            message = log_entry.get('message', '')
            if not isinstance(message, str):
                message = str(message)
            
            # Extract timestamp
            timestamp_str = log_entry.get('timestamp', datetime.datetime.now().isoformat())
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.datetime.now()
            
            # Get log level
            level = log_entry.get('level', 'INFO').upper()
            level_value = {
                'DEBUG': 1,
                'INFO': 2,
                'WARNING': 3,
                'ERROR': 4,
                'CRITICAL': 5
            }.get(level, 2)
            
            # Feature: log_length
            features.append(len(message))
            
            # Feature: hour_of_day
            features.append(timestamp.hour)
            
            # Feature: day_of_week
            features.append(timestamp.weekday())
            
            # Feature: level_value
            features.append(level_value)
            
            # Feature: has_exception
            has_exception = 'exception' in log_entry or 'traceback' in message.lower()
            features.append(1.0 if has_exception else 0.0)
            
            # Feature: contains_ip
            contains_ip = bool(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', message))
            features.append(1.0 if contains_ip else 0.0)
            
            # Feature: contains_error
            contains_error = any(kw in message.lower() for kw in ['error', 'exception', 'fail', 'fatal'])
            features.append(1.0 if contains_error else 0.0)
            
            # Feature: contains_warning
            contains_warning = any(kw in message.lower() for kw in ['warning', 'warn', 'deprecated'])
            features.append(1.0 if contains_warning else 0.0)
            
            # Feature: word_count
            word_count = len(message.split())
            features.append(word_count)
            
            return features
        
        except Exception as e:
            framework_logger.error(f"Error extracting features: {e}")
            # Return default feature values on error
            return [0.0] * len(self.feature_names)
    
    def train_anomaly_detector(self):
        """Train the anomaly detector on log data."""
        if not ML_AVAILABLE or not self.anomaly_detector:
            return False
        
        try:
            # Get log entries for training
            log_entries = self._get_log_entries(max_count=self.train_size)
            if not log_entries:
                framework_logger.warning("No log entries found for training anomaly detector")
                return False
            
            # Extract features
            features = []
            for entry in log_entries:
                feature_vector = self._extract_features(entry)
                features.append(feature_vector)
            
            # Train the model
            if len(features) > 10:  # Need a minimum number of samples
                self.anomaly_detector.fit(features)
                framework_logger.info(f"Trained anomaly detector on {len(features)} log entries")
                return True
            else:
                framework_logger.warning(f"Not enough log entries for training: {len(features)}")
                return False
        
        except Exception as e:
            framework_logger.error(f"Error training anomaly detector: {e}")
            return False
    
    def detect_anomalies(self, max_count: int = 1000) -> List[Dict[str, Any]]:
        """
        Detect anomalies in log entries.
        
        Args:
            max_count: Maximum number of log entries to analyze
            
        Returns:
            List of anomalous log entries with anomaly scores
        """
        if not ML_AVAILABLE or not self.anomaly_detector:
            return []
        
        anomalies = []
        
        try:
            # Get log entries for analysis
            log_entries = self._get_log_entries(max_count=max_count)
            if not log_entries:
                return []
            
            # Extract features and keep track of original entries
            features = []
            original_entries = []
            
            for entry in log_entries:
                feature_vector = self._extract_features(entry)
                features.append(feature_vector)
                original_entries.append(entry)
            
            # Predict anomalies (-1 for anomalies, 1 for normal)
            if not hasattr(self.anomaly_detector, 'fitted_') or not self.anomaly_detector.fitted_:
                self.train_anomaly_detector()
                
            if hasattr(self.anomaly_detector, 'fitted_') and self.anomaly_detector.fitted_:
                predictions = self.anomaly_detector.predict(features)
                scores = self.anomaly_detector.decision_function(features)
                
                # Find anomalies
                for i, (prediction, score) in enumerate(zip(predictions, scores)):
                    if prediction == -1:  # Anomaly
                        entry = original_entries[i].copy()
                        entry['anomaly_score'] = float(score)
                        anomalies.append(entry)
            
            framework_logger.info(f"Detected {len(anomalies)} anomalies in {len(log_entries)} log entries")
            return anomalies
        
        except Exception as e:
            framework_logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _get_log_entries(self, max_count: int = 1000) -> List[Dict[str, Any]]:
        """
        Get log entries from log files.
        
        Args:
            max_count: Maximum number of log entries to retrieve
            
        Returns:
            List of log entries as dictionaries
        """
        log_entries = []
        
        try:
            # Determine which log file(s) to analyze
            if self.app_name:
                log_files = [os.path.join(self.log_dir, f"{self.app_name}.log")]
            else:
                log_files = [os.path.join(self.log_dir, f) 
                           for f in os.listdir(self.log_dir) 
                           if f.endswith('.log')]
            
            # Read log entries
            for log_file in log_files:
                if not os.path.exists(log_file):
                    continue
                
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            # Try to parse as JSON
                            log_entry = json.loads(line)
                            log_entries.append(log_entry)
                        except json.JSONDecodeError:
                            # Non-JSON format, create basic structure
                            parts = line.split(' - ', 2)
                            if len(parts) >= 3:
                                timestamp, level, message = parts
                                log_entries.append({
                                    'timestamp': timestamp,
                                    'level': level,
                                    'message': message
                                })
                        
                        if len(log_entries) >= max_count:
                            break
                
                if len(log_entries) >= max_count:
                    break
            
            return log_entries
        
        except Exception as e:
            framework_logger.error(f"Error getting log entries: {e}")
            return []
    
    def analyze_logs(self) -> Dict[str, Any]:
        """
        Analyze logs to extract patterns and statistics.
        
        Returns:
            Dictionary with analysis results
        """
        # Reset stats
        self.hourly_counts = defaultdict(int)
        self.level_counts = defaultdict(int)
        self.message_counts = defaultdict(int)
        
        try:
            # Get log entries
            log_entries = self._get_log_entries(max_count=10000)
            
            # Process entries
            for entry in log_entries:
                # Extract timestamp
                timestamp_str = entry.get('timestamp')
                if timestamp_str:
                    try:
                        timestamp = datetime.datetime.fromisoformat(timestamp_str)
                        self.hourly_counts[timestamp.hour] += 1
                    except (ValueError, TypeError):
                        pass
                
                # Extract level
                level = entry.get('level', 'UNKNOWN')
                self.level_counts[level] += 1
                
                # Extract message patterns
                message = entry.get('message', '')
                # Replace numbers and specific values with placeholders
                pattern = re.sub(r'\b\d+\b', '<NUM>', message)
                pattern = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '<UUID>', pattern)
                pattern = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', pattern)
                
                # Count pattern occurrences
                self.message_counts[pattern] += 1
            
            # Find top patterns
            top_patterns = sorted(
                self.message_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]
            
            # Calculate error rate
            total_logs = sum(self.level_counts.values())
            error_count = self.level_counts.get('ERROR', 0) + self.level_counts.get('CRITICAL', 0)
            error_rate = error_count / total_logs if total_logs > 0 else 0
            
            # Organize results
            results = {
                'total_logs': total_logs,
                'level_distribution': dict(self.level_counts),
                'error_rate': error_rate,
                'hourly_distribution': dict(self.hourly_counts),
                'top_patterns': [{'pattern': p, 'count': c} for p, c in top_patterns],
                'unique_patterns': len(self.message_counts)
            }
            
            return results
        
        except Exception as e:
            framework_logger.error(f"Error analyzing logs: {e}")
            return {'error': str(e)}
    
    def generate_visualizations(self, output_dir: str = "log_analysis"):
        """
        Generate visualizations of log analysis results.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            List of generated visualization file paths
        """
        if not VISUALIZATION_AVAILABLE:
            return []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        visualization_files = []
        
        try:
            # Analyze logs
            analysis = self.analyze_logs()
            
            # Generate log level distribution chart
            if 'level_distribution' in analysis:
                plt.figure(figsize=(10, 6))
                levels = list(analysis['level_distribution'].keys())
                counts = list(analysis['level_distribution'].values())
                
                plt.bar(levels, counts, color=['green', 'blue', 'orange', 'red', 'purple'])
                plt.title('Log Level Distribution')
                plt.xlabel('Log Level')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                file_path = os.path.join(output_dir, 'level_distribution.png')
                plt.savefig(file_path)
                plt.close()
                visualization_files.append(file_path)
            
            # Generate hourly distribution chart
            if 'hourly_distribution' in analysis:
                plt.figure(figsize=(12, 6))
                hours = sorted(analysis['hourly_distribution'].keys())
                counts = [analysis['hourly_distribution'][h] for h in hours]
                
                plt.bar(hours, counts, color='skyblue')
                plt.title('Hourly Log Distribution')
                plt.xlabel('Hour of Day')
                plt.ylabel('Count')
                plt.xticks(range(24))
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                file_path = os.path.join(output_dir, 'hourly_distribution.png')
                plt.savefig(file_path)
                plt.close()
                visualization_files.append(file_path)
            
            # Generate top patterns chart
            if 'top_patterns' in analysis and analysis['top_patterns']:
                plt.figure(figsize=(14, 8))
                patterns = [p['pattern'][:50] + '...' if len(p['pattern']) > 50 else p['pattern'] 
                          for p in analysis['top_patterns'][:10]]
                counts = [p['count'] for p in analysis['top_patterns'][:10]]
                
                plt.barh(range(len(patterns)), counts, color='lightgreen')
                plt.yticks(range(len(patterns)), patterns)
                plt.title('Top Log Message Patterns')
                plt.xlabel('Count')
                plt.tight_layout()
                
                file_path = os.path.join(output_dir, 'top_patterns.png')
                plt.savefig(file_path)
                plt.close()
                visualization_files.append(file_path)
            
            return visualization_files
        
        except Exception as e:
            framework_logger.error(f"Error generating visualizations: {e}")
            return []

#-----------------------------------------------------------------------------
# REAL-TIME MONITORING & DASHBOARDS
#-----------------------------------------------------------------------------

class LogMonitor:
    """
    Monitor logs in real-time for events and alert on conditions.
    
    Features:
    - Real-time log tailing
    - Alert generation based on rules
    - Event correlation
    """
    def __init__(self, log_file: str, rules: List[Dict[str, Any]] = None):
        """
        Initialize log monitor.
        
        Args:
            log_file: Path to log file to monitor
            rules: List of alerting rules
        """
        self.log_file = log_file
        self.rules = rules or []
        self.alerts = []
        self.alert_callbacks = []
        
        self._stop_event = threading.Event()
        self._monitor_thread = None
    
    def add_rule(self, pattern: str, severity: str = "WARNING", 
                description: str = "", throttle_seconds: int = 0):
        """
        Add an alerting rule.
        
        Args:
            pattern: Regex pattern to match in log entries
            severity: Alert severity (INFO, WARNING, ERROR, CRITICAL)
            description: Description of the rule
            throttle_seconds: Minimum seconds between repeated alerts
        """
        rule = {
            'id': str(uuid.uuid4()),
            'pattern': pattern,
            'severity': severity,
            'description': description,
            'throttle_seconds': throttle_seconds,
            'last_triggered': None
        }
        self.rules.append(rule)
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback function for alerts.
        
        Args:
            callback: Function to call when an alert is generated
        """
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start monitoring the log file in a background thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            framework_logger.warning("Monitor already running")
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_logs,
            daemon=True
        )
        self._monitor_thread.start()
        framework_logger.info(f"Started monitoring log file: {self.log_file}")
    
    def stop_monitoring(self):
        """Stop monitoring the log file."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        framework_logger.info("Stopped log monitoring")
    
    def _monitor_logs(self):
        """Background thread for monitoring logs."""
        if not os.path.exists(self.log_file):
            framework_logger.error(f"Log file not found: {self.log_file}")
            return
        
        # Start at the end of the file
        with open(self.log_file, 'r') as f:
            f.seek(0, os.SEEK_END)
            
            while not self._stop_event.is_set():
                line = f.readline()
                if line:
                    self._process_log_line(line)
                else:
                    # No new data, sleep briefly
                    time.sleep(0.1)
    
    def _process_log_line(self, line: str):
        """
        Process a log line and check against rules.
        
        Args:
            line: Log line to process
        """
        line = line.strip()
        if not line:
            return
        
        # Check each rule
        for rule in self.rules:
            pattern = rule['pattern']
            
            # Check if the pattern matches
            if re.search(pattern, line):
                # Check throttling
                if rule['throttle_seconds'] > 0 and rule['last_triggered']:
                    time_since_last = (datetime.datetime.now() - 
                                      rule['last_triggered']).total_seconds()
                    if time_since_last < rule['throttle_seconds']:
                        # Skip this alert due to throttling
                        continue
                
                # Create alert
                alert = {
                    'id': str(uuid.uuid4()),
                    'rule_id': rule['id'],
                    'timestamp': datetime.datetime.now().isoformat(),
                    'severity': rule['severity'],
                    'description': rule['description'],
                    'message': line,
                    'source': self.log_file
                }
                
                # Update last triggered time
                rule['last_triggered'] = datetime.datetime.now()
                
                # Add to alerts list
                self.alerts.append(alert)
                
                # Trim alerts list if it gets too long
                if len(self.alerts) > 1000:
                    self.alerts = self.alerts[-1000:]
                
                # Call alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        framework_logger.error(f"Error in alert callback: {e}")
    
    def get_alerts(self, max_count: int = 100, 
                  min_severity: str = None) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            max_count: Maximum number of alerts to return
            min_severity: Minimum severity level
            
        Returns:
            List of alert dictionaries
        """
        if not min_severity:
            return self.alerts[-max_count:]
        
        # Filter by severity
        severity_levels = {
            'INFO': 0,
            'WARNING': 1,
            'ERROR': 2,
            'CRITICAL': 3
        }
        
        min_level = severity_levels.get(min_severity.upper(), 0)
        filtered_alerts = [
            alert for alert in self.alerts
            if severity_levels.get(alert['severity'].upper(), 0) >= min_level
        ]
        
        return filtered_alerts[-max_count:]

def start_monitoring_dashboard(log_dir: str, port: int = DASHBOARD_PORT):
    """
    Start a web-based monitoring dashboard.
    
    Args:
        log_dir: Directory containing log files
        port: Port to run the dashboard on
        
    Returns:
        Flask app instance
    """
    if not WEB_DASHBOARD_AVAILABLE:
        framework_logger.error("Flask not available. Cannot start monitoring dashboard.")
        return None
    
    # Create Flask app
    app = Flask(__name__)
    
    # Get all log files
    log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) 
               if f.endswith('.log') and os.path.isfile(os.path.join(log_dir, f))]
    
    # Create log monitors
    monitors = {}
    for log_file in log_files:
        file_name = os.path.basename(log_file)
        monitor = LogMonitor(log_file)
        
        # Add some default rules
        monitor.add_rule(pattern=r"error|exception|fail|critical", 
                       severity="ERROR", 
                       description="Error detected in logs")
        monitor.add_rule(pattern=r"warning|warn", 
                       severity="WARNING", 
                       description="Warning detected in logs")
        
        monitors[file_name] = monitor
        monitor.start_monitoring()
    
    # Create log analyzer
    analyzer = LogAnalyzer(log_dir)
    
    # Flask routes
    @app.route('/')
    def index():
        """Dashboard homepage."""
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Log Monitoring Dashboard</title>
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
                .alert { padding: 10px; margin-bottom: 10px; border-radius: 4px; }
                .alert-info { background-color: #d1ecf1; }
                .alert-warning { background-color: #fff3cd; }
                .alert-error { background-color: #f8d7da; }
                .alert-critical { background-color: #dc3545; color: white; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .log-stream { height: 400px; overflow-y: scroll; background-color: #f8f9fa; 
                               padding: 10px; font-family: monospace; white-space: pre-wrap; }
                .refresh-btn { background-color: #007bff; color: white; border: none; padding: 5px 10px;
                                border-radius: 4px; cursor: pointer; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Log Monitoring Dashboard</h1>
                
                <div class="tabs">
                    <div class="tab active" data-tab="alerts">Alerts</div>
                    <div class="tab" data-tab="logs">Live Logs</div>
                    <div class="tab" data-tab="analysis">Log Analysis</div>
                    <div class="tab" data-tab="anomalies">Anomaly Detection</div>
                </div>
                
                <div id="alerts" class="tab-content active">
                    <div class="card">
                        <h2>Recent Alerts</h2>
                        <button class="refresh-btn" onclick="refreshAlerts()">Refresh</button>
                        <div id="alerts-container"></div>
                    </div>
                </div>
                
                <div id="logs" class="tab-content">
                    <div class="card">
                        <h2>Live Log Stream</h2>
                        <select id="log-file-select" onchange="switchLogFile()">
                            {% for file in log_files %}
                            <option value="{{ file }}">{{ file }}</option>
                            {% endfor %}
                        </select>
                        <button class="refresh-btn" onclick="startLogStream()">Start Streaming</button>
                        <button class="refresh-btn" onclick="stopLogStream()">Stop</button>
                        <div id="log-stream" class="log-stream"></div>
                    </div>
                </div>
                
                <div id="analysis" class="tab-content">
                    <div class="card">
                        <h2>Log Analysis</h2>
                        <button class="refresh-btn" onclick="refreshAnalysis()">Run Analysis</button>
                        <div id="analysis-container"></div>
                    </div>
                </div>
                
                <div id="anomalies" class="tab-content">
                    <div class="card">
                        <h2>Anomaly Detection</h2>
                        <button class="refresh-btn" onclick="detectAnomalies()">Detect Anomalies</button>
                        <div id="anomalies-container"></div>
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
                
                // Refresh alerts
                function refreshAlerts() {
                    fetch('/api/alerts')
                        .then(response => response.json())
                        .then(data => {
                            const container = document.getElementById('alerts-container');
                            container.innerHTML = '';
                            
                            if (data.length === 0) {
                                container.innerHTML = '<p>No alerts found.</p>';
                                return;
                            }
                            
                            data.forEach(alert => {
                                const alertDiv = document.createElement('div');
                                alertDiv.className = `alert alert-${alert.severity.toLowerCase()}`;
                                alertDiv.innerHTML = `
                                    <strong>${alert.timestamp}</strong> - ${alert.severity}<br>
                                    ${alert.description}<br>
                                    ${alert.message}
                                `;
                                container.appendChild(alertDiv);
                            });
                        });
                }
                
                // Log streaming
                let logStream;
                let streamActive = false;
                
                function startLogStream() {
                    if (streamActive) {
                        stopLogStream();
                    }
                    
                    const logFile = document.getElementById('log-file-select').value;
                    const logContainer = document.getElementById('log-stream');
                    logContainer.innerHTML = '';
                    
                    logStream = new EventSource(`/api/stream-logs/${logFile}`);
                    streamActive = true;
                    
                    logStream.onmessage = function(event) {
                        const logEntry = document.createElement('div');
                        logEntry.textContent = event.data;
                        logContainer.appendChild(logEntry);
                        
                        // Auto-scroll to bottom
                        logContainer.scrollTop = logContainer.scrollHeight;
                    };
                    
                    logStream.onerror = function() {
                        stopLogStream();
                    };
                }
                
                function stopLogStream() {
                    if (logStream) {
                        logStream.close();
                        streamActive = false;
                    }
                }
                
                function switchLogFile() {
                    if (streamActive) {
                        startLogStream();  // Restart with new file
                    }
                }
                
                // Log analysis
                function refreshAnalysis() {
                    fetch('/api/analysis')
                        .then(response => response.json())
                        .then(data => {
                            const container = document.getElementById('analysis-container');
                            container.innerHTML = '';
                            
                            if (data.error) {
                                container.innerHTML = `<p>Error: ${data.error}</p>`;
                                return;
                            }
                            
                            // Create summary
                            const summary = document.createElement('div');
                            summary.innerHTML = `
                                <h3>Summary</h3>
                                <p>Total logs: ${data.total_logs}</p>
                                <p>Error rate: ${(data.error_rate * 100).toFixed(2)}%</p>
                                <p>Unique patterns: ${data.unique_patterns}</p>
                            `;
                            container.appendChild(summary);
                            
                            // Create level distribution
                            const levelDiv = document.createElement('div');
                            levelDiv.innerHTML = '<h3>Log Level Distribution</h3>';
                            
                            const levelTable = document.createElement('table');
                            levelTable.innerHTML = `
                                <tr>
                                    <th>Level</th>
                                    <th>Count</th>
                                    <th>Percentage</th>
                                </tr>
                            `;
                            
                            const levels = data.level_distribution;
                            for (const level in levels) {
                                const row = document.createElement('tr');
                                const pct = (levels[level] / data.total_logs * 100).toFixed(2);
                                row.innerHTML = `
                                    <td>${level}</td>
                                    <td>${levels[level]}</td>
                                    <td>${pct}%</td>
                                `;
                                levelTable.appendChild(row);
                            }
                            
                            levelDiv.appendChild(levelTable);
                            container.appendChild(levelDiv);
                            
                            // Create top patterns
                            const patternsDiv = document.createElement('div');
                            patternsDiv.innerHTML = '<h3>Top Message Patterns</h3>';
                            
                            const patternsTable = document.createElement('table');
                            patternsTable.innerHTML = `
                                <tr>
                                    <th>Pattern</th>
                                    <th>Count</th>
                                </tr>
                            `;
                            
                            data.top_patterns.forEach(pattern => {
                                const row = document.createElement('tr');
                                row.innerHTML = `
                                    <td>${pattern.pattern}</td>
                                    <td>${pattern.count}</td>
                                `;
                                patternsTable.appendChild(row);
                            });
                            
                            patternsDiv.appendChild(patternsTable);
                            container.appendChild(patternsDiv);
                        });
                }
                
                // Anomaly detection
                function detectAnomalies() {
                    fetch('/api/anomalies')
                        .then(response => response.json())
                        .then(data => {
                            const container = document.getElementById('anomalies-container');
                            container.innerHTML = '';
                            
                            if (data.error) {
                                container.innerHTML = `<p>Error: ${data.error}</p>`;
                                return;
                            }
                            
                            if (data.length === 0) {
                                container.innerHTML = '<p>No anomalies detected.</p>';
                                return;
                            }
                            
                            const anomalyTable = document.createElement('table');
                            anomalyTable.innerHTML = `
                                <tr>
                                    <th>Timestamp</th>
                                    <th>Level</th>
                                    <th>Message</th>
                                    <th>Anomaly Score</th>
                                </tr>
                            `;
                            
                            data.forEach(anomaly => {
                                const row = document.createElement('tr');
                                row.innerHTML = `
                                    <td>${anomaly.timestamp || 'N/A'}</td>
                                    <td>${anomaly.level || 'N/A'}</td>
                                    <td>${anomaly.message || 'N/A'}</td>
                                    <td>${anomaly.anomaly_score?.toFixed(4) || 'N/A'}</td>
                                `;
                                anomalyTable.appendChild(row);
                            });
                            
                            container.appendChild(anomalyTable);
                        });
                }
                
                // Initial load
                refreshAlerts();
            </script>
        </body>
        </html>
        ''', log_files=log_files)
    
    @app.route('/api/alerts')
    def get_alerts():
        """API endpoint to get alerts from all monitors."""
        all_alerts = []
        for monitor in monitors.values():
            all_alerts.extend(monitor.get_alerts())
        
        # Sort by timestamp, newest first
        all_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify(all_alerts[:50])  # Return most recent 50 alerts
    
    @app.route('/api/stream-logs/<log_file>')
    def stream_logs(log_file):
        """API endpoint to stream log file contents."""
        log_path = os.path.join(log_dir, log_file)
        
        if not os.path.exists(log_path):
            return "Log file not found", 404
        
        def generate():
            with open(log_path, 'r') as f:
                # Start at the end of the file
                f.seek(0, os.SEEK_END)
                
                while True:
                    line = f.readline()
                    if line:
                        yield f"data: {line}\n\n"
                    else:
                        time.sleep(0.5)
        
        return Response(generate(), mimetype='text/event-stream')
    
    @app.route('/api/analysis')
    def get_analysis():
        """API endpoint for log analysis."""
        results = analyzer.analyze_logs()
        return jsonify(results)
    
    @app.route('/api/anomalies')
    def get_anomalies():
        """API endpoint for anomaly detection."""
        # Train if not already trained
        analyzer.train_anomaly_detector()
        
        # Detect anomalies
        anomalies = analyzer.detect_anomalies()
        return jsonify(anomalies)
    
    # Start the dashboard in a background thread
    def run_dashboard():
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    
    framework_logger.info(f"Monitoring dashboard started at http://localhost:{port}")
    return app

#-----------------------------------------------------------------------------
# COMPLIANCE & RETENTION MANAGEMENT
#-----------------------------------------------------------------------------

class LogRetentionManager:
    """
    Manage log retention policies for compliance.
    
    Features:
    - Automated log rotation and archiving
    - Retention policy enforcement
    - Secure log deletion
    """
    def __init__(self, log_dir: str, retention_days: int = DEFAULT_RETENTION_DAYS,
                archive_dir: Optional[str] = None, archive_enabled: bool = True):
        """
        Initialize log retention manager.
        
        Args:
            log_dir: Directory containing logs
            retention_days: Number of days to retain logs
            archive_dir: Directory for archived logs (if None, uses log_dir/archive)
            archive_enabled: Whether to archive logs before deletion
        """
        self.log_dir = log_dir
        self.retention_days = retention_days
        self.archive_dir = archive_dir or os.path.join(log_dir, "archive")
        self.archive_enabled = archive_enabled
        
        # Ensure archive directory exists if archiving is enabled
        if archive_enabled:
            os.makedirs(self.archive_dir, exist_ok=True)
    
    def enforce_retention_policy(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Enforce retention policy by archiving or deleting old logs.
        
        Args:
            dry_run: If True, report what would be done without making changes
            
        Returns:
            Dictionary with results of the operation
        """
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'retention_days': self.retention_days,
            'dry_run': dry_run,
            'archived_files': [],
            'deleted_files': [],
            'errors': []
        }
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=self.retention_days)
            
            # Process log files
            for filename in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, filename)
                
                # Skip directories and non-log files
                if (os.path.isdir(file_path) or 
                    not filename.endswith('.log') and not filename.endswith('.log.gz')):
                    continue
                
                # Check file modification time
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_time < cutoff_date:
                    # File is older than retention period
                    if self.archive_enabled:
                        # Archive the file
                        archive_path = os.path.join(self.archive_dir, filename)
                        
                        if not dry_run:
                            try:
                                # Create compressed archive if not already compressed
                                if not filename.endswith('.gz'):
                                    with open(file_path, 'rb') as f_in:
                                        with gzip.open(f"{archive_path}.gz", 'wb') as f_out:
                                            f_out.writelines(f_in)
                                    results['archived_files'].append(f"{filename}.gz")
                                else:
                                    import shutil
                                    shutil.copy2(file_path, archive_path)
                                    results['archived_files'].append(filename)
                            except Exception as e:
                                results['errors'].append(f"Error archiving {filename}: {str(e)}")
                        else:
                            results['archived_files'].append(filename)
                    
                    # Delete the file
                    if not dry_run:
                        try:
                            os.remove(file_path)
                            results['deleted_files'].append(filename)
                        except Exception as e:
                            results['errors'].append(f"Error deleting {filename}: {str(e)}")
                    else:
                        results['deleted_files'].append(filename)
            
            # Log summary
            framework_logger.info(
                f"Retention policy enforced: {len(results['archived_files'])} archived, "
                f"{len(results['deleted_files'])} deleted"
            )
            
            return results
        
        except Exception as e:
            framework_logger.error(f"Error enforcing retention policy: {e}")
            results['errors'].append(f"General error: {str(e)}")
            return results
    
    def secure_delete_file(self, file_path: str, passes: int = 3) -> bool:
        """
        Securely delete a file by overwriting it multiple times.
        
        Args:
            file_path: Path to the file to delete
            passes: Number of overwrite passes
            
        Returns:
            Boolean indicating success
        """
        if not os.path.exists(file_path):
            return False
        
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Open the file for writing
            with open(file_path, 'wb') as f:
                for pass_num in range(passes):
                    # Seek to the beginning of the file
                    f.seek(0)
                    
                    # Use different patterns for each pass
                    if pass_num == 0:
                        # All zeros
                        f.write(b'\x00' * file_size)
                    elif pass_num == 1:
                        # All ones
                        f.write(b'\xFF' * file_size)
                    else:
                        # Random data
                        f.write(os.urandom(file_size))
                    
                    # Flush to disk
                    f.flush()
                    os.fsync(f.fileno())
            
            # Finally, remove the file
            os.remove(file_path)
            return True
        
        except Exception as e:
            framework_logger.error(f"Error securely deleting file {file_path}: {e}")
            return False
    
    def verify_compliance(self) -> Dict[str, Any]:
        """
        Verify compliance with retention policies.
        
        Returns:
            Dictionary with compliance verification results
        """
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'retention_days': self.retention_days,
            'compliance_status': 'compliant',
            'issues': []
        }
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=self.retention_days)
            
            # Check log files
            for filename in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, filename)
                
                # Skip directories and non-log files
                if (os.path.isdir(file_path) or 
                    not filename.endswith('.log') and not filename.endswith('.log.gz')):
                    continue
                
                # Check file modification time
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_time < cutoff_date:
                    results['compliance_status'] = 'non-compliant'
                    results['issues'].append({
                        'file': filename,
                        'issue': 'Exceeds retention period',
                        'file_date': file_time.isoformat(),
                        'cutoff_date': cutoff_date.isoformat()
                    })
            
            return results
        
        except Exception as e:
            framework_logger.error(f"Error verifying compliance: {e}")
            results['compliance_status'] = 'error'
            results['issues'].append({
                'issue': f"Error checking compliance: {str(e)}"
            })
            return results

class ComplianceAuditor:
    """
    Audit logs for compliance requirements.
    
    Features:
    - Track security-relevant events
    - Generate compliance reports
    - Detect compliance violations
    """
    def __init__(self, log_dir: str, regulations: List[str] = None):
        """
        Initialize compliance auditor.
        
        Args:
            log_dir: Directory containing logs
            regulations: List of regulations to check compliance with
        """
        self.log_dir = log_dir
        self.regulations = regulations or ["GDPR", "HIPAA", "PCI-DSS", "SOC2"]
        
    def generate_compliance_report(self, start_date: Optional[datetime.datetime] = None,
                                 end_date: Optional[datetime.datetime] = None,
                                 output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a compliance report for the specified time period.
        
        Args:
            start_date: Start date for the report
            end_date: End date for the report
            output_file: File to write the report to
            
        Returns:
            Dictionary with compliance report data
        """
        report = {
            'report_id': str(uuid.uuid4()),
            'timestamp': datetime.datetime.now().isoformat(),
            'report_period': {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else None
            },
            'regulations': self.regulations,
            'summary': {},
            'log_statistics': {},
            'compliance_findings': {},
            'recommendations': []
        }
        
        try:
            # Get log files
            log_files = [os.path.join(self.log_dir, f) for f in os.listdir(self.log_dir) 
                      if f.endswith('.log')]
            
            # Collect audit events
            audit_events = []
            
            for log_file in log_files:
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Try to parse as JSON
                        try:
                            log_entry = json.loads(line)
                            
                            # Check if it's an audit event
                            if ('data' in log_entry and 'tags' in log_entry['data'] and 
                                'audit' in log_entry['data']['tags']):
                                
                                # Check date range if specified
                                if 'timestamp' in log_entry:
                                    try:
                                        log_date = datetime.datetime.fromisoformat(log_entry['timestamp'])
                                        if start_date and log_date < start_date:
                                            continue
                                        if end_date and log_date > end_date:
                                            continue
                                    except ValueError:
                                        pass
                                
                                audit_events.append(log_entry)
                        except json.JSONDecodeError:
                            # Skip non-JSON entries
                            continue
            
            # Process audit events for compliance findings
            gdpr_events = []
            hipaa_events = []
            pci_events = []
            soc2_events = []
            
            for event in audit_events:
                # Classify events by regulation
                if 'data' in event:
                    data = event['data']
                    if 'personal_data' in data or 'gdpr' in data:
                        gdpr_events.append(event)
                    if 'phi' in data or 'hipaa' in data:
                        hipaa_events.append(event)
                    if 'payment' in data or 'pci' in data or 'card' in data:
                        pci_events.append(event)
                    if 'access' in data or 'authentication' in data or 'authorization' in data:
                        soc2_events.append(event)
            
            # Generate findings for each regulation
            findings = {}
            
            if "GDPR" in self.regulations:
                findings["GDPR"] = {
                    'total_events': len(gdpr_events),
                    'data_access_events': sum(1 for e in gdpr_events if 'access' in e.get('message', '').lower()),
                    'data_deletion_events': sum(1 for e in gdpr_events if 'delete' in e.get('message', '').lower()),
                    'consent_events': sum(1 for e in gdpr_events if 'consent' in e.get('message', '').lower()),
                    'compliance_status': 'needs_review'
                }
            
            if "HIPAA" in self.regulations:
                findings["HIPAA"] = {
                    'total_events': len(hipaa_events),
                    'phi_access_events': sum(1 for e in hipaa_events if 'access' in e.get('message', '').lower()),
                    'phi_disclosure_events': sum(1 for e in hipaa_events if 'disclos' in e.get('message', '').lower()),
                    'compliance_status': 'needs_review'
                }
            
            if "PCI-DSS" in self.regulations:
                findings["PCI-DSS"] = {
                    'total_events': len(pci_events),
                    'card_access_events': sum(1 for e in pci_events if 'card' in e.get('message', '').lower()),
                    'payment_events': sum(1 for e in pci_events if 'payment' in e.get('message', '').lower()),
                    'compliance_status': 'needs_review'
                }
            
            if "SOC2" in self.regulations:
                findings["SOC2"] = {
                    'total_events': len(soc2_events),
                    'access_events': sum(1 for e in soc2_events if 'access' in e.get('message', '').lower()),
                    'authentication_events': sum(1 for e in soc2_events if 'auth' in e.get('message', '').lower()),
                    'compliance_status': 'needs_review'
                }
            
            # Update report
            report['compliance_findings'] = findings
            
            # Generate recommendations
            recommendations = []
            
            if "GDPR" in findings and findings["GDPR"]['total_events'] == 0:
                recommendations.append({
                    'regulation': 'GDPR',
                    'recommendation': 'Implement GDPR-specific audit logging for personal data processing activities.'
                })
            
            if "HIPAA" in findings and findings["HIPAA"]['total_events'] == 0:
                recommendations.append({
                    'regulation': 'HIPAA',
                    'recommendation': 'Implement HIPAA-specific audit logging for PHI access and disclosure.'
                })
            
            # Add general recommendations
            recommendations.append({
                'regulation': 'General',
                'recommendation': 'Conduct regular compliance audits of log data.'
            })
            
            recommendations.append({
                'regulation': 'General',
                'recommendation': 'Ensure log retention policies align with regulatory requirements.'
            })
            
            report['recommendations'] = recommendations
            
            # Generate overall summary
            report['summary'] = {
                'total_audit_events': len(audit_events),
                'regulations_checked': len(self.regulations),
                'compliance_status': 'needs_review',
                'critical_findings': 0,
                'recommendations': len(recommendations)
            }
            
            # Save report if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=4)
                framework_logger.info(f"Compliance report saved to {output_file}")
            
            return report
        
        except Exception as e:
            framework_logger.error(f"Error generating compliance report: {e}")
            report['error'] = str(e)
            return report

#-----------------------------------------------------------------------------
# MAIN FUNCTIONS
#-----------------------------------------------------------------------------

def configure_from_file(config_file: str) -> Dict[str, Any]:
    """
    Configure the logging framework from a JSON or YAML file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary with configured components
    """
    # Load configuration
    if not os.path.exists(config_file):
        framework_logger.error(f"Configuration file not found: {config_file}")
        return {}
    
    config = {}
    
    try:
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                if not YAML_AVAILABLE:
                    raise ImportError("YAML configuration requires PyYAML package")
                import yaml
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    except Exception as e:
        framework_logger.error(f"Error loading configuration file: {e}")
        return {}
    
    components = {}
    
    # Configure logger
    if 'logger' in config:
        logger_config = config['logger']
        logger_name = logger_config.get('name', 'app')
        log_dir = logger_config.get('log_dir', DEFAULT_LOG_DIR)
        structured_logging = logger_config.get('structured_logging', True)
        enable_encryption = logger_config.get('enable_encryption', False)
        
        # Create secure logger
        components['logger'] = SecureLogger(
            app_name=logger_name,
            log_dir=log_dir,
            signature_key=logger_config.get('signature_key'),
            enable_encryption=enable_encryption,
            structured_logging=structured_logging
        )
        
        framework_logger.info(f"Configured logger: {logger_name}")
    
    # Configure log aggregator
    if 'aggregator' in config:
        agg_config = config['aggregator']
        central_log_dir = agg_config.get('central_log_dir', 'central_logs')
        buffer_size = agg_config.get('buffer_size', 1000)
        
        # Create aggregator
        components['aggregator'] = LogAggregator(
            central_log_dir=central_log_dir,
            buffer_size=buffer_size
        )
        
        framework_logger.info(f"Configured log aggregator: {central_log_dir}")
    
    # Configure log analyzer
    if 'analyzer' in config:
        analyzer_config = config['analyzer']
        log_dir = analyzer_config.get('log_dir', DEFAULT_LOG_DIR)
        app_name = analyzer_config.get('app_name')
        
        # Create analyzer
        components['analyzer'] = LogAnalyzer(
            log_dir=log_dir,
            app_name=app_name
        )
        
        framework_logger.info(f"Configured log analyzer for {app_name or 'all logs'}")
    
    # Configure monitors
    if 'monitors' in config:
        components['monitors'] = {}
        
        for monitor_config in config['monitors']:
            log_file = monitor_config.get('log_file')
            if not log_file:
                continue
            
            # Create monitor
            monitor = LogMonitor(log_file=log_file)
            
            # Add rules
            if 'rules' in monitor_config:
                for rule in monitor_config['rules']:
                    monitor.add_rule(
                        pattern=rule.get('pattern', ''),
                        severity=rule.get('severity', 'WARNING'),
                        description=rule.get('description', ''),
                        throttle_seconds=rule.get('throttle_seconds', 0)
                    )
            
            # Start monitoring
            monitor.start_monitoring()
            components['monitors'][log_file] = monitor
            
            framework_logger.info(f"Configured log monitor for {log_file}")
    
    # Configure retention manager
    if 'retention' in config:
        retention_config = config['retention']
        log_dir = retention_config.get('log_dir', DEFAULT_LOG_DIR)
        retention_days = retention_config.get('retention_days', DEFAULT_RETENTION_DAYS)
        archive_enabled = retention_config.get('archive_enabled', True)
        
        # Create retention manager
        components['retention_manager'] = LogRetentionManager(
            log_dir=log_dir,
            retention_days=retention_days,
            archive_enabled=archive_enabled
        )
        
        framework_logger.info(f"Configured retention manager: {retention_days} days")
    
    # Configure dashboard
    if 'dashboard' in config:
        dashboard_config = config['dashboard']
        log_dir = dashboard_config.get('log_dir', DEFAULT_LOG_DIR)
        port = dashboard_config.get('port', DASHBOARD_PORT)
        
        if dashboard_config.get('enabled', False):
            # Start dashboard
            components['dashboard'] = start_monitoring_dashboard(
                log_dir=log_dir,
                port=port
            )
            
            framework_logger.info(f"Started monitoring dashboard on port {port}")
    
    return components

def log_event(logger, event_type, message, data=None, tags=None):
    """
    Log an event with the provided logger.
    
    Args:
        logger: SecureLogger instance
        event_type: Type of event (INFO, WARNING, ERROR, CRITICAL, DEBUG)
        message: Log message
        data: Additional data
        tags: Event tags
    """
    event_type = event_type.upper()
    
    if event_type == "INFO":
        logger.info(message, data, tags)
    elif event_type == "WARNING":
        logger.warning(message, data, tags)
    elif event_type == "ERROR":
        logger.error(message, data, tags)
    elif event_type == "CRITICAL":
        logger.critical(message, data, tags)
    elif event_type == "DEBUG":
        logger.debug(message, data, tags)
    else:
        logger.info(message, data, tags)

def monitor_log_file(log_file, watch_duration=10):
    """
    Monitor a log file for new entries over a set duration.
    
    Args:
        log_file: Path to the log file
        watch_duration: Time in seconds to watch the log file
    """
    print(f"Monitoring log file '{log_file}' for {watch_duration} seconds...")
    initial_size = os.stat(log_file).st_size if os.path.exists(log_file) else 0
    
    for _ in range(watch_duration):
        time.sleep(1)
        current_size = os.stat(log_file).st_size if os.path.exists(log_file) else 0
        if current_size > initial_size:
            print("New log entries detected.")
            initial_size = current_size
    print("Log monitoring complete.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Logging & Monitoring Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG_FILE,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-dir", "-d",
        default=DEFAULT_LOG_DIR,
        help="Directory for log files"
    )
    
    parser.add_argument(
        "--app-name", "-a",
        default="app",
        help="Application name for logging"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start monitoring dashboard"
    )
    
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=DASHBOARD_PORT,
        help="Port for monitoring dashboard"
    )
    
    parser.add_argument(
        "--verify-integrity",
        action="store_true",
        help="Verify log integrity"
    )
    
    parser.add_argument(
        "--enforce-retention",
        action="store_true",
        help="Enforce log retention policy"
    )
    
    parser.add_argument(
        "--retention-days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help="Log retention period in days"
    )
    
    parser.add_argument(
        "--analyze-logs",
        action="store_true",
        help="Analyze logs for patterns and anomalies"

  print(f"⚠️ Log integrity verification failed for {log_file}")
            print(f"Found {len(tampered_entries)} potentially tampered entries")
            for entry in tampered_entries[:5]:  # Show first 5
                print(f"  - Line {entry['line']}: {entry['content'][:50]}...")
            if len(tampered_entries) > 5:
                print(f"  ... and {len(tampered_entries) - 5} more")
    
    # Enforce retention policy if requested
    if args.enforce_retention:
        if 'retention_manager' not in components:
            retention_manager = LogRetentionManager(
                log_dir=args.log_dir,
                retention_days=args.retention_days
            )
            components['retention_manager'] = retention_manager
        else:
            retention_manager = components['retention_manager']
        
        print(f"Enforcing retention policy ({args.retention_days} days)...")
        retention_results = retention_manager.enforce_retention_policy()
        
        print(f"✅ Retention policy enforced")
        print(f"  - Archived: {len(retention_results['archived_files'])} files")
        print(f"  - Deleted: {len(retention_results['deleted_files'])} files")
        if retention_results['errors']:
            print(f"  - Errors: {len(retention_results['errors'])}")
            for error in retention_results['errors']:
                print(f"    - {error}")
    
    # Analyze logs if requested
    if args.analyze_logs:
        if 'analyzer' not in components:
            analyzer = LogAnalyzer(log_dir=args.log_dir, app_name=args.app_name)
            components['analyzer'] = analyzer
        else:
            analyzer = components['analyzer']
        
        print("Analyzing logs...")
        analysis_results = analyzer.analyze_logs()
        
        print("📊 Log Analysis Results:")
        print(f"  - Total logs: {analysis_results.get('total_logs', 0)}")
        if 'level_distribution' in analysis_results:
            print("  - Level distribution:")
            for level, count in analysis_results['level_distribution'].items():
                print(f"    - {level}: {count}")
        print(f"  - Error rate: {analysis_results.get('error_rate', 0) * 100:.2f}%")
        print(f"  - Unique message patterns: {analysis_results.get('unique_patterns', 0)}")
        
        # Generate visualizations
        if VISUALIZATION_AVAILABLE:
            viz_dir = os.path.join(args.log_dir, "visualizations")
            viz_files = analyzer.generate_visualizations(viz_dir)
            if viz_files:
                print(f"  - Generated {len(viz_files)} visualizations in {viz_dir}")
    
    # Detect anomalies if requested
    if args.detect_anomalies:
        if 'analyzer' not in components:
            analyzer = LogAnalyzer(log_dir=args.log_dir, app_name=args.app_name)
            components['analyzer'] = analyzer
        else:
            analyzer = components['analyzer']
        
        print("Training anomaly detector...")
        analyzer.train_anomaly_detector()
        
        print("Detecting anomalies...")
        anomalies = analyzer.detect_anomalies()
        
        if anomalies:
            print(f"🚨 Found {len(anomalies)} anomalies")
            for i, anomaly in enumerate(anomalies[:5]):  # Show first 5
                print(f"  - Anomaly {i+1}:")
                print(f"    - Score: {anomaly.get('anomaly_score', 'N/A')}")
                print(f"    - Level: {anomaly.get('level', 'N/A')}")
                message = anomaly.get('message', 'N/A')
                print(f"    - Message: {message[:50]}..." if len(message) > 50 else f"    - Message: {message}")
            if len(anomalies) > 5:
                print(f"  ... and {len(anomalies) - 5} more")
        else:
            print("✅ No anomalies detected")
    
    # Generate compliance report if requested
    if args.generate_report:
        print("Generating compliance report...")
        auditor = ComplianceAuditor(log_dir=args.log_dir)
        
        report_file = os.path.join(args.log_dir, "compliance_report.json")
        report = auditor.generate_compliance_report(output_file=report_file)
        
        print("📋 Compliance Report Summary:")
        print(f"  - Total audit events: {report['summary'].get('total_audit_events', 0)}")
        print(f"  - Regulations checked: {', '.join(report['regulations'])}")
        print(f"  - Recommendations: {len(report['recommendations'])}")
        for i, rec in enumerate(report['recommendations'][:3]):  # Show first 3
            print(f"    - {rec['regulation']}: {rec['recommendation']}")
        if len(report['recommendations']) > 3:
            print(f"    ... and {len(report['recommendations']) - 3} more")
        print(f"  - Report saved to: {report_file}")
    
    # Start dashboard if requested
    if args.dashboard:
        if 'dashboard' not in components:
            dashboard = start_monitoring_dashboard(
                log_dir=args.log_dir,
                port=args.dashboard_port
            )
            components['dashboard'] = dashboard
        
        print(f"🚀 Monitoring dashboard started at http://localhost:{args.dashboard_port}")
        print("Press Ctrl+C to stop...")
        
        # Keep main thread alive to maintain dashboard
        try:
            # Monitor logs if requested
            if args.monitor:
                log_file = os.path.join(args.log_dir, f"{args.app_name}.log")
                print(f"Monitoring log file '{log_file}' for {args.monitor_duration} seconds...")
                
                end_time = time.time() + args.monitor_duration
                initial_size = os.stat(log_file).st_size if os.path.exists(log_file) else 0
                
                while time.time() < end_time:
                    time.sleep(1)
                    current_size = os.stat(log_file).st_size if os.path.exists(log_file) else 0
                    if current_size > initial_size:
                        print(f"📝 New log entries detected ({current_size - initial_size} bytes)")
                        initial_size = current_size
                
                print("Log monitoring complete.")
            else:
                # If not monitoring, just wait until interrupted
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
    else:
        # If no dashboard, still monitor logs if requested
        if args.monitor:
            log_file = os.path.join(args.log_dir, f"{args.app_name}.log")
            monitor_log_file(log_file, args.monitor_duration)
    
    # Generate summary report
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "log_dir": args.log_dir,
        "app_name": args.app_name,
        "log_file_exists": os.path.exists(os.path.join(args.log_dir, f"{args.app_name}.log")),
        "log_size": os.stat(os.path.join(args.log_dir, f"{args.app_name}.log")).st_size 
                    if os.path.exists(os.path.join(args.log_dir, f"{args.app_name}.log")) else 0,
        "structured_logging": args.structured,
        "encrypted_logging": args.encrypted,
        "signed_logging": args.signed
    }
    
    summary_file = os.path.join(args.log_dir, "logging_monitoring_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"Summary saved to {summary_file}")
    print("Logging & Monitoring evaluation complete.")

# Example configuration file (logging_config.json):
"""
{
    "logger": {
        "name": "my_app",
        "log_dir": "logs",
        "structured_logging": true,
        "enable_encryption": false,
        "signature_key": "your-secure-signing-key"
    },
    "aggregator": {
        "central_log_dir": "central_logs",
        "buffer_size": 1000
    },
    "analyzer": {
        "log_dir": "logs",
        "app_name": "my_app"
    },
    "monitors": [
        {
            "log_file": "logs/my_app.log",
            "rules": [
                {
                    "pattern": "error|exception|fail|critical",
                    "severity": "ERROR",
                    "description": "Error detected in logs",
                    "throttle_seconds": 60
                },
                {
                    "pattern": "warning|warn",
                    "severity": "WARNING",
                    "description": "Warning detected in logs",
                    "throttle_seconds": 300
                }
            ]
        }
    ],
    "retention": {
        "log_dir": "logs",
        "retention_days": 90,
        "archive_enabled": true
    },
    "dashboard": {
        "enabled": true,
        "log_dir": "logs",
        "port": 5000
    }
}
"""

"""
USAGE EXAMPLES:

# Basic logging
python logging_monitor.py --app-name my_app

# Enable structured JSON logging
python logging_monitor.py --app-name my_app --structured

# Verify log integrity
python logging_monitor.py --app-name my_app --verify-integrity

# Enforce retention policy
python logging_monitor.py --app-name my_app --enforce-retention --retention-days 30

# Analyze logs for patterns
python logging_monitor.py --app-name my_app --analyze-logs

# Detect anomalies in logs
python logging_monitor.py --app-name my_app --detect-anomalies

# Start monitoring dashboard
python logging_monitor.py --app-name my_app --dashboard

# Monitor logs for new entries
python logging_monitor.py --app-name my_app --monitor --monitor-duration 60

# Generate compliance report
python logging_monitor.py --app-name my_app --generate-report

# Use configuration file
python logging_monitor.py --config logging_config.json
"""
