"""
A comprehensive framework for evaluating system integration compatibility, reliability,
and performance across multiple integration points and environments.

Key capabilities:
- Dependency validation and version compatibility checking
- API endpoint testing with multi-version support
- Database connection stability and performance testing
- Data format consistency verification with schema validation
- Mock service integration for isolated testing
- Environment configuration validation across dev/staging/prod
- Comprehensive logging and reporting for audit compliance
- CI/CD pipeline integration with automated testing
"""

import logging
import json
import os
import sys
import time
import re
import pkgutil
import socket
import ssl
import datetime
import platform
import traceback
import subprocess
import importlib
import importlib.util
import argparse
import concurrent.futures
import uuid
from pathlib import Path
from functools import wraps
from typing import Dict, List, Any, Union, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

# Try importing optional dependencies
try:
    import requests
    from requests.exceptions import RequestException
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

try:
    from unittest import mock
    MOCK_AVAILABLE = True
except ImportError:
    MOCK_AVAILABLE = False

# Database connectors - try to import common ones
DB_CONNECTORS = {}

try:
    import sqlite3
    DB_CONNECTORS["sqlite"] = True
except ImportError:
    DB_CONNECTORS["sqlite"] = False

try:
    import pymysql
    DB_CONNECTORS["mysql"] = True
except ImportError:
    DB_CONNECTORS["mysql"] = False

try:
    import psycopg2
    DB_CONNECTORS["postgresql"] = True
except ImportError:
    DB_CONNECTORS["postgresql"] = False

try:
    import pymongo
    DB_CONNECTORS["mongodb"] = True
except ImportError:
    DB_CONNECTORS["mongodb"] = False

try:
    import redis
    DB_CONNECTORS["redis"] = True
except ImportError:
    DB_CONNECTORS["redis"] = False

# Constants
DEFAULT_TIMEOUT = 10
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_CONFIG_FILE = 'integration_test_config.yaml'

# Configure structured logging for auditability
def setup_logger(log_file='integration_feasibility.log', log_level=logging.INFO, 
                log_format=DEFAULT_LOG_FORMAT, console_output=True):
    """
    Set up a logger with both file and console output.
    
    Args:
        log_file: Path to log file
        log_level: Logging level
        log_format: Format string for log messages
        console_output: Whether to output logs to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('integration_testing')
    logger.setLevel(log_level)
    logger.handlers = []  # Clear any existing handlers
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

# Utility functions
def timed(func):
    """Decorator to measure and log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Function {func.__name__} completed in {elapsed:.3f} seconds")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {elapsed:.3f} seconds: {str(e)}")
            raise
    return wrapper

def get_system_info():
    """Gather system information for diagnostic purposes."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "timestamp": datetime.datetime.now().isoformat(),
        "cpu_count": os.cpu_count(),
        "available_memory": None,  # Will be filled if psutil is available
    }
    
    try:
        import psutil
        info["available_memory"] = f"{psutil.virtual_memory().available / (1024 * 1024):.2f} MB"
    except ImportError:
        pass
        
    return info

def is_port_open(host, port, timeout=DEFAULT_TIMEOUT):
    """Check if a TCP port is open on a host."""
    try:
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.close()
        return True
    except (socket.timeout, socket.error, ConnectionRefusedError):
        return False

def parse_requirements_file(file_path):
    """
    Parse a requirements.txt file into a dictionary of package names and version specs.
    
    Args:
        file_path: Path to requirements.txt file
        
    Returns:
        Dictionary mapping package names to version specifications
    """
    requirements = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Handle direct references (git+, http:// etc.)
                if any(line.startswith(prefix) for prefix in ['git+', 'hg+', 'svn+', 'http://', 'https://']):
                    parts = line.split('#egg=')
                    if len(parts) > 1:
                        package = parts[1].split('&')[0]
                        requirements[package] = line
                    continue
                
                # Regular package specifications
                parts = re.split(r'[=><~!]', line, 1)
                package = parts[0].strip()
                version_spec = line[len(package):].strip() if len(parts) > 1 else None
                requirements[package] = version_spec
                
    except FileNotFoundError:
        logger.error(f"Requirements file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error parsing requirements file: {str(e)}")
    
    return requirements

def validate_json_schema(data, schema):
    """
    Validate JSON data against a schema.
    
    Args:
        data: The JSON data to validate
        schema: JSON schema to validate against
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not JSONSCHEMA_AVAILABLE:
        logger.warning("jsonschema package not available, schema validation skipped")
        return True, "Schema validation skipped (jsonschema not installed)"
    
    try:
        jsonschema.validate(data, schema)
        return True, None
    except jsonschema.exceptions.ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Schema validation error: {str(e)}"

def load_config(config_file=DEFAULT_CONFIG_FILE):
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config = {}
    
    if not os.path.exists(config_file):
        logger.warning(f"Configuration file not found: {config_file}")
        return config
    
    try:
        file_ext = os.path.splitext(config_file)[1].lower()
        
        if file_ext in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                logger.error("YAML configuration file specified but yaml package not installed")
                return config
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        
        elif file_ext == '.json':
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        else:
            logger.error(f"Unsupported configuration file format: {file_ext}")
            
    except Exception as e:
        logger.error(f"Error loading configuration file: {str(e)}")
    
    return config

# Data classes for test definitions and results
class TestSeverity(Enum):
    """Enumeration for test severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

class TestStatus(Enum):
    """Enumeration for test result status."""
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"

@dataclass
class IntegrationTest:
    """Base class for integration tests."""
    name: str
    description: str = ""
    severity: TestSeverity = TestSeverity.MEDIUM
    enabled: bool = True
    timeout: int = DEFAULT_TIMEOUT
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)

@dataclass
class DependencyTest(IntegrationTest):
    """Test for checking package dependencies."""
    package_name: str
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    exact_version: Optional[str] = None

@dataclass
class ApiTest(IntegrationTest):
    """Test for API endpoint validation."""
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)
    data: Any = None
    expected_status: int = 200
    expected_content_type: Optional[str] = None
    expected_schema: Optional[Dict] = None
    version_header: Optional[str] = None
    auth: Optional[Dict] = None

@dataclass
class DatabaseTest(IntegrationTest):
    """Test for database connection and query validation."""
    db_type: str
    connection_string: str
    query: Optional[str] = None
    expected_result: Any = None
    connection_count: int = 1
    connection_timeout: int = 5

@dataclass
class DataFormatTest(IntegrationTest):
    """Test for data format compatibility."""
    data_sample: Dict
    expected_format: Dict
    strict_types: bool = True
    allow_extra_fields: bool = False
    required_fields: List[str] = field(default_factory=list)

@dataclass
class MockServiceTest(IntegrationTest):
    """Test with mocked external service."""
    service_to_mock: str
    mock_response: Any
    function_to_test: Callable
    function_args: List = field(default_factory=list)
    function_kwargs: Dict = field(default_factory=dict)
    expected_result: Any = None

@dataclass
class EnvironmentTest(IntegrationTest):
    """Test for environment configuration."""
    env_var: str
    expected_value: Optional[str] = None
    required: bool = True

@dataclass
class NetworkTest(IntegrationTest):
    """Test for network connectivity."""
    host: str
    port: int
    protocol: str = "tcp"
    expected_latency: Optional[float] = None

@dataclass
class TestResult:
    """Results of a test execution."""
    test: IntegrationTest
    status: TestStatus
    message: str = ""
    duration: float = 0.0
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Exception] = None
    exception_traceback: Optional[str] = None

# Core testing functions
@timed
def check_dependency_availability(package_name, min_version=None, max_version=None, exact_version=None):
    """
    Check if a required module is available with version constraints.
    
    Args:
        package_name: Name of the module to check
        min_version: Minimum acceptable version
        max_version: Maximum acceptable version
        exact_version: Exact version required
        
    Returns:
        Tuple of (is_available, version, message)
    """
    # First check if the module is installed
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False, None, f"Package {package_name} is not installed"
    
    # Try to get the version
    version = None
    try:
        module = importlib.import_module(package_name)
        
        # Check common version attributes
        for attr in ['__version__', 'VERSION', 'version']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if isinstance(version, (list, tuple)):
                    version = '.'.join(map(str, version))
                break
        
        # If version not found in attributes, try pkg_resources
        if version is None:
            try:
                import pkg_resources
                version = pkg_resources.get_distribution(package_name).version
            except Exception:
                pass
    except Exception as e:
        return True, None, f"Package {package_name} is installed but error getting version: {str(e)}"
    
    # If we still don't have a version, it's available but we can't check constraints
    if version is None:
        if any([min_version, max_version, exact_version]):
            return True, None, f"Package {package_name} is installed but version could not be determined"
        return True, None, f"Package {package_name} is installed"
    
    # Check version constraints
    if exact_version and version != exact_version:
        return False, version, f"Package {package_name} version {version} does not match required version {exact_version}"
    
    if min_version:
        # Simple version comparison - this is not perfect but works for many cases
        min_parts = list(map(int, re.findall(r'\d+', min_version)))
        ver_parts = list(map(int, re.findall(r'\d+', version)))
        
        for i in range(min(len(min_parts), len(ver_parts))):
            if ver_parts[i] < min_parts[i]:
                return False, version, f"Package {package_name} version {version} is less than minimum version {min_version}"
            elif ver_parts[i] > min_parts[i]:
                break
    
    if max_version:
        # Simple version comparison
        max_parts = list(map(int, re.findall(r'\d+', max_version)))
        ver_parts = list(map(int, re.findall(r'\d+', version)))
        
        for i in range(min(len(max_parts), len(ver_parts))):
            if ver_parts[i] > max_parts[i]:
                return False, version, f"Package {package_name} version {version} is greater than maximum version {max_version}"
            elif ver_parts[i] < max_parts[i]:
                break
    
    return True, version, f"Package {package_name} version {version} meets requirements"

@timed
def check_all_dependencies(requirements_file=None, packages=None):
    """
    Check all dependencies from a requirements file or list.
    
    Args:
        requirements_file: Path to requirements.txt file
        packages: Dictionary of package specs or list of package names
        
    Returns:
        Dictionary mapping package names to check results
    """
    results = {}
    
    # Parse requirements file if provided
    if requirements_file:
        requirements = parse_requirements_file(requirements_file)
    elif packages:
        # Handle direct package specifications
        if isinstance(packages, list):
            requirements = {pkg: None for pkg in packages}
        elif isinstance(packages, dict):
            requirements = packages
        else:
            logger.error("Invalid package specification format")
            return results
    else:
        logger.error("No requirements file or packages specified")
        return results
    
    # Check each package
    for package, version_spec in requirements.items():
        # Parse version specification
        min_version = None
        max_version = None
        exact_version = None
        
        if version_spec:
            if '==' in version_spec:
                exact_version = version_spec.split('==')[1].strip()
            elif '>=' in version_spec:
                min_version = version_spec.split('>=')[1].strip()
            elif '>' in version_spec:
                # Convert > to >= by adding a small version increment
                version_parts = version_spec.split('>')[1].strip().split('.')
                version_parts[-1] = str(int(version_parts[-1]) + 1)
                min_version = '.'.join(version_parts)
            elif '<=' in version_spec:
                max_version = version_spec.split('<=')[1].strip()
            elif '<' in version_spec:
                # Convert < to <= by subtracting a small version increment
                version_parts = version_spec.split('<')[1].strip().split('.')
                version_parts[-1] = str(int(version_parts[-1]) - 1)
                max_version = '.'.join(version_parts)
        
        # Check the package
        is_available, version, message = check_dependency_availability(
            package, min_version, max_version, exact_version
        )
        
        results[package] = {
            "available": is_available,
            "version": version,
            "message": message,
            "version_spec": version_spec
        }
    
    return results

@timed
def validate_api_response(url, method="GET", headers=None, params=None, data=None, 
                         expected_status=200, expected_content_type=None, 
                         expected_schema=None, auth=None, timeout=DEFAULT_TIMEOUT):
    """
    Check if an API endpoint returns the expected response.
    
    Args:
        url: API endpoint URL
        method: HTTP method (GET, POST, etc.)
        headers: HTTP headers to include
        params: Query parameters
        data: Request body data
        expected_status: Expected HTTP status code
        expected_content_type: Expected content-type header
        expected_schema: JSON schema to validate against
        auth: Authentication tuple or object
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (is_valid, response, message)
    """
    if not REQUESTS_AVAILABLE:
        return False, None, "Requests package not available"
    
    method = method.upper()
    headers = headers or {}
    params = params or {}
    
    # Add a user agent to avoid being blocked by some servers
    if 'User-Agent' not in headers:
        headers['User-Agent'] = 'IntegrationTester/1.0'
    
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=data if method in ["POST", "PUT", "PATCH"] and isinstance(data, (dict, list)) else None,
            data=data if method in ["POST", "PUT", "PATCH"] and not isinstance(data, (dict, list)) else None,
            auth=auth,
            timeout=timeout
        )
        
        # Check status code
        if response.status_code != expected_status:
            return False, response, f"Unexpected status code: {response.status_code}, expected: {expected_status}"
        
        # Check content type if specified
        if expected_content_type and expected_content_type not in response.headers.get('Content-Type', ''):
            return False, response, f"Unexpected content type: {response.headers.get('Content-Type')}, expected: {expected_content_type}"
        
        # Validate against schema if provided
        if expected_schema and 'application/json' in response.headers.get('Content-Type', ''):
            try:
                json_data = response.json()
                is_valid, error = validate_json_schema(json_data, expected_schema)
                if not is_valid:
                    return False, response, f"Response failed schema validation: {error}"
            except ValueError:
                return False, response, "Response contains invalid JSON"
        
        return True, response, f"API response valid (status: {response.status_code})"
        
    except RequestException as e:
        return False, None, f"API request failed: {str(e)}"
    except Exception as e:
        return False, None, f"Error validating API response: {str(e)}"

@timed
def check_api_versions(base_url, version_paths, headers=None, params=None, timeout=DEFAULT_TIMEOUT):
    """
    Check multiple versions of an API endpoint.
    
    Args:
        base_url: Base URL for the API
        version_paths: Dictionary mapping version names to endpoint paths
        headers: HTTP headers to include
        params: Query parameters
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary mapping version names to validation results
    """
    results = {}
    
    for version, path in version_paths.items():
        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        is_valid, response, message = validate_api_response(
            url=url,
            headers=headers,
            params=params,
            timeout=timeout
        )
        
        results[version] = {
            "valid": is_valid,
            "status_code": response.status_code if response else None,
            "message": message
        }
    
    return results

@timed
def check_data_format_consistency(data_sample, expected_format, strict_types=True, 
                                allow_extra_fields=False, required_fields=None):
    """
    Check if a sample dataset matches the expected format (keys and types).
    
    Args:
        data_sample: Example dictionary representing the dataset
        expected_format: Expected dictionary with field names and types
        strict_types: Whether to require exact type matches
        allow_extra_fields: Whether to allow fields not in expected_format
        required_fields: List of field names that must be present
        
    Returns:
        Tuple of (is_consistent, missing_fields, type_mismatches, extra_fields)
    """
    if not isinstance(data_sample, dict) or not isinstance(expected_format, dict):
        return False, ["Data or format is not a dictionary"], [], []
    
    # Check required fields
    required_fields = required_fields or list(expected_format.keys())
    missing_fields = [field for field in required_fields if field not in data_sample]
    
    # Check for fields in data_sample not in expected_format
    extra_fields = []
    if not allow_extra_fields:
        extra_fields = [field for field in data_sample if field not in expected_format]
    
    # Check type consistency
    type_mismatches = []
    for field, expected_type in expected_format.items():
        if field in data_sample:
            actual_value = data_sample[field]
            
            # Handle None values based on configuration
            if actual_value is None:
                if strict_types and expected_type is not type(None):
                    type_mismatches.append((field, expected_type, type(None)))
                continue
            
            if strict_types:
                # Strict type checking
                if not isinstance(actual_value, expected_type):
                    type_mismatches.append((field, expected_type, type(actual_value)))
            else:
                # Looser type checking (e.g., int can be considered float)
                if expected_type is float and isinstance(actual_value, int):
                    continue  # int can be considered float
                elif expected_type is str and isinstance(actual_value, (int, float, bool)):
                    continue  # Numbers and booleans can be converted to strings
                elif not isinstance(actual_value, expected_type):
                    type_mismatches.append((field, expected_type, type(actual_value)))
    
    # Format is consistent if there are no missing fields, type mismatches, or extra fields
    is_consistent = not (missing_fields or type_mismatches or extra_fields)
    
    return is_consistent, missing_fields, type_mismatches, extra_fields

@timed
def validate_database_connection(db_type, connection_string, query=None, expected_result=None, 
                                connection_count=1, connection_timeout=5):
    """
    Test database connection and optionally execute a query.
    
    Args:
        db_type: Type of database (sqlite, mysql, postgresql, mongodb, redis)
        connection_string: Connection string or parameters
        query: Optional query to execute
        expected_result: Expected result from query
        connection_count: Number of connections to establish
        connection_timeout: Connection timeout in seconds
        
    Returns:
        Tuple of (is_valid, results, message)
    """
    if db_type.lower() not in DB_CONNECTORS:
        return False, None, f"Database type {db_type} not supported"
    
    if not DB_CONNECTORS[db_type.lower()]:
        return False, None, f"Database connector for {db_type} not installed"
    
    try:
        # Handle different database types
        if db_type.lower() == 'sqlite':
            import sqlite3
            
            # Test connections
            connections = []
            for _ in range(connection_count):
                conn = sqlite3.connect(connection_string, timeout=connection_timeout)
                connections.append(conn)
            
            # Execute query if provided
            if query:
                results = []
                for conn in connections:
                    cursor = conn.cursor()
                    cursor.execute(query)
                    result = cursor.fetchall()
                    results.append(result)
                
                # Close connections
                for conn in connections:
                    conn.close()
                
                # Check expected result if provided
                if expected_result is not None:
                    if results[0] != expected_result:
                        return False, results, f"Query result does not match expected result: {results[0]} != {expected_result}"
                
                return True, results, "Database connection and query successful"
            
            # Close connections if no query
            for conn in connections:
                conn.close()
            
            return True, None, f"Successfully established {connection_count} SQLite connections"
            
        elif db_type.lower() == 'mysql':
            import pymysql
            
            # Parse connection string or use as is if it's a dictionary
            if isinstance(connection_string, str):
                # Very basic parsing - for production, use proper URL parsing
                params = {}
                for part in connection_string.split():
                    if '=' in part:
                        key, value = part.split('=', 1)
                        params[key.strip()] = value.strip()
            else:
                params = connection_string
            
            # Test connections
            connections = []
            for _ in range(connection_count):
                conn = pymysql.connect(**params, connect_timeout=connection_timeout)
                connections.append(conn)
            
            # Execute query if provided
            if query:
                results = []
                for conn in connections:
                    with conn.cursor() as cursor:
                        cursor.execute(query)
                        result = cursor.fetchall()
                        results.append(result)
                
                # Close connections
                for conn in connections:
                    conn.close()
                
                # Check expected result if provided
                if expected_result is not None:
                    if results[0] != expected_result:
                        return False, results, f"Query result does not match expected result"
                
                return True, results, "Database connection and query successful"
            
            # Close connections if no query
            for conn in connections:
                conn.close()
            
            return True, None, f"Successfully established {connection_count} MySQL connections"
            
        elif db_type.lower() == 'postgresql':
            import psycopg2
            
            # Test connections
            connections = []
            for _ in range(connection_count):
                conn = psycopg2.connect(connection_string, connect_timeout=connection_timeout)
                connections.append(conn)
            
            # Execute query if provided
            if query:
                results = []
                for conn in connections:
                    with conn.cursor() as cursor:
                        cursor.execute(query)
                        result = cursor.fetchall()
                        results.append(result)
                
                # Close connections
                for conn in connections:
                    conn.close()
                
                # Check expected result if provided
                if expected_result is not None:
                    if results[0] != expected_result:
                        return False, results, f"Query result does not match expected result"
                
                return True, results, "Database connection and query successful"
            
            # Close connections if no query
            for conn in connections:
                conn.close()
            
            return True, None, f"Successfully established {connection_count} PostgreSQL connections"
            
        elif db_type.lower() == 'mongodb':
            import pymongo
            
            # Create client
            client = pymongo.MongoClient(connection_string, serverSelectionTimeoutMS=connection_timeout*1000)
            
            # Force connection to verify it works
            client.server_info()
            
            # Execute query if provided
            if query and isinstance(query, dict):
                # Simple query support - just find documents in a collection
                if 'collection' in query and 'database' in query:
                    db = client[query['database']]
                    collection = db[query['collection']]
                    
                    # Execute find or other operations
                    if 'find' in query:
                        results = list(collection.find(query['find']))
                        
                        # Check expected result if provided
                        if expected_result is not None:
                            if results != expected_result:
                                client.close()
                                return False, results, f"Query result does not match expected result"
                        
                        client.close()
                        return True, results, "MongoDB connection and query successful"
                
                client.close()
                return True, None, "MongoDB connection successful, but query not executed (invalid format)"
            
            client.close()
            return True, None, "MongoDB connection successful"
            
        elif db_type.lower() == 'redis':
            import redis
            
            # Create Redis client
            r = redis.Redis.from_url(connection_string, socket_timeout=connection_timeout)
            
            # Ping to verify connection
            if r.ping():
                # Execute command if provided
                if query:
                    # Simple command support
                    parts = query.split()
                    command = parts[0].lower()
                    args = parts[1:]
                    
                    if hasattr(r, command):
                        result = getattr(r, command)(*args)
                        
                        # Check expected result if provided
                        if expected_result is not None:
                            if result != expected_result:
                                return False, result, f"Command result does not match expected result"
                        
                        return True, result, "Redis connection and command successful"
                
                return True, None, "Redis connection successful"
            else:
                return False, None, "Redis ping failed"
        
        # Add other database types as needed
            
        return False, None, f"Unsupported database type: {db_type}"
        
    except Exception as e:
        return False, None, f"Database connection error: {str(e)}"

@timed
def run_mock_service_test(service_to_mock, mock_response, function_to_test, 
                         function_args=None, function_kwargs=None, expected_result=None):
    """
    Test a function with a mocked external service.
    
    Args:
        service_to_mock: Module or function to mock
        mock_response: Response to inject from the mock
        function_to_test: Function to test
        function_args: Arguments to pass to the function
        function_kwargs: Keyword arguments to pass to the function
        expected_result: Expected result from the function
        
    Returns:
        Tuple of (is_valid, result, message)
    """
    if not MOCK_AVAILABLE:
        return False, None, "Mocking not available (unittest.mock not importable)"
    
    function_args = function_args or []
    function_kwargs = function_kwargs or {}
    
    try:
        # Create the mock
        with mock.patch(service_to_mock) as mock_service:
            # Configure the mock
            if callable(mock_response):
                # If mock_response is a function, use it as a side_effect
                mock_service.side_effect = mock_response
            else:
                # Otherwise, use it as a return_value
                mock_service.return_value = mock_response
            
            # Call the function
            result = function_to_test(*function_args, **function_kwargs)
            
            # Check the result if expected_result is provided
            if expected_result is not None:
                if result != expected_result:
                    return False, result, f"Function result {result} does not match expected result {expected_result}"
            
            return True, result, "Function executed successfully with mock service"
    
    except Exception as e:
        return False, None, f"Error running mock service test: {str(e)}"

@timed
def check_environment_configuration(env_var, expected_value=None, required=True):
    """
    Check if an environment variable is properly configured.
    
    Args:
        env_var: Name of the environment variable
        expected_value: Expected value (if None, just checks existence)
        required: Whether the variable is required
        
    Returns:
        Tuple of (is_valid, actual_value, message)
    """
    value = os.environ.get(env_var)
    
    # Check if the variable exists
    if value is None:
        if required:
            return False, None, f"Required environment variable {env_var} is not set"
        else:
            return True, None, f"Optional environment variable {env_var} is not set"
    
    # Check the value if expected_value is provided
    if expected_value is not None and value != expected_value:
        return False, value, f"Environment variable {env_var} has value '{value}', expected '{expected_value}'"
    
    return True, value, f"Environment variable {env_var} is properly configured"

@timed
def check_network_connectivity(host, port, protocol="tcp", expected_latency=None, 
                             timeout=DEFAULT_TIMEOUT):
    """
    Check network connectivity to a host and port.
    
    Args:
        host: Hostname or IP address
        port: Port number
        protocol: Protocol to use (tcp or udp)
        expected_latency: Maximum expected latency in seconds
        timeout: Connection timeout in seconds
        
    Returns:
        Tuple of (is_connected, latency, message)
    """
    try:
        start_time = time.time()
        
        if protocol.lower() == "tcp":
            is_open = is_port_open(host, port, timeout)
        elif protocol.lower() == "udp":
            # UDP is connectionless, so we can't easily check if a port is "open"
            # For now, just try to create a socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            is_open = True
        else:
            return False, None, f"Unsupported protocol: {protocol}"
        
        latency = time.time() - start_time
        
        if not is_open:
            return False, latency, f"Could not connect to {host}:{port} using {protocol}"
        
        # Check latency if expected_latency is provided
        if expected_latency is not None and latency > expected_latency:
            return False, latency, f"Connection latency ({latency:.3f}s) exceeds expected latency ({expected_latency:.3f}s)"
        
        return True, latency, f"Successfully connected to {host}:{port} using {protocol} (latency: {latency:.3f}s)"
        
    except Exception as e:
        return False, None, f"Error checking network connectivity: {str(e)}"

def run_integration_test(test):
    """
    Run a single integration test based on its type.
    
    Args:
        test: IntegrationTest object
        
    Returns:
        TestResult object
    """
    if not test.enabled:
        return TestResult(
            test=test,
            status=TestStatus.SKIPPED,
            message="Test disabled"
        )
    
    start_time = time.time()
    
    try:
        # Run the appropriate test based on the test type
        if isinstance(test, DependencyTest):
            is_available, version, message = check_dependency_availability(
                test.package_name, test.min_version, test.max_version, test.exact_version
            )
            
            return TestResult(
                test=test,
                status=TestStatus.PASS if is_available else TestStatus.FAIL,
                message=message,
                duration=time.time() - start_time,
                details={"version": version}
            )
            
        elif isinstance(test, ApiTest):
            is_valid, response, message = validate_api_response(
                url=test.url,
                method=test.method,
                headers=test.headers,
                params=test.params,
                data=test.data,
                expected_status=test.expected_status,
                expected_content_type=test.expected_content_type,
                expected_schema=test.expected_schema,
                auth=test.auth,
                timeout=test.timeout
            )
            
            details = {}
            if response:
                details["status_code"] = response.status_code
                details["headers"] = dict(response.headers)
                try:
                    if 'application/json' in response.headers.get('Content-Type', ''):
                        details["content"] = response.json()
                    else:
                        # Include a truncated version of the content
                        content = response.text
                        if len(content) > 1000:
                            content = content[:1000] + "... [truncated]"
                        details["content"] = content
                except:
                    details["content"] = "[Error parsing content]"
            
            return TestResult(
                test=test,
                status=TestStatus.PASS if is_valid else TestStatus.FAIL,
                message=message,
                duration=time.time() - start_time,
                details=details
            )
            
        elif isinstance(test, DatabaseTest):
            is_valid, results, message = validate_database_connection(
                db_type=test.db_type,
                connection_string=test.connection_string,
                query=test.query,
                expected_result=test.expected_result,
                connection_count=test.connection_count,
                connection_timeout=test.connection_timeout
            )
            
            return TestResult(
                test=test,
                status=TestStatus.PASS if is_valid else TestStatus.FAIL,
                message=message,
                duration=time.time() - start_time,
                details={"results": results}
            )
            
        elif isinstance(test, DataFormatTest):
            is_consistent, missing_fields, type_mismatches, extra_fields = check_data_format_consistency(
                data_sample=test.data_sample,
                expected_format=test.expected_format,
                strict_types=test.strict_types,
                allow_extra_fields=test.allow_extra_fields,
                required_fields=test.required_fields
            )
            
            details = {
                "missing_fields": missing_fields,
                "type_mismatches": [
                    {
                        "field": field,
                        "expected_type": str(expected_type),
                        "actual_type": str(actual_type)
                    }
                    for field, expected_type, actual_type in type_mismatches
                ],
                "extra_fields": extra_fields
            }
            
            message = "Data format is consistent" if is_consistent else "Data format inconsistencies found"
            
            return TestResult(
                test=test,
                status=TestStatus.PASS if is_consistent else TestStatus.FAIL,
                message=message,
                duration=time.time() - start_time,
                details=details
            )
            
        elif isinstance(test, MockServiceTest):
            is_valid, result, message = run_mock_service_test(
                service_to_mock=test.service_to_mock,
                mock_response=test.mock_response,
                function_to_test=test.function_to_test,
                function_args=test.function_args,
                function_kwargs=test.function_kwargs,
                expected_result=test.expected_result
            )
            
            return TestResult(
                test=test,
                status=TestStatus.PASS if is_valid else TestStatus.FAIL,
                message=message,
                duration=time.time() - start_time,
                details={"result": result}
            )
            
        elif isinstance(test, EnvironmentTest):
            is_valid, value, message = check_environment_configuration(
                env_var=test.env_var,
                expected_value=test.expected_value,
                required=test.required
            )
            
            return TestResult(
                test=test,
                status=TestStatus.PASS if is_valid else TestStatus.FAIL,
                message=message,
                duration=time.time() - start_time,
                details={"value": value}
            )
            
        elif isinstance(test, NetworkTest):
            is_connected, latency, message = check_network_connectivity(
                host=test.host,
                port=test.port,
                protocol=test.protocol,
                expected_latency=test.expected_latency,
                timeout=test.timeout
            )
            
            return TestResult(
                test=test,
                status=TestStatus.PASS if is_connected else TestStatus.FAIL,
                message=message,
                duration=time.time() - start_time,
                details={"latency": latency}
            )
            
        else:
            return TestResult(
                test=test,
                status=TestStatus.ERROR,
                message=f"Unsupported test type: {type(test).__name__}",
                duration=time.time() - start_time
            )
    
    except Exception as e:
        return TestResult(
            test=test,
            status=TestStatus.ERROR,
            message=f"Error running test: {str(e)}",
            duration=time.time() - start_time,
            exception=e,
            exception_traceback=traceback.format_exc()
        )

def run_integration_tests(tests, parallel=False, max_workers=None):
    """
    Run a list of integration tests.
    
    Args:
        tests: List of IntegrationTest objects
        parallel: Whether to run tests in parallel
        max_workers: Maximum number of worker threads or processes
        
    Returns:
        List of TestResult objects
    """
    if not tests:
        logger.warning("No tests to run")
        return []
    
    logger.info(f"Running {len(tests)} integration tests")
    
    # Sort tests by prerequisites
    # Simple approach: tests with no prerequisites go first
    tests_with_prereqs = [t for t in tests if t.prerequisites]
    tests_without_prereqs = [t for t in tests if not t.prerequisites]
    
    # Ensure all prerequisites are met
    prereq_names = {t.name for t in tests}
    for test in tests_with_prereqs:
        missing_prereqs = [p for p in test.prerequisites if p not in prereq_names]
        if missing_prereqs:
            logger.warning(f"Test '{test.name}' has missing prerequisites: {missing_prereqs}")
    
    # Combine sorted tests
    sorted_tests = tests_without_prereqs + tests_with_prereqs
    
    # Dictionary to track test results by name
    results_by_name = {}
    
    # Run tests without prerequisites first
    if parallel and len(tests_without_prereqs) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {executor.submit(run_integration_test, test): test for test in tests_without_prereqs}
            
            for future in concurrent.futures.as_completed(future_to_test):
                test = future_to_test[future]
                try:
                    result = future.result()
                    results_by_name[test.name] = result
                except Exception as e:
                    logger.error(f"Error running test '{test.name}': {str(e)}")
                    results_by_name[test.name] = TestResult(
                        test=test,
                        status=TestStatus.ERROR,
                        message=f"Execution error: {str(e)}",
                        exception=e,
                        exception_traceback=traceback.format_exc()
                    )
    else:
        for test in tests_without_prereqs:
            result = run_integration_test(test)
            results_by_name[test.name] = result
    
    # Run tests with prerequisites, ensuring prerequisites are met
    for test in tests_with_prereqs:
        # Check if all prerequisites passed
        prereqs_passed = all(
            results_by_name.get(prereq, TestResult(test=None, status=TestStatus.ERROR)).status == TestStatus.PASS
            for prereq in test.prerequisites
        )
        
        if not prereqs_passed:
            # Skip this test since prerequisites failed
            results_by_name[test.name] = TestResult(
                test=test,
                status=TestStatus.SKIPPED,
                message="Prerequisites failed"
            )
            continue
        
        # Run the test
        result = run_integration_test(test)
        results_by_name[test.name] = result
    
    # Return results in the same order as the input tests
    return [results_by_name.get(test.name) for test in tests]

def generate_test_report(results, output_file='integration_report.json'):
    """
    Generate a report from test results.
    
    Args:
        results: List of TestResult objects
        output_file: Path to save the report
        
    Returns:
        Dictionary containing the report data
    """
    # Gather test statistics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.status == TestStatus.PASS)
    failed_tests = sum(1 for r in results if r.status == TestStatus.FAIL)
    error_tests = sum(1 for r in results if r.status == TestStatus.ERROR)
    skipped_tests = sum(1 for r in results if r.status == TestStatus.SKIPPED)
    
    # Calculate pass percentage
    pass_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Group results by test type
    results_by_type = {}
    for result in results:
        test_type = type(result.test).__name__
        if test_type not in results_by_type:
            results_by_type[test_type] = []
        results_by_type[test_type].append(result)
    
    # Create the report structure
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "error_tests": error_tests,
            "skipped_tests": skipped_tests,
            "pass_percentage": pass_percentage
        },
        "test_types": {
            test_type: {
                "total": len(results_of_type),
                "passed": sum(1 for r in results_of_type if r.status == TestStatus.PASS),
                "failed": sum(1 for r in results_of_type if r.status == TestStatus.FAIL),
                "error": sum(1 for r in results_of_type if r.status == TestStatus.ERROR),
                "skipped": sum(1 for r in results_of_type if r.status == TestStatus.SKIPPED)
            }
            for test_type, results_of_type in results_by_type.items()
        },
        "system_info": get_system_info(),
        "test_results": []
    }
    
    # Add detailed test results
    for result in results:
        # Create a serializable version of the test result
        # (some fields may not be directly serializable to JSON)
        test_result = {
            "name": result.test.name,
            "description": result.test.description,
            "type": type(result.test).__name__,
            "status": result.status.value,
            "severity": result.test.severity.value,
            "message": result.message,
            "duration": result.duration,
            "timestamp": result.timestamp.isoformat(),
            "tags": result.test.tags
        }
        
        # Add details if they exist and are serializable
        if result.details:
            try:
                # Test JSON serialization
                json.dumps(result.details)
                test_result["details"] = result.details
            except (TypeError, OverflowError):
                # If not serializable, convert to string representation
                test_result["details"] = str(result.details)
        
        # Add exception information if it exists
        if result.exception:
            test_result["exception"] = {
                "type": type(result.exception).__name__,
                "message": str(result.exception)
            }
            if result.exception_traceback:
                test_result["exception"]["traceback"] = result.exception_traceback
        
        report["test_results"].append(test_result)
    
    # Save the report to a file if an output file is specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"Test report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving test report: {str(e)}")
    
    return report

def generate_html_report(results, output_file='integration_report.html'):
    """
    Generate an HTML report from test results.
    
    Args:
        results: List of TestResult objects
        output_file: Path to save the HTML report
        
    Returns:
        Boolean indicating success
    """
    try:
        # Generate JSON report first (reusing the same code)
        json_report = generate_test_report(results, output_file=None)
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Integration Testing Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            color: #333;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        .summary {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .progress-bar {{
            width: 100%;
            background-color: #e9ecef;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .progress {{
            height: 20px;
            border-radius: 5px;
            text-align: center;
            line-height: 20px;
            color: white;
        }}
        .pass {{
            background-color: #28a745;
        }}
        .fail {{
            background-color: #dc3545;
        }}
        .error {{
            background-color: #dc3545;
        }}
        .skipped {{
            background-color: #6c757d;
        }}
        .test-details {{
            margin-bottom: 30px;
        }}
        .test {{
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }}
        .test-pass {{
            background-color: #d4edda;
        }}
        .test-fail {{
            background-color: #f8d7da;
        }}
        .test-error {{
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
        }}
        .test-skipped {{
            background-color: #e9ecef;
        }}
        .test-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }}
        .test-message {{
            margin: 5px 0;
        }}
        .test-duration {{
            font-style: italic;
            color: #6c757d;
        }}
        .tag {{
            display: inline-block;
            padding: 2px 5px;
            border-radius: 4px;
            background-color: #e9ecef;
            margin-right: 5px;
            font-size: 0.8em;
        }}
        .collapsible {{
            cursor: pointer;
        }}
        .collapsible::after {{
            content: "\\002B"; /* Plus sign */
            float: right;
            font-weight: bold;
        }}
        .active::after {{
            content: "\\2212"; /* Minus sign */
        }}
        .content {{
            display: none;
            overflow: hidden;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .severity-CRITICAL {{
            color: #dc3545;
            font-weight: bold;
        }}
        .severity-HIGH {{
            color: #fd7e14;
            font-weight: bold;
        }}
        .severity-MEDIUM {{
            color: #ffc107;
        }}
        .severity-LOW {{
            color: #20c997;
        }}
        .severity-INFO {{
            color: #17a2b8;
        }}
    </style>
</head>
<body>
    <h1>Integration Testing Report</h1>
    <p>Generated: {json_report["timestamp"]}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <div class="progress-bar">
            <div class="progress pass" style="width: {json_report["summary"]["pass_percentage"]}%;">
                {json_report["summary"]["pass_percentage"]:.1f}%
            </div>
        </div>
        <table>
            <tr>
                <th>Total Tests</th>
                <td>{json_report["summary"]["total_tests"]}</td>
            </tr>
            <tr>
                <th>Passed</th>
                <td>{json_report["summary"]["passed_tests"]}</td>
            </tr>
            <tr>
                <th>Failed</th>
                <td>{json_report["summary"]["failed_tests"]}</td>
            </tr>
            <tr>
                <th>Errors</th>
                <td>{json_report["summary"]["error_tests"]}</td>
            </tr>
            <tr>
                <th>Skipped</th>
                <td>{json_report["summary"]["skipped_tests"]}</td>
            </tr>
            <tr>
                <th>Pass Percentage</th>
                <td>{json_report["summary"]["pass_percentage"]:.1f}%</td>
            </tr>
        </table>
        
        <h3>Results by Test Type</h3>
        <table>
            <tr>
                <th>Test Type</th>
                <th>Total</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Errors</th>
                <th>Skipped</th>
            </tr>
        """
        
        # Add rows for each test type
        for test_type, stats in json_report["test_types"].items():
            html_content += f"""
            <tr>
                <td>{test_type}</td>
                <td>{stats["total"]}</td>
                <td>{stats["passed"]}</td>
                <td>{stats["failed"]}</td>
                <td>{stats["error"]}</td>
                <td>{stats["skipped"]}</td>
            </tr>
            """
        
        html_content += """
        </table>
    </div>
    
    <h2>System Information</h2>
    <table>
        """
        
        # Add system information
        for key, value in json_report["system_info"].items():
            if value is not None:
                html_content += f"<tr><th>{key}</th><td>{value}</td></tr>"
        
        html_content += """
    </table>
    
    <h2>Test Results</h2>
    
    <div class="test-details">
        """
        
        # Group test results by status for better organization
        result_groups = {
            "error": [r for r in json_report["test_results"] if r["status"] == "ERROR"],
            "fail": [r for r in json_report["test_results"] if r["status"] == "FAIL"],
            "skipped": [r for r in json_report["test_results"] if r["status"] == "SKIPPED"],
            "pass": [r for r in json_report["test_results"] if r["status"] == "PASS"]
        }
        
        # First show errors and failures
        for status in ["error", "fail", "skipped", "pass"]:
            if result_groups[status]:
                status_name = status.upper()
                html_content += f"<h3>{status_name} ({len(result_groups[status])})</h3>"
                
                for result in result_groups[status]:
                    # Calculate CSS classes for the test
                    status_class = f"test-{status.lower()}"
                    
                    html_content += f"""
                    <div class="test {status_class}">
                        <div class="test-header">
                            <span>{result["name"]} ({result["type"]}) <span class="severity-{result["severity"]}">[{result["severity"]}]</span></span>
                            <span class="test-duration">{result["duration"]:.3f}s</span>
                        </div>
                        """
                    
                    # Add description if available
                    if result["description"]:
                        html_content += f"<p>{result['description']}</p>"
                    
                    # Add message
                    html_content += f"<div class='test-message'>{result['message']}</div>"
                    
                    # Add tags if any
                    if result["tags"]:
                        tags_html = " ".join([f"<span class='tag'>{tag}</span>" for tag in result["tags"]])
                        html_content += f"<div>{tags_html}</div>"
                    
                    # Add collapsible details section if there are details or an exception
                    has_details = "details" in result or "exception" in result
                    if has_details:
                        html_content += f"""
                        <button class="collapsible">Details</button>
                        <div class="content">
                        """
                        
                        # Add details if available
                        if "details" in result:
                            details = result["details"]
                            if isinstance(details, str):
                                html_content += f"<pre>{details}</pre>"
                            else:
                                html_content += f"<pre>{json.dumps(details, indent=2)}</pre>"
                        
                        # Add exception if available
                        if "exception" in result:
                            exception = result["exception"]
                            html_content += f"<h4>Exception: {exception['type']}</h4>"
                            html_content += f"<p>{exception['message']}</p>"
                            
                            if "traceback" in exception:
                                html_content += f"<pre>{exception['traceback']}</pre>"
                        
                        html_content += "</div>"
                    
                    html_content += "</div>"
        
        # Add JavaScript for collapsible sections
        html_content += """
    </div>
    
    <script>
        // Add functionality for collapsible sections
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {
                    content.style.display = "none";
                } else {
                    content.style.display = "block";
                }
            });
        }
    </script>
</body>
</html>
        """
        
        # Write to the output file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        return False

def load_tests_from_config(config_file=DEFAULT_CONFIG_FILE):
    """
    Load test definitions from a configuration file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        List of IntegrationTest objects
    """
    tests = []
    
    # Load the configuration
    config = load_config(config_file)
    if not config:
        logger.error(f"Failed to load configuration from {config_file}")
        return tests
    
    # Parse test definitions
    test_definitions = config.get("tests", [])
    for test_def in test_definitions:
        try:
            test_type = test_def.get("type")
            if not test_type:
                logger.warning(f"Missing test type in test definition: {test_def}")
                continue
            
            # Common fields for all test types
            common_args = {
                "name": test_def.get("name", f"test_{uuid.uuid4().hex[:8]}"),
                "description": test_def.get("description", ""),
                "severity": TestSeverity(test_def.get("severity", "MEDIUM")),
                "enabled": test_def.get("enabled", True),
                "timeout": test_def.get("timeout", DEFAULT_TIMEOUT),
                "tags": test_def.get("tags", []),
                "prerequisites": test_def.get("prerequisites", [])
            }
            
            # Create the appropriate test object based on type
            if test_type.lower() == "dependency":
                tests.append(DependencyTest(
                    **common_args,
                    package_name=test_def["package_name"],
                    min_version=test_def.get("min_version"),
                    max_version=test_def.get("max_version"),
                    exact_version=test_def.get("exact_version")
                ))
                
            elif test_type.lower() == "api":
                tests.append(ApiTest(
                    **common_args,
                    url=test_def["url"],
                    method=test_def.get("method", "GET"),
                    headers=test_def.get("headers", {}),
                    params=test_def.get("params", {}),
                    data=test_def.get("data"),
                    expected_status=test_def.get("expected_status", 200),
                    expected_content_type=test_def.get("expected_content_type"),
                    expected_schema=test_def.get("expected_schema"),
                    version_header=test_def.get("version_header"),
                    auth=test_def.get("auth")
                ))
                
            elif test_type.lower() == "database":
                tests.append(DatabaseTest(
                    **common_args,
                    db_type=test_def["db_type"],
                    connection_string=test_def["connection_string"],
                    query=test_def.get("query"),
                    expected_result=test_def.get("expected_result"),
                    connection_count=test_def.get("connection_count", 1),
                    connection_timeout=test_def.get("connection_timeout", 5)
                ))
                
            elif test_type.lower() == "data_format":
                tests.append(DataFormatTest(
                    **common_args,
                    data_sample=test_def["data_sample"],
                    expected_format=test_def["expected_format"],
                    strict_types=test_def.get("strict_types", True),
                    allow_extra_fields=test_def.get("allow_extra_fields", False),
                    required_fields=test_def.get("required_fields", [])
                ))
                
            elif test_type.lower() == "environment":
                tests.append(EnvironmentTest(
                    **common_args,
                    env_var=test_def["env_var"],
                    expected_value=test_def.get("expected_value"),
                    required=test_def.get("required", True)
                ))
                
            elif test_type.lower() == "network":
                tests.append(NetworkTest(
                    **common_args,
                    host=test_def["host"],
                    port=test_def["port"],
                    protocol=test_def.get("protocol", "tcp"),
                    expected_latency=test_def.get("expected_latency")
                ))
                
            else:
                logger.warning(f"Unknown test type: {test_type}")
            
        except KeyError as e:
            logger.error(f"Missing required field in test definition: {e}")
        except Exception as e:
            logger.error(f"Error parsing test definition: {str(e)}")
    
    return tests

def run_integration_suite(config_file=DEFAULT_CONFIG_FILE, output_dir='.', parallel=False):
    """
    Run a full integration test suite from a configuration file.
    
    Args:
        config_file: Path to configuration file
        output_dir: Directory to save reports
        parallel: Whether to run tests in parallel
        
    Returns:
        Dictionary with test results and statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tests from configuration
    tests = load_tests_from_config(config_file)
    if not tests:
        logger.error("No tests loaded from configuration")
        return {
            "success": False,
            "message": "No tests loaded from configuration",
            "tests_run": 0
        }
    
    logger.info(f"Loaded {len(tests)} tests from configuration")
    
    # Run the tests
    results = run_integration_tests(tests, parallel=parallel)
    
    # Generate reports
    json_report_path = os.path.join(output_dir, "integration_report.json")
    html_report_path = os.path.join(output_dir, "integration_report.html")
    
    report_data = generate_test_report(results, output_file=json_report_path)
    generate_html_report(results, output_file=html_report_path)
    
    # Print summary to console
    total = len(results)
    passed = sum(1 for r in results if r.status == TestStatus.PASS)
    failed = sum(1 for r in results if r.status == TestStatus.FAIL)
    error = sum(1 for r in results if r.status == TestStatus.ERROR)
    skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
    
    logger.info("=" * 60)
    logger.info("INTEGRATION TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Errors: {error}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Pass Rate: {(passed / total * 100) if total > 0 else 0:.1f}%")
    logger.info("=" * 60)
    logger.info(f"Reports saved to {output_dir}")
    
    return {
        "success": True,
        "tests_run": total,
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_error": error,
        "tests_skipped": skipped,
        "pass_rate": (passed / total * 100) if total > 0 else 0,
        "report_paths": {
            "json": json_report_path,
            "html": html_report_path
        }
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Integration Feasibility Testing Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG_FILE,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="integration_results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--check-dependencies",
        action="store_true",
        help="Check all dependencies in requirements.txt"
    )
    
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
        help="Path to requirements.txt file"
    )
    
    return parser.parse_args()

# Example test cases
if __name__ == "__main__":
    args = parse_args()
    
    # Configure logger verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)
    
    # Check dependencies if requested
    if args.check_dependencies:
        if os.path.exists(args.requirements):
            logger.info(f"Checking dependencies in {args.requirements}")
            results = check_all_dependencies(args.requirements)
            
            # Print results
            print("\nDEPENDENCY CHECK RESULTS")
            print("=" * 60)
            for package, result in results.items():
                status = "AVAILABLE" if result["available"] else "MISSING"
                version_info = f" (version: {result['version']})" if result["version"] else ""
                spec_info = f" (spec: {result['version_spec']})" if result["version_spec"] else ""
                print(f"{package}: {status}{version_info}{spec_info}")
            
            # Count availability
            available = sum(1 for r in results.values() if r["available"])
            print("=" * 60)
            print(f"Available: {available}/{len(results)} packages")
            print("")
            
            sys.exit(0)
        else:
            logger.error(f"Requirements file not found: {args.requirements}")
            sys.exit(1)
    
    # Run example tests or from configuration file
    if os.path.exists(args.config):
        # Run tests from configuration
        logger.info(f"Running integration tests from {args.config}")
        result = run_integration_suite(
            config_file=args.config,
            output_dir=args.output_dir,
            parallel=args.parallel
        )
        
        if not result["success"]:
            sys.exit(1)
            
        # Exit with code based on pass rate
        if result["pass_rate"] < 100:
            sys.exit(1)
            
    else:
        # Run example tests
        logger.info("Configuration file not found, running example tests")
        
        # Example tests
        tests = [
            DependencyTest(
                name="Check Requests Package",
                description="Verify the requests package is installed",
                package_name="requests"
            ),
            
            ApiTest(
                name="JSONPlaceholder API Test",
                description="Check if JSONPlaceholder API is accessible",
                url="https://jsonplaceholder.typicode.com/posts/1",
                expected_status=200
            ),
            
            DataFormatTest(
                name="Sample Data Format Check",
                description="Verify sample data format is consistent",
                data_sample={"id": 1, "name": "Test", "active": True},
                expected_format={"id": int, "name": str, "active": bool}
            ),
            
            EnvironmentTest(
                name="Check Python Path",
                description="Verify PYTHONPATH is set",
                env_var="PYTHONPATH",
                required=False
            ),
            
            NetworkTest(
                name="Check Google Connectivity",
                description="Verify connectivity to Google",
                host="www.google.com",
                port=443
            )
        ]
        
        # Run tests
        results = run_integration_tests(tests)
        
        # Generate reports
        os.makedirs(args.output_dir, exist_ok=True)
        json_report_path = os.path.join(args.output_dir, "integration_report.json")
        html_report_path = os.path.join(args.output_dir, "integration_report.html")
        
        generate_test_report(results, output_file=json_report_path)
        generate_html_report(results, output_file=html_report_path)
        
        # Print summary
        total = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASS)
        print(f"\nIntegration Feasibility testing complete. {passed}/{total} tests passed.")
        print(f"Check '{args.output_dir}' directory for detailed reports.")

"""
TODO:
- Add support for testing authentication and authorization flows
- Implement service virtualization for mocking complex system interactions
- Add support for performance benchmarking during integration tests
- Create automated test documentation generation
- Build adapter for popular CI/CD platforms (Jenkins, GitHub Actions, etc.)
- Implement chaos engineering capabilities for resilience testing
- Add contract testing for API producer/consumer verification
- Develop plugin system for custom test types and reporters
- Add support for message queue testing (Kafka, RabbitMQ, etc.)
- Implement intelligent test ordering based on dependency graph analysis
"""
