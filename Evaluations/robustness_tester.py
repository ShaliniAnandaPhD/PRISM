"""

This advanced framework systematically evaluates function and model resilience
against a wide range of challenging inputs and edge cases.

Key capabilities:
- Comprehensive input type testing (numbers, strings, collections, objects)
- Boundary value analysis and edge case detection
- Randomized fuzz testing with smart input generation
- Performance degradation analysis under stress conditions
- Exception handling verification and classification
- Memory/resource usage monitoring
- Thorough reporting with visualizations and actionable insights
"""

import logging
import json
import time
import inspect
import random
import string
import math
import sys
import os
import gc
import traceback
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
import pandas as pd

# Configure logging with both file and console handlers
def setup_logging(log_file='robustness_testing.log', console_level=logging.INFO):
    """Set up logging to both file and console."""
    logger = logging.getLogger('robustness')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler if not already there
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Create console handler if not already there
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Constants for testing
MAX_INT = sys.maxsize
MIN_INT = -sys.maxsize - 1
MAX_FLOAT = sys.float_info.max
MIN_FLOAT = sys.float_info.min
EPSILON = sys.float_info.epsilon

# Classes for test case definition and results
@dataclass
class TestCase:
    """A single test case with input, expected output, and metadata."""
    name: str
    inputs: tuple
    expected: Any = None
    category: str = "general"
    should_raise: Optional[type] = None
    timeout: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None

@dataclass
class TestResult:
    """Result of a single test case execution."""
    test_case: TestCase
    passed: bool
    actual: Any = None
    exception: Optional[Exception] = None
    traceback: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: Optional[float] = None
    note: Optional[str] = None

# Utility functions for input generation
def generate_numeric_edge_cases():
    """Generate a set of numeric edge cases."""
    return [
        0,  # Zero
        1,  # Unit value
        -1,  # Negative unit value
        MAX_INT,  # Maximum integer
        MIN_INT,  # Minimum integer
        MAX_FLOAT,  # Maximum float
        -MAX_FLOAT,  # Negative maximum float
        MIN_FLOAT,  # Minimum positive float
        -MIN_FLOAT,  # Negative minimum float
        float('inf'),  # Infinity
        float('-inf'),  # Negative infinity
        float('nan'),  # Not a number
        math.pi,  # Pi
        math.e,  # e
        10**6,  # Large number
        10**-6,  # Small number
        0.1 + 0.2,  # Floating point precision issue
        MAX_INT + 1,  # Overflow potential
        2**32,  # Power of 2 edge
        2**64,  # Another power of 2 edge
    ]

def generate_string_edge_cases():
    """Generate a set of string edge cases."""
    return [
        "",  # Empty string
        " ",  # Space
        "a",  # Single character
        "A",  # Single uppercase character
        "0",  # Single digit character
        "abcdefghijklmnopqrstuvwxyz",  # Alphabet
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",  # Uppercase alphabet
        "0123456789",  # Digits
        "!@#$%^&*()_+-=[]{}|;':\",./<>?",  # Special characters
        "a" * 1000,  # Very long string
        "\n",  # Newline
        "\t",  # Tab
        "\r",  # Carriage return
        "\0",  # Null byte
        "\u0000",  # Unicode null
        "\u2022",  # Unicode bullet
        "你好",  # Unicode (Chinese)
        "مرحبا",  # Unicode (Arabic)
        "Hello\nWorld",  # Multiline
        "<!--XSS-->",  # Potential injection
        "<script>alert('XSS')</script>",  # Script injection
        "DROP TABLE users;",  # SQL injection
        "null",  # String that looks like null
        "undefined",  # String that looks like undefined
        "true",  # String that looks like boolean
        "123",  # String that looks like number
        "NaN",  # String that looks like NaN
    ]

def generate_collection_edge_cases():
    """Generate a set of collection edge cases."""
    return [
        [],  # Empty list
        (),  # Empty tuple
        {},  # Empty dict
        set(),  # Empty set
        [1],  # Single item list
        (1,),  # Single item tuple
        {1: 1},  # Single item dict
        {1},  # Single item set
        [1, 2, 3],  # Small list
        list(range(1000)),  # Large list
        [[]],  # Nested empty list
        [None],  # List with None
        [1, "a", None, []],  # Mixed type list
        {"key": "value"},  # Simple dict
        {"": ""},  # Empty string keys and values
        {1: 2, 3: 4},  # Numeric keys
        {None: None},  # None as key and value
        {1: {2: {3: 4}}},  # Deeply nested dict
        [[1, 2], [3, 4]],  # List of lists (matrix-like)
        {frozenset([1, 2]): "value"},  # Frozen set as key
    ]

def generate_special_edge_cases():
    """Generate a set of special edge cases."""
    return [
        None,  # None value
        object(),  # Generic object
        type,  # Type object
        True,  # Boolean true
        False,  # Boolean false
        NotImplemented,  # Not implemented
        ...,  # Ellipsis
        type('CustomClass', (), {}),  # Dynamic class
        lambda x: x,  # Lambda function
        (x for x in range(10)),  # Generator
        iter([1, 2, 3]),  # Iterator
        bytes([65, 66, 67]),  # Bytes
        bytearray([65, 66, 67]),  # Bytearray
        memoryview(bytes([65, 66, 67])),  # Memory view
        complex(1, 2),  # Complex number
        frozenset([1, 2, 3]),  # Frozen set
    ]

def generate_datetime_edge_cases():
    """Generate edge cases for date and time inputs."""
    from datetime import datetime, date, time, timedelta
    return [
        datetime.min,  # Minimum datetime
        datetime.max,  # Maximum datetime
        datetime.now(),  # Current datetime
        date.min,  # Minimum date
        date.max,  # Maximum date
        date.today(),  # Today's date
        time.min,  # Minimum time
        time.max,  # Maximum time
        timedelta.min,  # Minimum timedelta
        timedelta.max,  # Maximum timedelta
        timedelta(0),  # Zero timedelta
    ]

def generate_fuzz_inputs(param_type, n_samples=10):
    """Generate random fuzz inputs based on the parameter type."""
    if param_type is int or param_type == 'int':
        return [random.randint(MIN_INT // 1000, MAX_INT // 1000) for _ in range(n_samples)]
    elif param_type is float or param_type == 'float':
        return [random.uniform(-MAX_FLOAT / 1000, MAX_FLOAT / 1000) for _ in range(n_samples)]
    elif param_type is str or param_type == 'str':
        return [''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, 
                                     k=random.randint(0, 100))) for _ in range(n_samples)]
    elif param_type is bool or param_type == 'bool':
        return [random.choice([True, False]) for _ in range(n_samples)]
    elif param_type is list or param_type == 'list':
        return [random.choices(range(-100, 100), k=random.randint(0, 20)) for _ in range(n_samples)]
    elif param_type is dict or param_type == 'dict':
        return [{str(random.randint(1, 100)): random.randint(-100, 100) for _ in range(random.randint(0, 10))} 
                for _ in range(n_samples)]
    elif param_type is tuple or param_type == 'tuple':
        return [tuple(random.choices(range(-100, 100), k=random.randint(0, 10))) for _ in range(n_samples)]
    elif param_type is set or param_type == 'set':
        return [{random.randint(-100, 100) for _ in range(random.randint(0, 10))} for _ in range(n_samples)]
    else:
        # Default case, try to use some basic types
        mixed_types = [10, -10, 0, 1.5, -1.5, 0.0, "test", "", True, False, None, [], {}, ()]
        return [random.choice(mixed_types) for _ in range(n_samples)]

def generate_type_mutation_inputs(original_value):
    """Generate inputs by mutating the type of the original value."""
    if original_value is None:
        return [0, "", False, [], {}]
        
    if isinstance(original_value, (int, float)):
        # Number mutations
        return [
            str(original_value),  # Convert to string
            bool(original_value),  # Convert to boolean
            [original_value],  # Wrap in list
            {original_value: original_value},  # Use as dict key and value
            complex(original_value, 1),  # Convert to complex
        ]
    elif isinstance(original_value, str):
        # String mutations
        try:
            as_int = int(original_value)
        except:
            as_int = 0
            
        try:
            as_float = float(original_value)
        except:
            as_float = 0.0
            
        return [
            as_int,  # Try to convert to int
            as_float,  # Try to convert to float
            bool(original_value),  # Convert to boolean
            [original_value],  # Wrap in list
            {original_value: original_value},  # Use as dict key and value
            original_value.encode() if original_value else b'',  # Convert to bytes
        ]
    elif isinstance(original_value, (list, tuple, set)):
        # Collection mutations
        return [
            str(original_value),  # Convert to string
            bool(original_value),  # Convert to boolean
            {i: v for i, v in enumerate(original_value)} if original_value else {},  # Convert to dict
            tuple(original_value) if isinstance(original_value, (list, set)) else list(original_value),  # Change collection type
        ]
    elif isinstance(original_value, dict):
        # Dict mutations
        return [
            str(original_value),  # Convert to string
            bool(original_value),  # Convert to boolean
            list(original_value.items()),  # Convert to list of tuples
            list(original_value.keys()),  # Get keys as list
            list(original_value.values()),  # Get values as list
        ]
    elif isinstance(original_value, bool):
        # Boolean mutations
        return [
            int(original_value),  # Convert to integer
            float(original_value),  # Convert to float
            str(original_value),  # Convert to string
            [original_value],  # Wrap in list
        ]
    else:
        # Default for other types, try basic mutations
        return [
            str(original_value),
            bool(original_value),
            hash(original_value) if hasattr(original_value, '__hash__') and original_value.__hash__ is not None else 0,
            [original_value],
            {0: original_value},
        ]

def memory_usage():
    """Get current memory usage of the process."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # in MB
    except ImportError:
        # If psutil is not available, return None
        return None

# Core testing functions
def analyze_function_signature(function):
    """
    Analyze a function's signature to understand its parameters and return type.
    
    :param function: The function to analyze
    :return: Dictionary with parameter information
    """
    # Get function signature
    sig = inspect.signature(function)
    params = sig.parameters
    
    # Get function annotations
    annotations = function.__annotations__
    
    # Analyze parameters
    param_info = {}
    for name, param in params.items():
        param_type = annotations.get(name, None)
        default = param.default if param.default is not inspect.Parameter.empty else None
        has_default = param.default is not inspect.Parameter.empty
        
        param_info[name] = {
            'position': list(params.keys()).index(name),
            'type': param_type,
            'has_default': has_default,
            'default': default,
            'kind': str(param.kind)
        }
    
    # Get return type annotation
    return_type = annotations.get('return', None)
    
    # Check for docstring 
    docstring = inspect.getdoc(function)
    
    result = {
        'name': function.__name__,
        'module': function.__module__,
        'parameter_count': len(params),
        'parameters': param_info,
        'return_type': return_type,
        'has_docstring': docstring is not None,
        'docstring_length': len(docstring) if docstring else 0,
        'is_method': False,  # Will be set to True for class methods
        'is_generator': inspect.isgeneratorfunction(function),
        'is_coroutine': inspect.iscoroutinefunction(function),
        'is_async_generator': inspect.isasyncgenfunction(function),
    }
    
    return result

def generate_test_cases_from_signature(signature, n_cases=3):
    """
    Generate test cases based on the function signature.
    
    :param signature: Function signature information from analyze_function_signature
    :param n_cases: Number of test cases to generate per parameter
    :return: List of TestCase objects
    """
    test_cases = []
    param_names = list(signature['parameters'].keys())
    param_count = len(param_names)
    
    # Generate basic test cases for each parameter
    for param_name, param_info in signature['parameters'].items():
        param_type = param_info['type']
        position = param_info['position']
        has_default = param_info['has_default']
        
        # Skip if it's a *args or **kwargs parameter
        if 'VAR' in param_info['kind']:
            continue
        
        # Generate edge cases based on parameter type
        edge_values = []
        
        if param_type is None:
            # If no type annotation, try some common types
            edge_values.extend(generate_numeric_edge_cases()[:n_cases])
            edge_values.extend(generate_string_edge_cases()[:n_cases])
            edge_values.extend(generate_collection_edge_cases()[:n_cases])
        elif param_type == int or param_type is int:
            edge_values.extend(generate_numeric_edge_cases()[:n_cases*2])
        elif param_type == float or param_type is float:
            edge_values.extend([float(x) for x in generate_numeric_edge_cases()[:n_cases*2]])
        elif param_type == str or param_type is str:
            edge_values.extend(generate_string_edge_cases()[:n_cases*2])
        elif param_type == bool or param_type is bool:
            edge_values.extend([True, False, None])
        elif param_type in (list, tuple, set, dict) or param_type is list or param_type is tuple or param_type is set or param_type is dict:
            edge_values.extend(generate_collection_edge_cases()[:n_cases*2])
        else:
            # Try to use type-appropriate fuzzing
            edge_values.extend(generate_fuzz_inputs(param_type, n_cases))
        
        # Create test cases for each edge value
        for value in edge_values:
            # Create inputs tuple with this edge value
            if signature['is_method']:
                # If it's a method, we need at least two parameters (self plus others)
                if param_count < 2:
                    continue
                
                # Create parameters with defaults for all except our target
                inputs = [None]  # placeholder for self
                for i, p_name in enumerate(param_names[1:], start=1):  # Skip self
                    if i - 1 == position - 1:  # Adjust for self
                        inputs.append(value)
                    elif signature['parameters'][p_name]['has_default']:
                        inputs.append(signature['parameters'][p_name]['default'])
                    else:
                        inputs.append(None)  # Placeholder
            else:
                # Regular function
                inputs = []
                for i, p_name in enumerate(param_names):
                    if i == position:
                        inputs.append(value)
                    elif signature['parameters'][p_name]['has_default']:
                        inputs.append(signature['parameters'][p_name]['default'])
                    else:
                        inputs.append(None)  # Placeholder
            
            test_name = f"{param_name}_edge_{type(value).__name__}_{str(value)[:10]}"
            test_cases.append(TestCase(
                name=test_name,
                inputs=tuple(inputs),
                expected=None,  # We don't know the expected output
                category="edge_case",
                tags=[f"param:{param_name}", f"type:{type(value).__name__}"]
            ))
            
        # Also test None if the parameter has a default value
        if has_default:
            inputs = []
            for i, p_name in enumerate(param_names):
                if i == position:
                    inputs.append(None)
                elif signature['parameters'][p_name]['has_default']:
                    inputs.append(signature['parameters'][p_name]['default'])
                else:
                    inputs.append(None)  # Placeholder
                    
            test_cases.append(TestCase(
                name=f"{param_name}_none",
                inputs=tuple(inputs),
                expected=None,  # We don't know the expected output
                category="edge_case",
                tags=[f"param:{param_name}", "type:None"]
            ))
    
    # Create combinations of parameters for stress testing
    if param_count > 1 and len(test_cases) > param_count:
        # Take two random test cases for different parameters
        for _ in range(min(n_cases, param_count * (param_count - 1) // 2)):
            param_indices = random.sample(range(param_count), 2)
            test_case1 = random.choice([tc for tc in test_cases 
                               if f"param:{param_names[param_indices[0]]}" in tc.tags])
            test_case2 = random.choice([tc for tc in test_cases 
                               if f"param:{param_names[param_indices[1]]}" in tc.tags])
            
            # Combine them
            combined_inputs = list(test_case1.inputs)
            param2_pos = signature['parameters'][param_names[param_indices[1]]]['position']
            combined_inputs[param2_pos] = test_case2.inputs[param2_pos]
            
            test_cases.append(TestCase(
                name=f"combo_{param_names[param_indices[0]]}_{param_names[param_indices[1]]}",
                inputs=tuple(combined_inputs),
                expected=None,  # We don't know the expected output
                category="combination",
                tags=["combined_params"]
            ))
    
    return test_cases

def run_test_case(function, test_case):
    """
    Run a single test case and return the result.
    
    :param function: The function to test
    :param test_case: TestCase object
    :return: TestResult object
    """
    # Force garbage collection to get more accurate memory measurements
    gc.collect()
    
    # Capture starting memory
    start_memory = memory_usage()
    
    # Start timing
    start_time = time.time()
    
    result = TestResult(
        test_case=test_case,
        passed=False
    )
    
    try:
        # Call the function with the test inputs
        actual = function(*test_case.inputs)
        execution_time = time.time() - start_time
        
        # Check if an exception was expected
        if test_case.should_raise is not None:
            result.passed = False
            result.note = f"Expected {test_case.should_raise.__name__} but no exception was raised"
        else:
            # Check the result against expected if provided
            if test_case.expected is not None:
                # For numeric results, use almost equal comparison
                if isinstance(actual, (int, float)) and isinstance(test_case.expected, (int, float)):
                    try:
                        # Use numpy's isclose for better floating point comparison
                        if np.isclose(actual, test_case.expected, rtol=1e-5, atol=1e-8, equal_nan=True):
                            result.passed = True
                        else:
                            result.passed = False
                            result.note = f"Expected {test_case.expected} but got {actual}"
                    except:
                        # Fallback to direct comparison
                        result.passed = (actual == test_case.expected)
                        if not result.passed:
                            result.note = f"Expected {test_case.expected} but got {actual}"
                else:
                    # For other types, use direct equality
                    result.passed = (actual == test_case.expected)
                    if not result.passed:
                        result.note = f"Expected {test_case.expected} but got {actual}"
            else:
                # No expected value, consider it passed if no exception was raised
                result.passed = True
                
        result.actual = actual
        
    except Exception as e:
        execution_time = time.time() - start_time
        result.exception = e
        result.traceback = traceback.format_exc()
        
        # Check if this exception was expected
        if test_case.should_raise is not None and isinstance(e, test_case.should_raise):
            result.passed = True
        else:
            result.passed = False
            result.note = f"Unexpected exception: {type(e).__name__}: {str(e)}"
    
    # Calculate memory usage
    end_memory = memory_usage()
    if start_memory is not None and end_memory is not None:
        result.memory_usage = end_memory - start_memory
    
    # Record execution time
    result.execution_time = execution_time
    
    return result

def run_test_cases(function, test_cases):
    """
    Run a list of test cases on a function.
    
    :param function: The function to test
    :param test_cases: List of TestCase objects
    :return: List of TestResult objects
    """
    results = []
    
    for test_case in test_cases:
        logger.info(f"Running test case: {test_case.name}")
        
        # Run the test case
        result = run_test_case(function, test_case)
        
        # Log the result
        if result.passed:
            logger.info(f"✓ PASS: {test_case.name} (Execution time: {result.execution_time:.6f}s)")
        else:
            if result.exception:
                logger.error(f"✗ FAIL: {test_case.name} - {type(result.exception).__name__}: {str(result.exception)}")
                logger.debug(f"Traceback: {result.traceback}")
            else:
                logger.error(f"✗ FAIL: {test_case.name} - {result.note}")
        
        results.append(result)
    
    return results

def fuzz_test_function(function, n_tests=100, input_generator=None):
    """
    Perform fuzz testing on a function with randomly generated inputs.
    
    :param function: The function to test
    :param n_tests: Number of random tests to run
    :param input_generator: Optional custom input generator function
    :return: List of TestResult objects
    """
    # Analyze function signature to understand parameters
    signature = analyze_function_signature(function)
    param_names = list(signature['parameters'].keys())
    param_count = len(param_names)
    
    test_cases = []
    
    for i in range(n_tests):
        # Generate random inputs based on parameter types
        inputs = []
        for param_name in param_names:
            param_info = signature['parameters'][param_name]
            param_type = param_info['type']
            
            # Skip if it's a *args or **kwargs parameter
            if 'VAR' in param_info['kind']:
                continue
                
            # Generate appropriate random value based on type
            if input_generator:
                # Use custom generator if provided
                value = input_generator(param_name, param_type)
            else:
                # Use our built-in random generators
                values = generate_fuzz_inputs(param_type, 1)
                value = values[0] if values else None
                
            inputs.append(value)
            
        # Create test case
        test_case = TestCase(
            name=f"fuzz_test_{i+1}",
            inputs=tuple(inputs),
            expected=None,  # We don't know what to expect
            category="fuzz_test",
            tags=["fuzz", f"test:{i+1}"]
        )
        
        test_cases.append(test_case)
    
    # Run all the test cases
    logger.info(f"Starting fuzz testing with {n_tests} random inputs...")
    results = run_test_cases(function, test_cases)
    logger.info(f"Fuzz testing completed.")
    
    return results

def stress_test_function(function, test_case, repetitions=100):
    """
    Stress test a function by running it repeatedly with the same inputs.
    
    :param function: The function to test
    :param test_case: TestCase object to use repeatedly
    :param repetitions: Number of times to repeat the test
    :return: Dictionary with stress test metrics
    """
    execution_times = []
    memory_usages = []
    passed_count = 0
    exceptions = defaultdict(int)
    
    logger.info(f"Starting stress test with {repetitions} repetitions...")
    
    for i in range(repetitions):
        # Run the test case
        result = run_test_case(function, test_case)
        
        # Track metrics
        execution_times.append(result.execution_time)
        
        if result.memory_usage is not None:
            memory_usages.append(result.memory_usage)
            
        if result.passed:
            passed_count += 1
        else:
            if result.exception:
                exception_type = type(result.exception).__name__
                exceptions[exception_type] += 1
    
    # Calculate statistics
    stats = {
        'repetitions': repetitions,
        'pass_rate': passed_count / repetitions * 100,
        'execution_time': {
            'min': min(execution_times),
            'max': max(execution_times),
            'mean': sum(execution_times) / len(execution_times),
            'median': sorted(execution_times)[len(execution_times) // 2],
            'std': (sum((t - (sum(execution_times) / len(execution_times)))**2 for t in execution_times) / len(execution_times))**0.5
        },
        'exceptions': dict(exceptions)
    }
    
    if memory_usages:
        stats['memory_usage'] = {
            'min': min(memory_usages),
            'max': max(memory_usages),
            'mean': sum(memory_usages) / len(memory_usages)
        }
    
    logger.info(f"Stress test completed. Pass rate: {stats['pass_rate']:.2f}%")
    
    return stats

def load_test_function(function, test_case, concurrency=10, repetitions=10):
    """
    Load test a function by running it concurrently.
    
    :param function: The function to test
    :param test_case: TestCase object
    :param concurrency: Number of concurrent executions
    :param repetitions: Number of repetitions per thread
    :return: Dictionary with load test metrics
    """
    import threading
    
    # Results will be stored in thread-specific lists
    results = {
        'execution_times': [],
        'exceptions': [],
        'passed': [],
    }
    
    # Lock for thread-safe updates to results
    lock = threading.Lock()
    
    def worker():
        local_results = []
        for _ in range(repetitions):
            result = run_test_case(function, test_case)
            local_results.append(result)
        
        # Update global results with thread-local results
        with lock:
            for result in local_results:
                results['execution_times'].append(result.execution_time)
                results['passed'].append(result.passed)
                if result.exception:
                    results['exceptions'].append(type(result.exception).__name__)
    
    logger.info(f"Starting load test with {concurrency} concurrent threads...")
    
    # Create and start threads
    threads = []
    for _ in range(concurrency):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Calculate statistics
    total_executions = len(results['execution_times'])
    passed_count = sum(results['passed'])
    
    stats = {
        'concurrency': concurrency,
        'repetitions_per_thread': repetitions,
        'total_executions': total_executions,
        'pass_rate': passed_count / total_executions * 100 if total_executions > 0 else 0,
        'execution_time': {
            'min': min(results['execution_times']) if results['execution_times'] else None,
            'max': max(results['execution_times']) if results['execution_times'] else None,
            'mean': sum(results['execution_times']) / len(results['execution_times']) if results['execution_times'] else None,
        },
        'exception_counts': {ex: results['exceptions'].count(ex) for ex in set(results['exceptions'])}
    }
    
    logger.info(f"Load test completed. Pass rate: {stats['pass_rate']:.2f}%")
    
    return stats

def analyze_ml_model_robustness(model, X_test, y_test, n_perturbations=10, perturbation_scale=0.01):
    """
    Analyze the robustness of a machine learning model to small perturbations in inputs.
    
    :param model: The ML model to test (must have a predict method)
    :param X_test: Test features
    :param y_test: Test labels/targets
    :param n_perturbations: Number of perturbations to generate
    :param perturbation_scale: Scale of perturbations relative to data range
    :return: Dictionary with robustness metrics
    """
    if not hasattr(model, 'predict'):
        return {'error': 'Model must have a predict method'}
    
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score, mean_squared_error
        
        # Convert to numpy arrays if needed
        X = np.array(X_test)
        y = np.array(y_test)
        
        # Get original predictions and performance
        y_pred_original = model.predict(X)
        
        # Determine if classification or regression
        is_classification = len(np.unique(y)) < 20 or (hasattr(model, 'predict_proba') and len(y_pred_original.shape) > 1)
        
        if is_classification:
            original_performance = accuracy_score(y, y_pred_original)
            performance_metric = 'accuracy'
        else:
            original_performance = mean_squared_error(y, y_pred_original)
            performance_metric = 'mse'
        
        # Calculate data ranges for scaling perturbations
        data_ranges = np.max(X, axis=0) - np.min(X, axis=0)
        data_ranges = np.where(data_ranges > 0, data_ranges, 1.0)  # Avoid division by zero
        
        # Generate perturbations and test
        performance_changes = []
        prediction_changes = []
        perturbation_results = []
        
        for i in range(n_perturbations):
            # Generate random noise scaled to data range
            noise = np.random.normal(0, perturbation_scale, size=X.shape) * data_ranges
            
            # Apply perturbation
            X_perturbed = X + noise
            
            # Get predictions on perturbed data
            y_pred_perturbed = model.predict(X_perturbed)
            
            # Calculate performance on perturbed data
            if is_classification:
                perturbed_performance = accuracy_score(y, y_pred_perturbed)
            else:
                perturbed_performance = mean_squared_error(y, y_pred_perturbed)
            
            # Calculate percentage of predictions that changed
            if is_classification:
                pred_change_rate = np.mean(y_pred_original != y_pred_perturbed) * 100
            else:
                pred_change_rate = np.mean(np.abs(y_pred_original - y_pred_perturbed) > 0.01 * np.std(y_pred_original)) * 100
            
            # Calculate relative performance change
            if is_classification:
                perf_change = original_performance - perturbed_performance
            else:
                perf_change = (perturbed_performance - original_performance) / original_performance
            
            performance_changes.append(perf_change)
            prediction_changes.append(pred_change_rate)
            
            perturbation_results.append({
                'perturbation_id': i + 1,
                'performance_change': perf_change,
                'prediction_change_rate': pred_change_rate
            })
        
        # Calculate robustness metrics
        result = {
            'model_type': 'classification' if is_classification else 'regression',
            'performance_metric': performance_metric,
            'original_performance': original_performance,
            'perturbation_results': perturbation_results,
            'mean_performance_change': np.mean(performance_changes),
            'max_performance_change': np.max(performance_changes),
            'mean_prediction_change_rate': np.mean(prediction_changes),
            'robustness_score': 100 - np.mean(prediction_changes)
        }
        
        # Interpret the robustness
        if result['robustness_score'] > 90:
            result['robustness_level'] = 'High'
        elif result['robustness_score'] > 70:
            result['robustness_level'] = 'Medium'
        else:
            result['robustness_level'] = 'Low'
        
        return result
        
    except Exception as e:
        logger.error(f"Error in ML model robustness analysis: {str(e)}")
        return {'error': str(e)}

# Visualization and reporting functions
def generate_visualization(results, output_dir='robustness_results'):
    """
    Generate visualizations of test results.
    
    :param results: Dictionary with test results
    :param output_dir: Directory to save visualizations
    :return: List of paths to generated visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    visualization_paths = []
    
    try:
        # Extract test results
        test_results = results.get('test_results', [])
        if not test_results:
            return []
        
        # Distribution of execution times
        exec_times = [r.execution_time for r in test_results]
        plt.figure(figsize=(10, 6))
        plt.hist(exec_times, bins=20, alpha=0.7, color='blue')
        plt.xlabel('Execution Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Test Execution Times')
        plt.grid(True, alpha=0.3)
        
        # Add mean and median lines
        if exec_times:
            mean_time = sum(exec_times) / len(exec_times)
            median_time = sorted(exec_times)[len(exec_times) // 2]
            plt.axvline(x=mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.6f}s')
            plt.axvline(x=median_time, color='green', linestyle=':', label=f'Median: {median_time:.6f}s')
            plt.legend()
        
        path = os.path.join(output_dir, 'execution_time_distribution.png')
        plt.savefig(path)
        plt.close()
        visualization_paths.append(path)
        
        # Test outcomes by category
        categories = {}
        for result in test_results:
            category = result.test_case.category
            if category not in categories:
                categories[category] = {'pass': 0, 'fail': 0, 'total': 0}
            
            categories[category]['total'] += 1
            if result.passed:
                categories[category]['pass'] += 1
            else:
                categories[category]['fail'] += 1
        
        if categories:
            category_names = list(categories.keys())
            pass_counts = [categories[cat]['pass'] for cat in category_names]
            fail_counts = [categories[cat]['fail'] for cat in category_names]
            
            plt.figure(figsize=(12, 7))
            bar_width = 0.35
            x = range(len(category_names))
            
            plt.bar([i - bar_width/2 for i in x], pass_counts, bar_width, label='Pass', color='green')
            plt.bar([i + bar_width/2 for i in x], fail_counts, bar_width, label='Fail', color='red')
            
            plt.xlabel('Test Category')
            plt.ylabel('Count')
            plt.title('Test Outcomes by Category')
            plt.xticks(x, category_names, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            path = os.path.join(output_dir, 'test_outcomes_by_category.png')
            plt.savefig(path)
            plt.close()
            visualization_paths.append(path)
        
        # Exception types
        exceptions = {}
        for result in test_results:
            if result.exception:
                exception_type = type(result.exception).__name__
                exceptions[exception_type] = exceptions.get(exception_type, 0) + 1
        
        if exceptions:
            plt.figure(figsize=(10, 6))
            exception_names = list(exceptions.keys())
            exception_counts = list(exceptions.values())
            
            plt.bar(exception_names, exception_counts, color='orange')
            plt.xlabel('Exception Type')
            plt.ylabel('Count')
            plt.title('Exception Types Encountered')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            path = os.path.join(output_dir, 'exception_types.png')
            plt.savefig(path)
            plt.close()
            visualization_paths.append(path)
        
        # Memory usage if available
        memory_usages = [r.memory_usage for r in test_results if r.memory_usage is not None]
        if memory_usages:
            plt.figure(figsize=(10, 6))
            plt.hist(memory_usages, bins=20, alpha=0.7, color='purple')
            plt.xlabel('Memory Usage (MB)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Memory Usage')
            plt.grid(True, alpha=0.3)
            
            path = os.path.join(output_dir, 'memory_usage_distribution.png')
            plt.savefig(path)
            plt.close()
            visualization_paths.append(path)
        
        # Stress test results if available
        if 'stress_test' in results:
            stress_test = results['stress_test']
            exec_times = stress_test.get('execution_time', {})
            
            if 'min' in exec_times and 'max' in exec_times and 'mean' in exec_times:
                plt.figure(figsize=(8, 6))
                plt.bar(['Min', 'Mean', 'Max'], 
                      [exec_times['min'], exec_times['mean'], exec_times['max']], 
                      color=['green', 'blue', 'red'])
                plt.ylabel('Execution Time (seconds)')
                plt.title('Stress Test Performance')
                plt.grid(True, axis='y', alpha=0.3)
                
                path = os.path.join(output_dir, 'stress_test_performance.png')
                plt.savefig(path)
                plt.close()
                visualization_paths.append(path)
        
        # Load test results if available
        if 'load_test' in results:
            load_test = results['load_test']
            
            # Could create more sophisticated visualizations here
            # For example, showing how performance changes with concurrency
            
        # ML model robustness if available
        if 'ml_robustness' in results and isinstance(results['ml_robustness'], dict):
            robustness = results['ml_robustness']
            
            if 'perturbation_results' in robustness:
                perturbations = robustness['perturbation_results']
                perf_changes = [p['performance_change'] for p in perturbations]
                pred_changes = [p['prediction_change_rate'] for p in perturbations]
                
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                color = 'tab:red'
                ax1.set_xlabel('Perturbation ID')
                ax1.set_ylabel('Performance Change', color=color)
                ax1.plot(range(1, len(perf_changes) + 1), perf_changes, marker='o', color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                
                ax2 = ax1.twinx()
                color = 'tab:blue'
                ax2.set_ylabel('Prediction Change Rate (%)', color=color)
                ax2.plot(range(1, len(pred_changes) + 1), pred_changes, marker='s', color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                
                plt.title('ML Model Robustness to Perturbations')
                plt.grid(True, alpha=0.3)
                fig.tight_layout()
                
                path = os.path.join(output_dir, 'ml_robustness.png')
                plt.savefig(path)
                plt.close()
                visualization_paths.append(path)
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
    
    return visualization_paths

def generate_html_report(results, output_file='robustness_report.html', visualizations=None):
    """
    Generate a comprehensive HTML report of the robustness testing results.
    
    :param results: Dictionary with test results
    :param output_file: Path to output HTML file
    :param visualizations: List of paths to visualization images
    :return: Boolean indicating success
    """
    try:
        # Basic HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Robustness Testing Report</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; margin-top: 20px; }
                h3 { color: #2c3e50; }
                .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .passed { color: #27ae60; }
                .failed { color: #e74c3c; }
                .test-result { margin-bottom: 10px; padding: 10px; border-radius: 5px; }
                .test-result.pass { background-color: #d4edda; }
                .test-result.fail { background-color: #f8d7da; }
                .exception { font-family: monospace; background-color: #f8f9fa; padding: 10px; border-left: 3px solid #e74c3c; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .visualization { max-width: 100%; margin: 20px 0; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>Robustness Testing Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                {summary_html}
            </div>
            
            <h2>Test Results</h2>
            {test_results_html}
            
            {stress_test_html}
            
            {load_test_html}
            
            {fuzz_test_html}
            
            {ml_robustness_html}
            
            <h2>Visualizations</h2>
            {visualizations_html}
            
            <footer>
                <p>Generated on {timestamp}</p>
            </footer>
        </body>
        </html>
        """
        
        # Generate summary section
        test_results = results.get('test_results', [])
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.passed)
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        summary_html = f"""
        <p><strong>Function:</strong> {results.get('function_name', 'Unknown')}</p>
        <p><strong>Total Tests:</strong> {total_tests}</p>
        <p><strong>Passed Tests:</strong> {passed_tests} ({pass_rate:.2f}%)</p>
        <p><strong>Failed Tests:</strong> {total_tests - passed_tests} ({100 - pass_rate:.2f}%)</p>
        """
        
        if 'robustness_score' in results:
            score = results['robustness_score']
            level = "High" if score >= 80 else "Medium" if score >= 60 else "Low"
            summary_html += f"<p><strong>Overall Robustness Score:</strong> {score:.2f}/100 ({level})</p>"
        
        # Generate test results section
        test_results_html = ""
        if test_results:
            # Group by category
            categories = {}
            for result in test_results:
                category = result.test_case.category
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)
            
            # Generate HTML for each category
            for category, category_results in categories.items():
                test_results_html += f"<h3>{category.title()} ({len(category_results)} tests)</h3>"
                
                # Summary table for this category
                category_pass = sum(1 for r in category_results if r.passed)
                category_pass_rate = (category_pass / len(category_results) * 100) if category_results else 0
                
                test_results_html += f"""
                <p>Pass rate: {category_pass_rate:.2f}% ({category_pass}/{len(category_results)})</p>
                <table>
                    <thead>
                        <tr>
                            <th>Test Name</th>
                            <th>Result</th>
                            <th>Execution Time</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for result in category_results:
                    # Determine result styling
                    result_class = "pass" if result.passed else "fail"
                    result_text = "PASS" if result.passed else "FAIL"
                    
                    # Format details
                    if result.exception:
                        details = f"{type(result.exception).__name__}: {str(result.exception)}"
                    elif result.note:
                        details = result.note
                    else:
                        details = "N/A"
                    
                    test_results_html += f"""
                    <tr class="{result_class}">
                        <td>{result.test_case.name}</td>
                        <td class="{'passed' if result.passed else 'failed'}">{result_text}</td>
                        <td>{result.execution_time:.6f}s</td>
                        <td>{details}</td>
                    </tr>
                    """
                
                test_results_html += """
                    </tbody>
                </table>
                """
        else:
            test_results_html = "<p>No test results available.</p>"
        
        # Generate stress test section
        stress_test_html = ""
        if 'stress_test' in results:
            stress_test = results['stress_test']
            stress_test_html = """
            <h2>Stress Test Results</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
            """
            
            stress_test_html += f"<tr><td>Repetitions</td><td>{stress_test.get('repetitions', 'N/A')}</td></tr>"
            stress_test_html += f"<tr><td>Pass Rate</td><td>{stress_test.get('pass_rate', 0):.2f}%</td></tr>"
            
            # Add execution time metrics
            if 'execution_time' in stress_test:
                exec_time = stress_test['execution_time']
                stress_test_html += f"<tr><td>Min Execution Time</td><td>{exec_time.get('min', 'N/A'):.6f}s</td></tr>"
                stress_test_html += f"<tr><td>Max Execution Time</td><td>{exec_time.get('max', 'N/A'):.6f}s</td></tr>"
                stress_test_html += f"<tr><td>Mean Execution Time</td><td>{exec_time.get('mean', 'N/A'):.6f}s</td></tr>"
                stress_test_html += f"<tr><td>Median Execution Time</td><td>{exec_time.get('median', 'N/A'):.6f}s</td></tr>"
                
            # Add memory usage metrics if available
            if 'memory_usage' in stress_test:
                mem_usage = stress_test['memory_usage']
                stress_test_html += f"<tr><td>Min Memory Usage</td><td>{mem_usage.get('min', 'N/A'):.2f} MB</td></tr>"
                stress_test_html += f"<tr><td>Max Memory Usage</td><td>{mem_usage.get('max', 'N/A'):.2f} MB</td></tr>"
                stress_test_html += f"<tr><td>Mean Memory Usage</td><td>{mem_usage.get('mean', 'N/A'):.2f} MB</td></tr>"
                
            # Add exception counts if any
            if 'exceptions' in stress_test and stress_test['exceptions']:
                stress_test_html += "<tr><td>Exceptions</td><td>"
                for ex_type, count in stress_test['exceptions'].items():
                    stress_test_html += f"{ex_type}: {count}<br>"
                stress_test_html += "</td></tr>"
                
            stress_test_html += "</table>"
        
        # Generate load test section
        load_test_html = ""
        if 'load_test' in results:
            load_test = results['load_test']
            load_test_html = """
            <h2>Load Test Results</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
            """
            
            load_test_html += f"<tr><td>Concurrency</td><td>{load_test.get('concurrency', 'N/A')}</td></tr>"
            load_test_html += f"<tr><td>Repetitions per Thread</td><td>{load_test.get('repetitions_per_thread', 'N/A')}</td></tr>"
            load_test_html += f"<tr><td>Total Executions</td><td>{load_test.get('total_executions', 'N/A')}</td></tr>"
            load_test_html += f"<tr><td>Pass Rate</td><td>{load_test.get('pass_rate', 0):.2f}%</td></tr>"
            
            # Add execution time metrics
            if 'execution_time' in load_test:
                exec_time = load_test['execution_time']
                load_test_html += f"<tr><td>Min Execution Time</td><td>{exec_time.get('min', 'N/A'):.6f}s</td></tr>"
                load_test_html += f"<tr><td>Max Execution Time</td><td>{exec_time.get('max', 'N/A'):.6f}s</td></tr>"
                load_test_html += f"<tr><td>Mean Execution Time</td><td>{exec_time.get('mean', 'N/A'):.6f}s</td></tr>"
                
            # Add exception counts if any
            if 'exception_counts' in load_test and load_test['exception_counts']:
                load_test_html += "<tr><td>Exceptions</td><td>"
                for ex_type, count in load_test['exception_counts'].items():
                    load_test_html += f"{ex_type}: {count}<br>"
                load_test_html += "</td></tr>"
                
            load_test_html += "</table>"
        
        # Generate fuzz test section
        fuzz_test_html = ""
        if 'fuzz_test_results' in results:
            fuzz_results = results['fuzz_test_results']
            passed_fuzz = sum(1 for r in fuzz_results if r.passed)
            total_fuzz = len(fuzz_results)
            pass_rate_fuzz = (passed_fuzz / total_fuzz * 100) if total_fuzz > 0 else 0
            
            fuzz_test_html = f"""
            <h2>Fuzz Test Results</h2>
            <p>Total fuzz tests: {total_fuzz}</p>
            <p>Pass rate: {pass_rate_fuzz:.2f}% ({passed_fuzz}/{total_fuzz})</p>
            """
            
            # Count exception types
            exception_counts = {}
            for result in fuzz_results:
                if result.exception:
                    ex_type = type(result.exception).__name__
                    exception_counts[ex_type] = exception_counts.get(ex_type, 0) + 1
            
            if exception_counts:
                fuzz_test_html += """
                <h3>Exception Types Encountered</h3>
                <table>
                    <tr><th>Exception Type</th><th>Count</th><th>Percentage</th></tr>
                """
                
                for ex_type, count in sorted(exception_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / total_fuzz * 100
                    fuzz_test_html += f"<tr><td>{ex_type}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>"
                
                fuzz_test_html += "</table>"
        
        # Generate ML robustness section
        ml_robustness_html = ""
        if 'ml_robustness' in results:
            robustness = results['ml_robustness']
            
            if 'error' in robustness:
                ml_robustness_html = f"""
                <h2>ML Model Robustness</h2>
                <p class="failed">Error analyzing model robustness: {robustness['error']}</p>
                """
            else:
                ml_robustness_html = """
                <h2>ML Model Robustness</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                """
                
                ml_robustness_html += f"<tr><td>Model Type</td><td>{robustness.get('model_type', 'Unknown')}</td></tr>"
                ml_robustness_html += f"<tr><td>Performance Metric</td><td>{robustness.get('performance_metric', 'Unknown')}</td></tr>"
                ml_robustness_html += f"<tr><td>Original Performance</td><td>{robustness.get('original_performance', 'N/A'):.4f}</td></tr>"
                ml_robustness_html += f"<tr><td>Mean Performance Change</td><td>{robustness.get('mean_performance_change', 'N/A'):.4f}</td></tr>"
                ml_robustness_html += f"<tr><td>Mean Prediction Change Rate</td><td>{robustness.get('mean_prediction_change_rate', 'N/A'):.2f}%</td></tr>"
                ml_robustness_html += f"<tr><td>Robustness Score</td><td>{robustness.get('robustness_score', 'N/A'):.2f}/100</td></tr>"
                ml_robustness_html += f"<tr><td>Robustness Level</td><td>{robustness.get('robustness_level', 'Unknown')}</td></tr>"
                
                ml_robustness_html += "</table>"
        
        # Generate visualizations section
        visualizations_html = ""
        if visualizations:
            for viz_path in visualizations:
                # Get relative path for HTML
                viz_filename = os.path.basename(viz_path)
                visualizations_html += f'<div><img src="{viz_filename}" alt="{viz_filename}" class="visualization"/></div>'
        else:
            visualizations_html = "<p>No visualizations available.</p>"
        
        # Fill in the template
        html_content = html_template.format(
            summary_html=summary_html,
            test_results_html=test_results_html,
            stress_test_html=stress_test_html,
            load_test_html=load_test_html,
            fuzz_test_html=fuzz_test_html,
            ml_robustness_html=ml_robustness_html,
            visualizations_html=visualizations_html,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        return False

def generate_json_report(results, output_file='robustness_report.json'):
    """
    Generate a JSON report of the robustness testing results.
    
    :param results: Dictionary with test results
    :param output_file: Path to output JSON file
    :return: Boolean indicating success
    """
    try:
        # Create a serializable version of the results
        json_results = {
            'function_name': results.get('function_name', 'Unknown'),
            'timestamp': datetime.datetime.now().isoformat(),
            'test_summary': {
                'total_tests': len(results.get('test_results', [])),
                'passed_tests': sum(1 for r in results.get('test_results', []) if r.passed),
                'categories': {}
            }
        }
        
        # Add robustness score if available
        if 'robustness_score' in results:
            json_results['robustness_score'] = results['robustness_score']
            json_results['robustness_level'] = results['robustness_level']
        
        # Process test results
        test_results = results.get('test_results', [])
        if test_results:
            # Categorize tests
            categories = {}
            for result in test_results:
                category = result.test_case.category
                if category not in categories:
                    categories[category] = {'total': 0, 'passed': 0}
                
                categories[category]['total'] += 1
                if result.passed:
                    categories[category]['passed'] += 1
            
            # Add to report
            for category, counts in categories.items():
                json_results['test_summary']['categories'][category] = {
                    'total': counts['total'],
                    'passed': counts['passed'],
                    'pass_rate': (counts['passed'] / counts['total'] * 100) if counts['total'] > 0 else 0
                }
            
            # Add detailed test results
            json_results['detailed_results'] = []
            for result in test_results:
                json_result = {
                    'name': result.test_case.name,
                    'category': result.test_case.category,
                    'passed': result.passed,
                    'execution_time': result.execution_time
                }
                
                if result.memory_usage is not None:
                    json_result['memory_usage'] = result.memory_usage
                
                if result.exception:
                    json_result['exception'] = {
                        'type': type(result.exception).__name__,
                        'message': str(result.exception)
                    }
                
                if result.note:
                    json_result['note'] = result.note
                    
                json_results['detailed_results'].append(json_result)
        
        # Add stress test results
        if 'stress_test' in results:
            json_results['stress_test'] = results['stress_test']
        
        # Add load test results
        if 'load_test' in results:
            json_results['load_test'] = results['load_test']
        
        # Add fuzz test results
        if 'fuzz_test_results' in results:
            fuzz_results = results['fuzz_test_results']
            json_results['fuzz_test'] = {
                'total_tests': len(fuzz_results),
                'passed_tests': sum(1 for r in fuzz_results if r.passed),
                'pass_rate': (sum(1 for r in fuzz_results if r.passed) / len(fuzz_results) * 100) if fuzz_results else 0,
                'exception_counts': {}
            }
            
            # Count exception types
            for result in fuzz_results:
                if result.exception:
                    ex_type = type(result.exception).__name__
                    if ex_type not in json_results['fuzz_test']['exception_counts']:
                        json_results['fuzz_test']['exception_counts'][ex_type] = 0
                    json_results['fuzz_test']['exception_counts'][ex_type] += 1
        
        # Add ML robustness results
        if 'ml_robustness' in results:
            json_results['ml_robustness'] = results['ml_robustness']
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        logger.info(f"JSON report saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating JSON report: {str(e)}")
        return False

# Main testing function
def test_function_robustness(function, custom_test_cases=None, output_dir='robustness_results'):
    """
    Perform comprehensive robustness testing on a function.
    
    :param function: The function to test
    :param custom_test_cases: Optional list of custom TestCase objects
    :param output_dir: Directory to save results and reports
    :return: Dictionary with all test results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'function_name': function.__name__,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Analyze function signature
    logger.info(f"Analyzing function: {function.__name__}")
    signature = analyze_function_signature(function)
    results['function_signature'] = signature
    
    # Generate test cases from signature
    auto_test_cases = generate_test_cases_from_signature(signature)
    
    # Combine with custom test cases if provided
    test_cases = auto_test_cases
    if custom_test_cases:
        test_cases.extend(custom_test_cases)
    
    # Run all test cases
    logger.info(f"Running {len(test_cases)} test cases...")
    test_results = run_test_cases(function, test_cases)
    results['test_results'] = test_results
    
    # Calculate pass rate statistics
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r.passed)
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    logger.info(f"Test execution complete. Pass rate: {pass_rate:.2f}% ({passed_tests}/{total_tests})")
    
    # Run stress test with a representative test case
    if test_results and any(r.passed for r in test_results):
        # Find a passing test case
        representative_test = next((r.test_case for r in test_results if r.passed), None)
        
        if representative_test:
            logger.info("Running stress test...")
            stress_results = stress_test_function(function, representative_test, repetitions=100)
            results['stress_test'] = stress_results
    
    # Run load test
    if test_results and any(r.passed for r in test_results):
        # Use the same representative test case
        representative_test = next((r.test_case for r in test_results if r.passed), None)
        
        if representative_test:
            logger.info("Running load test...")
            load_results = load_test_function(function, representative_test, concurrency=5, repetitions=10)
            results['load_test'] = load_results
    
    # Run fuzz testing
    logger.info("Running fuzz testing...")
    fuzz_results = fuzz_test_function(function, n_tests=50)
    results['fuzz_test_results'] = fuzz_results
    
    # Calculate robustness score
    # 50% from pass rate, 20% from stress test, 15% from load test, 15% from fuzz test
    robustness_score = pass_rate * 0.5  # Base on pass rate
    
    # Add stress test component
    if 'stress_test' in results:
        stress_score = results['stress_test']['pass_rate']
        robustness_score += stress_score * 0.2
    
    # Add load test component
    if 'load_test' in results:
        load_score = results['load_test']['pass_rate']
        robustness_score += load_score * 0.15
    
    # Add fuzz test component
    fuzz_pass_rate = (sum(1 for r in fuzz_results if r.passed) / len(fuzz_results) * 100) if fuzz_results else 0
    robustness_score += fuzz_pass_rate * 0.15
    
    # Determine robustness level
    if robustness_score >= 80:
        robustness_level = "High"
    elif robustness_score >= 60:
        robustness_level = "Medium"
    else:
        robustness_level = "Low"
    
    results['robustness_score'] = robustness_score
    results['robustness_level'] = robustness_level
    
    logger.info(f"Robustness score: {robustness_score:.2f}/100 ({robustness_level})")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    visualization_paths = generate_visualization(results, output_dir)
    
    # Generate reports
    logger.info("Generating reports...")
    html_report_path = os.path.join(output_dir, 'robustness_report.html')
    generate_html_report(results, html_report_path, visualization_paths)
    
    json_report_path = os.path.join(output_dir, 'robustness_report.json')
    generate_json_report(results, json_report_path)
    
    logger.info(f"Robustness testing complete. Reports saved to {output_dir}")
    
    return results

def test_ml_model_robustness(model, X_test, y_test, custom_test_cases=None, output_dir='robustness_results'):
    """
    Perform robustness testing specific to machine learning models.
    
    :param model: The ML model to test (must have predict method)
    :param X_test: Test features
    :param y_test: Test labels/targets
    :param custom_test_cases: Optional list of custom test cases
    :param output_dir: Directory to save results and reports
    :return: Dictionary with all test results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'model_name': type(model).__name__,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Analyze robustness to perturbations
    logger.info("Analyzing ML model robustness to perturbations...")
    ml_robustness = analyze_ml_model_robustness(model, X_test, y_test)
    results['ml_robustness'] = ml_robustness
    
    # If the model has a predict method, test it as a function
    if hasattr(model, 'predict'):
        logger.info("Testing model predict function...")
        
        # Create a wrapper function to test
        def predict_wrapper(*args):
            if len(args) == 1:
                return model.predict(args[0])
            return model.predict(*args)
        
        # Create some test cases
        test_cases = []
        
        # Sample a few examples from X_test
        if len(X_test) > 0:
            for i in range(min(5, len(X_test))):
                # Single sample test
                if hasattr(X_test, 'iloc'):
                    sample = X_test.iloc[[i]]
                else:
                    sample = X_test[[i]]
                
                test_cases.append(TestCase(
                    name=f"sample_{i+1}",
                    inputs=(sample,),
                    category="sample_prediction"
                ))
        
        # Add custom test cases if provided
        if custom_test_cases:
            test_cases.extend(custom_test_cases)
        
        # Run test cases
        if test_cases:
            logger.info(f"Running {len(test_cases)} test cases...")
            test_results = run_test_cases(predict_wrapper, test_cases)
            results['test_results'] = test_results
    
    # Generate reports
    logger.info("Generating reports...")
    html_report_path = os.path.join(output_dir, 'ml_robustness_report.html')
    generate_html_report(results, html_report_path)
    
    json_report_path = os.path.join(output_dir, 'ml_robustness_report.json')
    generate_json_report(results, json_report_path)
    
    logger.info(f"ML model robustness testing complete. Reports saved to {output_dir}")
    
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Robustness Testing Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--function",
        help="Name of function to test"
    )
    
    parser.add_argument(
        "--module",
        help="Module containing the function"
    )
    
    parser.add_argument(
        "--ml-model",
        action="store_true",
        help="Test as an ML model"
    )
    
    parser.add_argument(
        "--data-file",
        help="Data file for ML model testing (CSV format)"
    )
    
    parser.add_argument(
        "--target-column",
        help="Target column name for ML model testing"
    )
    
    parser.add_argument(
        "--output-dir",
        default="robustness_results",
        help="Directory to save test results"
    )
    
    parser.add_argument(
        "--n-fuzz-tests",
        type=int,
        default=50,
        help="Number of fuzz tests to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

# Example test function
def example_function(x):
    return 10 / x  # Division operation that may fail on zero

# Run robustness tests
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)
    
    # Determine function to test
    test_func = None
    
    if args.function and args.module:
        # Import function from module
        try:
            module = __import__(args.module, fromlist=[args.function])
            test_func = getattr(module, args.function)
        except (ImportError, AttributeError) as e:
            logger.error(f"Error importing function: {e}")
            sys.exit(1)
    else:
        # Use example function
        test_func = example_function
        logger.info("No function specified, using example_function")
    
    # Define edge case tests
    test_cases = [
        TestCase("Zero Division", (0,), None, "edge_case", ZeroDivisionError, tags=["expected_error"]),
        TestCase("Negative Input", (-5,), -2.0, "edge_case"),
        TestCase("Large Number", (1e6,), 1e-5, "edge_case"),
        TestCase("Small Number", (1e-6,), 1e7, "edge_case"),
        TestCase("Integer Input", (5,), 2.0, "edge_case"),
        TestCase("Float Input", (5.0,), 2.0, "edge_case"),
    ]
    
    # Testing path
    if args.ml_model and args.data_file:
        # ML model testing
        try:
            # Check for required packages
            import pandas as pd
            import numpy as np
            
            # Load data
            data = pd.read_csv(args.data_file)
            
            if args.target_column:
                X = data.drop(columns=[args.target_column])
                y = data[args.target_column]
            else:
                # Assume last column is target
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]
            
            # Split data for testing
            test_size = min(len(X), 100)  # Use at most 100 samples for testing
            X_test = X.iloc[:test_size]
            y_test = y.iloc[:test_size]
            
            # Test the model
            test_ml_model_robustness(test_func, X_test, y_test, test_cases, args.output_dir)
            
        except ImportError:
            logger.error("ML model testing requires pandas and numpy")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error testing ML model: {e}")
            sys.exit(1)
            
    else:
        # Function testing
        test_function_robustness(test_func, test_cases, args.output_dir)
    
    print(f"Robustness testing complete. Reports saved to {args.output_dir}")
    print(f"Check '{args.output_dir}/robustness_report.html' for detailed results.")

"""
TODO:
- Add adversarial testing for ML models to find inputs that cause misclassification
- Implement distributed testing across multiple environments and platforms
- Add support for testing API endpoints and network services 
- Integrate with CI/CD pipelines for automated regression testing
- Implement domain-specific testing strategies (e.g., finance, healthcare)
- Add security vulnerability scanning for code robustness
- Implement metamorphic testing for complex functions without clear expected outputs
- Add support for testing asynchronous and concurrent code
- Extend visualization capabilities with interactive dashboards
- Implement legal compliance checking for regulatory requirements
"""
