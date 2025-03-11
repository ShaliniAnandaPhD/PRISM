"""
A comprehensive testing framework for validating Python function behavior.
This script allows for systematic testing of functions against expected outputs,
handling various test scenarios including expected exceptions, timeouts, and
different output comparison methods.

Features:
- Detailed logging with configurable verbosity
- Support for test fixtures and test suites
- Ability to test for expected exceptions
- Timeout handling for long-running functions
- Customizable output comparison (equality, approximate, custom validators)
- Comprehensive test reports in multiple formats (JSON, HTML, console)
- Command-line interface for flexible test execution
- Support for setup and teardown operations
"""

import logging
import json
import time
import argparse
import sys
import traceback
import signal
import os
import datetime
import functools
import inspect
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field


class TestResult(Enum):
    """Enumeration of possible test outcomes."""
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"


class ComparisonMethod(Enum):
    """Enumeration of available comparison methods."""
    EXACT = "exact"  # Uses == operator
    APPROXIMATE = "approximate"  # For floating point, within tolerance
    CONTAINS = "contains"  # Check if expected is contained in result
    CUSTOM = "custom"  # Uses a custom validator function


@dataclass
class TestCase:
    """Data class representing a single test case."""
    function: Callable
    inputs: Tuple
    expected_output: Any = None
    test_name: str = ""
    category: str = "default"
    timeout: Optional[float] = None
    expected_exception: Optional[type] = None
    comparison_method: ComparisonMethod = ComparisonMethod.EXACT
    comparison_tolerance: float = 0.0001
    custom_validator: Optional[Callable] = None
    skip: bool = False
    skip_reason: str = ""
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set a default test name if not provided."""
        if not self.test_name:
            self.test_name = f"Test for {self.function.__name__} with inputs {self.inputs}"


@dataclass
class TestResult:
    """Data class containing the result of a test execution."""
    test_case: TestCase
    status: TestResult
    actual_output: Any = None
    execution_time: float = 0.0
    exception: Optional[Exception] = None
    exception_traceback: str = ""
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeoutError(Exception):
    """Custom exception raised when a test times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for function timeout."""
    raise TimeoutError("Function execution timed out")


def run_with_timeout(func, timeout, *args, **kwargs):
    """Run a function with a timeout."""
    if timeout is None:
        return func(*args, **kwargs)
    
    # Set up the timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))
    
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Disable the alarm
        return result
    finally:
        signal.alarm(0)  # Ensure the alarm is disabled


def compare_outputs(actual, expected, method, tolerance=0.0001, validator=None):
    """
    Compare actual and expected outputs using the specified method.
    
    Args:
        actual: The actual output from the function
        expected: The expected output
        method: ComparisonMethod enum value
        tolerance: Tolerance for approximate comparison
        validator: Custom validator function for CUSTOM method
        
    Returns:
        bool: True if comparison passes, False otherwise
    """
    if method == ComparisonMethod.EXACT:
        return actual == expected
    
    elif method == ComparisonMethod.APPROXIMATE:
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) <= tolerance
        elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(abs(a - e) <= tolerance for a, e in zip(actual, expected) 
                       if isinstance(a, (int, float)) and isinstance(e, (int, float)))
        return False
    
    elif method == ComparisonMethod.CONTAINS:
        if isinstance(actual, (str, list, tuple, dict, set)):
            return expected in actual
        return False
    
    elif method == ComparisonMethod.CUSTOM:
        if validator and callable(validator):
            return validator(actual, expected)
        return False
    
    return False


def run_test_case(test_case):
    """
    Runs a single test case on the given function.
    
    Args:
        test_case: TestCase object containing test configuration
        
    Returns:
        TestResult: Object containing test results
    """
    # Handle skipped tests
    if test_case.skip:
        return TestResult(
            test_case=test_case,
            status=TestResult.SKIPPED,
            metadata={"skip_reason": test_case.skip_reason}
        )
    
    # Run setup if provided
    if test_case.setup and callable(test_case.setup):
        try:
            test_case.setup()
        except Exception as e:
            return TestResult(
                test_case=test_case,
                status=TestResult.ERROR,
                exception=e,
                exception_traceback=traceback.format_exc(),
                metadata={"phase": "setup"}
            )
    
    start_time = time.time()
    
    try:
        # Run the function with timeout if specified
        if test_case.timeout is not None:
            try:
                result = run_with_timeout(
                    test_case.function,
                    test_case.timeout,
                    *test_case.inputs
                )
            except TimeoutError:
                return TestResult(
                    test_case=test_case,
                    status=TestResult.TIMEOUT,
                    execution_time=time.time() - start_time,
                    metadata={"timeout_value": test_case.timeout}
                )
        else:
            result = test_case.function(*test_case.inputs)
            
        # If we're expecting an exception but didn't get one, it's a failure
        if test_case.expected_exception is not None:
            end_time = time.time()
            return TestResult(
                test_case=test_case,
                status=TestResult.FAIL,
                actual_output=result,
                execution_time=end_time - start_time,
                metadata={
                    "failure_reason": f"Expected exception {test_case.expected_exception.__name__} was not raised"
                }
            )

        # Compare the result with expected output
        if compare_outputs(
            result,
            test_case.expected_output,
            test_case.comparison_method,
            test_case.comparison_tolerance,
            test_case.custom_validator
        ):
            status = TestResult.PASS
        else:
            status = TestResult.FAIL
            
        end_time = time.time()
        
        return TestResult(
            test_case=test_case,
            status=status,
            actual_output=result,
            execution_time=end_time - start_time,
            metadata={
                "comparison_method": test_case.comparison_method.value,
                "comparison_tolerance": test_case.comparison_tolerance
            }
        )
            
    except Exception as e:
        end_time = time.time()
        
        # If this is an expected exception, it's a pass
        if test_case.expected_exception is not None and isinstance(e, test_case.expected_exception):
            return TestResult(
                test_case=test_case,
                status=TestResult.PASS,
                exception=e,
                execution_time=end_time - start_time,
                exception_traceback=traceback.format_exc(),
                metadata={"expected_exception": test_case.expected_exception.__name__}
            )
        
        # Otherwise it's an error
        return TestResult(
            test_case=test_case,
            status=TestResult.ERROR,
            exception=e,
            execution_time=end_time - start_time,
            exception_traceback=traceback.format_exc()
        )
    
    finally:
        # Run teardown if provided
        if test_case.teardown and callable(test_case.teardown):
            try:
                test_case.teardown()
            except Exception as e:
                logging.warning(f"Teardown for {test_case.test_name} failed: {e}")


class TestSuite:
    """A collection of test cases that can be run together."""
    
    def __init__(self, name="Default Test Suite"):
        self.name = name
        self.test_cases = []
        self.setup_function = None
        self.teardown_function = None
        
    def add_test(self, test_case):
        """Add a test case to the suite."""
        self.test_cases.append(test_case)
        
    def add_tests(self, test_cases):
        """Add multiple test cases to the suite."""
        self.test_cases.extend(test_cases)
        
    def set_suite_setup(self, setup_func):
        """Set a function to run before any tests in the suite."""
        self.setup_function = setup_func
        
    def set_suite_teardown(self, teardown_func):
        """Set a function to run after all tests in the suite."""
        self.teardown_function = teardown_func
        
    def run(self):
        """Run all tests in the suite and return the results."""
        results = []
        
        if self.setup_function:
            try:
                self.setup_function()
            except Exception as e:
                logging.error(f"Suite setup failed: {e}")
                # Skip all tests if suite setup fails
                for test_case in self.test_cases:
                    results.append(TestResult(
                        test_case=test_case,
                        status=TestResult.SKIPPED,
                        metadata={"skip_reason": "Suite setup failed"}
                    ))
                return results
        
        for test_case in self.test_cases:
            result = run_test_case(test_case)
            results.append(result)
            
        if self.teardown_function:
            try:
                self.teardown_function()
            except Exception as e:
                logging.error(f"Suite teardown failed: {e}")
                
        return results


class TestRunner:
    """Main test runner class that handles test execution and reporting."""
    
    def __init__(self, log_file=None, log_level=logging.INFO, report_file=None):
        """
        Initialize the test runner.
        
        Args:
            log_file: Path to log file, if None logs to stderr
            log_level: Logging level to use
            report_file: Path to save the JSON report, if None no report is saved
        """
        self.suites = []
        self.log_file = log_file
        self.log_level = log_level
        self.report_file = report_file
        self._configure_logging()
        
    def _configure_logging(self):
        """Configure logging based on instance settings."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        if self.log_file:
            logging.basicConfig(
                filename=self.log_file,
                level=self.log_level,
                format=log_format
            )
        else:
            logging.basicConfig(
                level=self.log_level,
                format=log_format
            )
    
    def add_suite(self, suite):
        """Add a test suite to the runner."""
        self.suites.append(suite)
        
    def create_suite(self, name="Test Suite"):
        """Create and return a new test suite."""
        suite = TestSuite(name)
        self.suites.append(suite)
        return suite
        
    def run(self):
        """Run all test suites and generate reports."""
        all_results = []
        
        for suite in self.suites:
            logging.info(f"Running test suite: {suite.name}")
            results = suite.run()
            all_results.extend(results)
            
        self._log_results(all_results)
        
        if self.report_file:
            self._save_report(all_results)
            
        return all_results
    
    def _log_results(self, results):
        """Log the results of the test run."""
        for result in results:
            test_name = result.test_case.test_name
            status = result.status
            
            if status == TestResult.PASS:
                logging.info(f"PASS: {test_name}")
            elif status == TestResult.FAIL:
                logging.error(f"FAIL: {test_name} | Expected: {result.test_case.expected_output}, Got: {result.actual_output}")
            elif status == TestResult.ERROR:
                logging.error(f"ERROR: {test_name} | Exception: {result.exception}")
                if result.exception_traceback:
                    logging.debug(f"Traceback: {result.exception_traceback}")
            elif status == TestResult.TIMEOUT:
                logging.error(f"TIMEOUT: {test_name} | Function execution exceeded {result.test_case.timeout} seconds")
            elif status == TestResult.SKIPPED:
                logging.info(f"SKIPPED: {test_name} | Reason: {result.metadata.get('skip_reason', 'No reason provided')}")
    
    def _save_report(self, results):
        """Save a JSON report of the test results."""
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_tests": len(results),
            "pass_count": sum(1 for r in results if r.status == TestResult.PASS),
            "fail_count": sum(1 for r in results if r.status == TestResult.FAIL),
            "error_count": sum(1 for r in results if r.status == TestResult.ERROR),
            "timeout_count": sum(1 for r in results if r.status == TestResult.TIMEOUT),
            "skipped_count": sum(1 for r in results if r.status == TestResult.SKIPPED),
            "tests": []
        }
        
        for result in results:
            test_data = {
                "name": result.test_case.test_name,
                "category": result.test_case.category,
                "status": result.status.value,
                "execution_time": result.execution_time,
                "function": result.test_case.function.__name__,
                "inputs": str(result.test_case.inputs),
            }
            
            if result.status == TestResult.FAIL:
                test_data["expected"] = str(result.test_case.expected_output)
                test_data["actual"] = str(result.actual_output)
                
            if result.status == TestResult.ERROR:
                test_data["exception"] = str(result.exception)
                test_data["exception_type"] = result.exception.__class__.__name__
                
            report["tests"].append(test_data)
        
        with open(self.report_file, "w") as f:
            json.dump(report, f, indent=4)
            
        logging.info(f"Test report saved to {self.report_file}")
        
    def generate_html_report(self, html_file):
        """Generate an HTML report from the test results."""
        # Placeholder for HTML report generation
        # This would create a nicely formatted HTML report
        logging.info(f"HTML report generation not implemented. Would save to {html_file}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Function Testing Framework")
    
    parser.add_argument(
        "--log-file",
        help="Path to log file. If not provided, logs to stderr."
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level."
    )
    
    parser.add_argument(
        "--report-file",
        help="Path to save the JSON report. If not provided, no report is saved."
    )
    
    parser.add_argument(
        "--html-report",
        help="Path to save an HTML report. If not provided, no HTML report is generated."
    )
    
    parser.add_argument(
        "--test-pattern",
        help="Run only tests that match this pattern."
    )
    
    parser.add_argument(
        "--test-category",
        help="Run only tests in this category."
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        help="Default timeout for all tests in seconds."
    )
    
    return parser.parse_args()


# Example functions to test
def add_numbers(x, y):
    """Simple addition function."""
    return x + y


def divide_numbers(x, y):
    """Division function that may raise ZeroDivisionError."""
    return x / y


def slow_function(seconds):
    """A slow function that sleeps for the given number of seconds."""
    time.sleep(seconds)
    return seconds


# Example test suite construction
def create_example_test_suite():
    """Create an example test suite with some basic tests."""
    suite = TestSuite("Example Test Suite")
    
    # Basic test cases
    suite.add_test(TestCase(
        function=add_numbers,
        inputs=(2, 3),
        expected_output=5,
        test_name="Addition Test"
    ))
    
    suite.add_test(TestCase(
        function=add_numbers,
        inputs=(-1, 1),
        expected_output=0,
        test_name="Zero Sum Test"
    ))
    
    # Test with expected exception
    suite.add_test(TestCase(
        function=divide_numbers,
        inputs=(1, 0),
        expected_exception=ZeroDivisionError,
        test_name="Division by Zero Test"
    ))
    
    # Test with timeout
    suite.add_test(TestCase(
        function=slow_function,
        inputs=(0.1,),
        expected_output=0.1,
        timeout=1.0,
        test_name="Slow Function Test (should pass)"
    ))
    
    suite.add_test(TestCase(
        function=slow_function,
        inputs=(2.0,),
        timeout=1.0,
        test_name="Slow Function Test (should timeout)"
    ))
    
    # Test with approximate comparison
    suite.add_test(TestCase(
        function=lambda x: x * 0.3333,
        inputs=(3,),
        expected_output=1.0,
        comparison_method=ComparisonMethod.APPROXIMATE,
        comparison_tolerance=0.001,
        test_name="Approximate Comparison Test"
    ))
    
    return suite


if __name__ == "__main__":
    args = parse_args()
    
    # Create test runner with command line settings
    runner = TestRunner(
        log_file=args.log_file or "function_test_results.log",
        log_level=getattr(logging, args.log_level),
        report_file=args.report_file or "function_test_results.json"
    )
    
    # Add example test suite
    runner.add_suite(create_example_test_suite())
    
    # Run all tests
    results = runner.run()
    
    # Generate HTML report if requested
    if args.html_report:
        runner.generate_html_report(args.html_report)
    
    # Print summary to console
    total = len(results)
    passed = sum(1 for r in results if r.status == TestResult.PASS)
    failed = sum(1 for r in results if r.status == TestResult.FAIL)
    errors = sum(1 for r in results if r.status == TestResult.ERROR)
    timeouts = sum(1 for r in results if r.status == TestResult.TIMEOUT)
    skipped = sum(1 for r in results if r.status == TestResult.SKIPPED)
    
    print("\n===== FUNCTION TEST RESULTS =====")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Timeouts: {timeouts}")
    print(f"Skipped: {skipped}")
    print(f"Pass rate: {passed/total*100:.2f}%")
    print("=================================")
    
    print(f"\nDetailed logs: {args.log_file or 'function_test_results.log'}")
    print(f"JSON report: {args.report_file or 'function_test_results.json'}")
    
    if args.html_report:
        print(f"HTML report: {args.html_report}")
