"""
I built this framework after getting tired of writing the same test boilerplate
over and over again. It helps me validate Python function behavior with minimal
setup and handles all the boring stuff automatically.

Key features:
- Logs test results so I don't have to print debug statements everywhere
- Handles test fixtures so I don't need to repeat setup/teardown code
- Catches expected exceptions (super useful for error case testing)
- Deals with timeouts for those functions that occasionally hang
- Lets me choose how to compare results (exact, approximate, custom)
- Generates reports I can share with the team
- Simple CLI so I can run specific test categories

Feel free to use, modify, and improve it!
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
    """Possible test outcomes - kept simple for clarity."""
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"


class ComparisonMethod(Enum):
    """Ways to compare test results - not everything is a simple equality check."""
    EXACT = "exact"         # Standard == operator, good for most cases
    APPROXIMATE = "approximate"  # For floating-point, super useful to avoid rounding issues
    CONTAINS = "contains"   # For checking if an item is in a collection
    CUSTOM = "custom"       # When things get weird and you need custom logic


@dataclass
class TestCase:
    """
    Container for a single test case.
    
    I've found dataclasses perfect for this - less boilerplate than a full class
    but more structure than a dict.
    """
    function: Callable             # The function to test
    inputs: Tuple                  # Arguments to pass to the function
    expected_output: Any = None    # What we expect back (None if testing exceptions)
    test_name: str = ""            # Human-readable name (auto-generated if blank)
    category: str = "default"      # For organizing tests
    timeout: Optional[float] = None  # In seconds, None means no timeout
    expected_exception: Optional[type] = None  # Exception we expect to be raised
    comparison_method: ComparisonMethod = ComparisonMethod.EXACT  # How to compare results
    comparison_tolerance: float = 0.0001  # For approximate comparisons
    custom_validator: Optional[Callable] = None  # For custom comparisons
    skip: bool = False             # Set to True to skip this test
    skip_reason: str = ""          # Why the test is skipped
    setup: Optional[Callable] = None  # Function to run before the test
    teardown: Optional[Callable] = None  # Function to run after the test
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra info about the test

    def __post_init__(self):
        """Set a default test name if not provided - saves typing for simple tests."""
        if not self.test_name:
            # Truncate long input lists for readability
            input_str = str(self.inputs)
            if len(input_str) > 50:
                input_str = input_str[:47] + "..."
            self.test_name = f"{self.function.__name__}({input_str})"


@dataclass
class TestResult:
    """Results of a test execution - captures everything we need for reporting."""
    test_case: TestCase
    status: TestResult
    actual_output: Any = None
    execution_time: float = 0.0
    exception: Optional[Exception] = None
    exception_traceback: str = ""
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeoutError(Exception):
    """
    Raised when a test takes too long.
    
    I've had too many tests hang indefinitely, so timeouts were a must-have feature.
    """
    pass


def timeout_handler(signum, frame):
    """
    Signal handler for function timeout.
    
    Signal handling isn't the most elegant solution, but it works reliably across platforms.
    Just make sure you don't use SIGALRM for anything else in your code.
    """
    raise TimeoutError("Function took too long to complete")


def run_with_timeout(func, timeout, *args, **kwargs):
    """
    Run a function with a timeout to prevent hanging tests.
    
    Note: This only works on Unix-like systems due to signal.SIGALRM.
    For Windows, we'd need a different approach (maybe threading).
    """
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
        # ALWAYS disable the alarm to avoid affecting other tests
        signal.alarm(0)


def compare_outputs(actual, expected, method, tolerance=0.0001, validator=None):
    """
    Compare actual and expected outputs using the selected method.
    
    I've found these four comparison types handle 99% of my use cases:
    - Exact: Basic equality (==)
    - Approximate: Floating-point with tolerance (lifesaver for numerical code)
    - Contains: For checking substrings, list items, etc.
    - Custom: For everything else (complex objects, special logic)
    
    Args:
        actual: What your function returned
        expected: What you expected it to return
        method: How to compare them (from ComparisonMethod enum)
        tolerance: How close is close enough (for approximate comparisons)
        validator: Your custom comparison function (for custom method)
        
    Returns:
        bool: True if comparison passes, False otherwise
    """
    if method == ComparisonMethod.EXACT:
        return actual == expected
    
    elif method == ComparisonMethod.APPROXIMATE:
        # Handle numbers
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) <= tolerance
        
        # Handle lists/tuples of numbers
        elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(abs(a - e) <= tolerance for a, e in zip(actual, expected) 
                      if isinstance(a, (int, float)) and isinstance(e, (int, float)))
        
        # Can't do approximate comparison on non-numeric types
        return False
    
    elif method == ComparisonMethod.CONTAINS:
        # Check if expected is contained in actual
        if isinstance(actual, (str, list, tuple, dict, set)):
            return expected in actual
        return False
    
    elif method == ComparisonMethod.CUSTOM:
        # Use the user-provided validator
        if validator and callable(validator):
            return validator(actual, expected)
        
        # If no validator provided for CUSTOM, that's a mistake
        logging.warning("CUSTOM comparison method used but no validator provided!")
        return False
    
    # We should never get here if using the enum correctly
    logging.error(f"Unknown comparison method: {method}")
    return False


def run_test_case(test_case):
    """
    Run a single test case and return the results.
    
    This handles all the messy details like timeouts, exceptions, setup/teardown,
    and result comparisons.
    """
    # Handle skipped tests - don't even try to run them
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
            # If setup fails, the test can't run
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
                # Handle timeout case
                return TestResult(
                    test_case=test_case,
                    status=TestResult.TIMEOUT,
                    execution_time=time.time() - start_time,
                    metadata={"timeout_value": test_case.timeout}
                )
        else:
            # Run without timeout
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
                    "failure_reason": f"Expected {test_case.expected_exception.__name__} but no exception was raised"
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
        
        # If this is an expected exception, it's actually a pass!
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
        # Always run teardown if provided, even if the test failed
        if test_case.teardown and callable(test_case.teardown):
            try:
                test_case.teardown()
            except Exception as e:
                # Log teardown failures but don't fail the test because of them
                logging.warning(f"Teardown for '{test_case.test_name}' failed: {e}")


class TestSuite:
    """
    A collection of related test cases.
    
    I like to organize tests by feature or function, makes it easier to run
    subsets of tests during development.
    """
    
    def __init__(self, name="Default Test Suite"):
        self.name = name
        self.test_cases = []
        self.setup_function = None
        self.teardown_function = None
        
    def add_test(self, test_case):
        """Add a single test case to the suite."""
        self.test_cases.append(test_case)
        
    def add_tests(self, test_cases):
        """Add multiple test cases at once."""
        self.test_cases.extend(test_cases)
        
    def set_suite_setup(self, setup_func):
        """Set a function to run before any tests in the suite."""
        self.setup_function = setup_func
        
    def set_suite_teardown(self, teardown_func):
        """Set a function to run after all tests in the suite."""
        self.teardown_function = teardown_func
        
    def run(self):
        """Run all tests in the suite and return results."""
        results = []
        
        # Run suite setup if it exists
        if self.setup_function:
            try:
                self.setup_function()
            except Exception as e:
                logging.error(f"Suite setup failed: {e}")
                # If suite setup fails, skip all tests
                for test_case in self.test_cases:
                    results.append(TestResult(
                        test_case=test_case,
                        status=TestResult.SKIPPED,
                        metadata={"skip_reason": "Suite setup failed"}
                    ))
                return results
        
        # Run each test
        for test_case in self.test_cases:
            result = run_test_case(test_case)
            results.append(result)
            
        # Run suite teardown if it exists
        if self.teardown_function:
            try:
                self.teardown_function()
            except Exception as e:
                logging.error(f"Suite teardown failed: {e}")
                
        return results


class TestRunner:
    """
    Main class that handles test execution and reporting.
    
    I've designed this to be simple for basic use cases but flexible enough
    for more complex testing needs.
    """
    
    def __init__(self, log_file=None, log_level=logging.INFO, report_file=None):
        """
        Set up the test runner with logging and reporting options.
        
        Args:
            log_file: Where to save logs (None for console only)
            log_level: How verbose the logs should be
            report_file: Where to save the JSON report (None for no report)
        """
        self.suites = []
        self.log_file = log_file
        self.log_level = log_level
        self.report_file = report_file
        self._configure_logging()
        
    def _configure_logging(self):
        """Set up logging based on constructor parameters."""
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
        """Add an existing test suite to the runner."""
        self.suites.append(suite)
        
    def create_suite(self, name="Test Suite"):
        """Create a new test suite and add it to the runner."""
        suite = TestSuite(name)
        self.suites.append(suite)
        return suite
        
    def run(self):
        """
        Run all test suites and generate reports.
        
        Returns:
            List of all test results from all suites
        """
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
                logging.error(f"FAIL: {test_name} - Expected: {result.test_case.expected_output}, Got: {result.actual_output}")
            elif status == TestResult.ERROR:
                logging.error(f"ERROR: {test_name} - {result.exception}")
                if result.exception_traceback:
                    logging.debug(f"Traceback: {result.exception_traceback}")
            elif status == TestResult.TIMEOUT:
                logging.error(f"TIMEOUT: {test_name} - Function didn't finish within {result.test_case.timeout} seconds")
            elif status == TestResult.SKIPPED:
                logging.info(f"SKIPPED: {test_name} - {result.metadata.get('skip_reason', 'No reason provided')}")
    
    def _save_report(self, results):
        """Save a JSON report of test results."""
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
        """
        Generate an HTML report from results.
        
        This creates a nicer looking report than the raw JSON - useful for sharing
        with non-technical team members.
        
        Note: Not fully implemented yet - on my TODO list!
        """
        # TODO: Implement HTML report generation with nice formatting
        logging.info(f"HTML report would be saved to {html_file} (not implemented yet)")


def parse_args():
    """Process command-line arguments for the test runner."""
    parser = argparse.ArgumentParser(description="PyFuncTest - Function Testing Framework")
    
    parser.add_argument(
        "--log-file",
        help="Where to save logs (default: stderr)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="How verbose logs should be (default: INFO)"
    )
    
    parser.add_argument(
        "--report-file",
        help="Path to save the JSON report (default: function_test_results.json)"
    )
    
    parser.add_argument(
        "--html-report",
        help="Path to save an HTML report (default: no HTML report)"
    )
    
    parser.add_argument(
        "--test-pattern",
        help="Only run tests with names matching this pattern"
    )
    
    parser.add_argument(
        "--test-category",
        help="Only run tests in this category"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        help="Default timeout for all tests in seconds"
    )
    
    return parser.parse_args()


# Some example functions we'll test
def add_numbers(x, y):
    """Add two numbers together - tough stuff!"""
    return x + y


def divide_numbers(x, y):
    """
    Divide x by y.
    
    This will explode if y is zero, which is useful for
    testing exception handling.
    """
    return x / y


def slow_function(seconds):
    """This function deliberately takes its time."""
    time.sleep(seconds)
    return seconds


# Example test suite to demonstrate the framework
def create_example_test_suite():
    """
    Create an example test suite showing various test features.
    
    Shows different comparison methods, timeouts, exception handling, etc.
    """
    suite = TestSuite("Example Test Suite")
    
    # Basic test cases
    suite.add_test(TestCase(
        function=add_numbers,
        inputs=(2, 3),
        expected_output=5,
        test_name="Simple Addition Test"
    ))
    
    suite.add_test(TestCase(
        function=add_numbers,
        inputs=(-1, 1),
        expected_output=0,
        test_name="Addition with Negative Number"
    ))
    
    # Test with expected exception
    suite.add_test(TestCase(
        function=divide_numbers,
        inputs=(1, 0),
        expected_exception=ZeroDivisionError,
        test_name="Division by Zero (should raise exception)"
    ))
    
    # Test with timeout
    suite.add_test(TestCase(
        function=slow_function,
        inputs=(0.1,),
        expected_output=0.1,
        timeout=1.0,
        test_name="Quick Function (should complete in time)"
    ))
    
    suite.add_test(TestCase(
        function=slow_function,
        inputs=(2.0,),
        timeout=1.0,
        test_name="Slow Function (should timeout)"
    ))
    
    # Test with approximate comparison
    suite.add_test(TestCase(
        function=lambda x: x * 0.3333,  # Deliberately imprecise
        inputs=(3,),
        expected_output=1.0,
        comparison_method=ComparisonMethod.APPROXIMATE,
        comparison_tolerance=0.001,
        test_name="Floating Point Comparison (approximate)"
    ))
    
    return suite


if __name__ == "__main__":
    args = parse_args()
    
    # Create test runner with CLI settings
    runner = TestRunner(
        log_file=args.log_file or "test_results.log",
        log_level=getattr(logging, args.log_level),
        report_file=args.report_file or "test_results.json"
    )
    
    # Add our example test suite
    runner.add_suite(create_example_test_suite())
    
    # Run all the tests
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
    
    print("\n===== PyFuncTest Results =====")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Timeouts: {timeouts}")
    print(f"Skipped: {skipped}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print("=============================")
    
    print(f"\nDetailed logs: {args.log_file or 'test_results.log'}")
    print(f"JSON report: {args.report_file or 'test_results.json'}")
    
    if args.html_report:
        print(f"HTML report: {args.html_report}")
    
    # Use a non-zero exit code if any tests failed (useful for CI/CD)
    if failed > 0 or errors > 0 or timeouts > 0:
        sys.exit(1)
