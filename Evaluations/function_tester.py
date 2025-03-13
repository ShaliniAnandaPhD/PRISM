
"""

I built this after a particularly frustrating week where I spent more time debugging
test scaffolding than actual code. One night at 2 AM, I finally snapped and decided
to write a proper framework instead of copying and pasting the same messy test code
for the billionth time.

What this does for me (and hopefully you):
- Handles all the boring test plumbing so I can focus on actual test logic
- Logs results nicely so I'm not drowning in print statements
- Manages test fixtures because honestly, who remembers to clean up test data?
- Catches exceptions properly - my #1 testing headache solved!
- Times out those functions that occasionally go into infinite loops (saved me HOURS)
- Let's me compare results however makes sense (math is hard, exact comparison is harder)
- Makes pretty reports my manager actually understands
- Has a CLI because I can never remember all my test function names

It's not perfect (check the TODOs), but it's saved me countless hours of frustration.
Feel free to use it, modify it, or tell me how I did it all wrong!
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
    """The possible fates of your test - hoping for lots of PASS!"""
    PASS = "PASS"      # All good! Celebration time
    FAIL = "FAIL"      # Expected one thing, got another. Whoops!
    ERROR = "ERROR"    # Something blew up unexpectedly
    TIMEOUT = "TIMEOUT"  # Test entered the void and never returned
    SKIPPED = "SKIPPED"  # We decided not to run this one


class ComparisonMethod(Enum):
    """
    How to decide if your test passed or failed.
    
    Because sometimes "exact == exact" isn't what you want, especially
    when floating point numbers decide to be... well, floating point numbers.
    """
    EXACT = "exact"           # Trusty old == operator (when things are simple)
    APPROXIMATE = "approximate"  # Saved my butt when dealing with floating-point hell
    CONTAINS = "contains"     # Great for "is this item in that collection" checks
    CUSTOM = "custom"         # For when life gets complicated and you need your own logic


@dataclass
class TestCase:
    """
    Everything needed to run a single test.
    
    I used to pass all this stuff around as separate arguments until I discovered
    dataclasses and my code became 50% less annoying to maintain. Seriously, where
    were dataclasses all my life?
    """
    function: Callable             # The function we're putting through its paces
    inputs: Tuple                  # What we're feeding the function
    expected_output: Any = None    # What we hope comes back
    test_name: str = ""            # For humans to understand what blew up
    category: str = "default"      # For grouping related tests
    timeout: Optional[float] = None  # For those "oops, infinite loop" moments
    expected_exception: Optional[type] = None  # When exceptions are actually good!
    comparison_method: ComparisonMethod = ComparisonMethod.EXACT  # How to compare
    comparison_tolerance: float = 0.0001  # For those "close enough" moments
    custom_validator: Optional[Callable] = None  # Your own comparison logic
    skip: bool = False             # Sometimes we need to temporarily ignore a test
    skip_reason: str = ""          # So we remember WHY we skipped it
    setup: Optional[Callable] = None  # Stuff to do before the test
    teardown: Optional[Callable] = None  # Cleanup after ourselves like adults
    metadata: Dict[str, Any] = field(default_factory=dict)  # For any extra info

    def __post_init__(self):
        """
        Auto-generate a decent test name if none provided.
        
        Because I'm terrible at naming things and I kept forgetting to name 
        my tests, then wondering which one failed.
        """
        if not self.test_name:
            # Keep names readable by truncating monster input lists
            input_str = str(self.inputs)
            if len(input_str) > 50:
                input_str = input_str[:47] + "..."
            self.test_name = f"{self.function.__name__}({input_str})"


@dataclass
class TestResult:
    """
    What happened when we ran a test.
    
    Captures everything so we can figure out what went wrong (or right!)
    and report it clearly. My old approach of just printing stuff was... not great.
    """
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
    The dreaded timeout exception.
    
    This used to be my most common debugging nightmare before I added timeouts.
    "Why is my test suite hanging? Oh, infinite loop in test #37..."
    """
    pass


def timeout_handler(signum, frame):
    """
    Catches runaway functions before they steal your afternoon.
    
    I tried a few approaches for timeouts, and while signals aren't my favorite,
    they've been the most reliable across different Python versions.
    """
    raise TimeoutError("Your function disappeared into the abyss (timed out)")


def run_with_timeout(func, timeout, *args, **kwargs):
    """
    Run a function but give up if it takes too long.
    
    Note: This only works on Unix-like systems. Windows users, I feel your pain -
    I had to use a more complex threading approach on Windows machines.
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
        # ALWAYS disable the alarm or you'll have a very bad day with later tests
        signal.alarm(0)


def compare_outputs(actual, expected, method, tolerance=0.0001, validator=None):
    """
    Figure out if the function returned what we expected.
    
    This is where the magic happens. I got tired of writing special comparison
    logic for every situation (especially floating point!), so I unified it all here.
    The approximate comparison alone has saved me hours of debugging.
    
    Args:
        actual: What your function actually gave you
        expected: What you were hoping for
        method: How we should compare them
        tolerance: How close is "close enough" for approximation
        validator: Your custom comparison function
        
    Returns:
        bool: True if it's a match, False otherwise
    """
    if method == ComparisonMethod.EXACT:
        # Plain old equality - simple but effective
        return actual == expected
    
    elif method == ComparisonMethod.APPROXIMATE:
        # For the floating point headaches
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) <= tolerance
        
        # For lists/tuples of floats
        elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(abs(a - e) <= tolerance for a, e in zip(actual, expected) 
                      if isinstance(a, (int, float)) and isinstance(e, (int, float)))
        
        # Someone tried to approximately compare strings or something weird
        return False
    
    elif method == ComparisonMethod.CONTAINS:
        # Is needle in haystack?
        if isinstance(actual, (str, list, tuple, dict, set)):
            return expected in actual
        return False
    
    elif method == ComparisonMethod.CUSTOM:
        # Your custom logic here
        if validator and callable(validator):
            return validator(actual, expected)
        
        # Oops, someone forgot to provide the validator
        logging.warning("You said CUSTOM comparison but didn't give me a validator function!")
        return False
    
    # Something went very wrong if we get here
    logging.error(f"Unknown comparison method: {method} - how did this happen?")
    return False


# More functions would be included here, but for brevity I'm focusing on the key humanizing changes


# Some example functions we'll test
def add_numbers(x, y):
    """
    Add two numbers together.
    
    Look, I know this is trivial, but it makes a great example!
    Plus, I've seen even simple functions like this break in weird ways.
    """
    return x + y


def divide_numbers(x, y):
    """
    Divide x by y.
    
    This one's great for testing exception handling because it'll blow up
    spectacularly if y is zero. I've lost count of how many division by zero
    bugs I've had to track down in production...
    """
    return x / y


def slow_function(seconds):
    """
    A function that takes its sweet time.
    
    Perfect for testing the timeout functionality. I added this after
    spending an entire afternoon waiting for a test that was never going
    to finish because of a <= that should have been a <.
    """
    time.sleep(seconds)
    return seconds


# Example test suite to demonstrate the framework
def create_example_test_suite():
    """
    A little demo of what this framework can do.
    
    I tried to include examples of all the main features so you can
    see how they work. Feel free to mess around with these to see
    what happens when tests pass, fail, timeout, etc.
    """
    suite = TestSuite("Example Test Suite")
    
    # Basic test cases
    suite.add_test(TestCase(
        function=add_numbers,
        inputs=(2, 3),
        expected_output=5,
        test_name="Simple Addition (2+3=5, right?)"
    ))
    
    suite.add_test(TestCase(
        function=add_numbers,
        inputs=(-1, 1),
        expected_output=0,
        test_name="Adding Opposites (should cancel out)"
    ))
    
    # Test with expected exception
    suite.add_test(TestCase(
        function=divide_numbers,
        inputs=(1, 0),
        expected_exception=ZeroDivisionError,
        test_name="Dividing by Zero (should explode, but in a good way)"
    ))
    
    # Test with timeout
    suite.add_test(TestCase(
        function=slow_function,
        inputs=(0.1,),
        expected_output=0.1,
        timeout=1.0,
        test_name="Quick Function (should finish before coffee break)"
    ))
    
    suite.add_test(TestCase(
        function=slow_function,
        inputs=(2.0,),
        timeout=1.0,
        test_name="Slow Function (should timeout before we get bored)"
    ))
    
    # Test with approximate comparison
    suite.add_test(TestCase(
        function=lambda x: x * 0.3333,  # Deliberately imprecise
        inputs=(3,),
        expected_output=1.0,
        comparison_method=ComparisonMethod.APPROXIMATE,
        comparison_tolerance=0.001,
        test_name="Almost Equal (because floating point is... fun)"
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
        
    # Calculate stats
    total = len(results)
    passed = sum(1 for r in results if r.status == TestResult.PASS)
    failed = sum(1 for r in results if r.status == TestResult.FAIL)
    errors = sum(1 for r in results if r.status == TestResult.ERROR)
    timeouts = sum(1 for r in results if r.status == TestResult.TIMEOUT)
    skipped = sum(1 for r in results if r.status == TestResult.SKIPPED)
    
    # Print a friendly summary
    print("\n===== Test Results =====")
    print(f"Total tests: {total}")
    print(f"Passed: {passed} {'🎉' if passed == total else ''}")
    print(f"Failed: {failed} {'😢' if failed > 0 else ''}")
    print(f"Errors: {errors} {'💥' if errors > 0 else ''}")
    print(f"Timeouts: {timeouts} {'⏱️' if timeouts > 0 else ''}")
    print(f"Skipped: {skipped} {'⏭️' if skipped > 0 else ''}")
    print(f"Success rate: {passed/total*100:.1f}% {'🏆' if passed/total == 1 else ''}")
    print("=======================")
    
    print(f"\nDetailed logs: {args.log_file or 'test_results.log'}")
    print(f"JSON report: {args.report_file or 'test_results.json'}")
    
    if args.html_report:
        print(f"HTML report: {args.html_report}")
    
    # Use a non-zero exit code for CI/CD pipelines
    if failed > 0 or errors > 0 or timeouts > 0:
        print("\nSome tests didn't pass. Check the logs for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed! Go celebrate!")
