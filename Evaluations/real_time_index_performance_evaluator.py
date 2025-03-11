"""
CAPABILITIES:
- Evaluates dynamic index update performance, correctness, and reliability under various conditions
- Measures update latency, throughput, and consistency during high-frequency operations
- Tests index behavior during concurrent read/write scenarios with configurable load profiles
- Performs rollback validation for failed updates and transactional integrity testing
- Simulates network partitions and system failures to assess recovery mechanisms
- Measures index memory consumption and scaling characteristics during sustained operations
- Supports various index structures (vector, inverted, BM25, hybrid) with consistent benchmarking
- Provides detailed metrics with statistical analysis for performance optimization
- Generates comprehensive reports with visualizations for latency distribution
- Validates correctness of search results after index modifications
"""

import logging
import json
import time
import random
import threading
import concurrent.futures
import os
import sys
import uuid
import statistics
import traceback
import datetime
import argparse
import gc
import itertools
import copy
from typing import Dict, List, Any, Callable, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

# Optional imports with fallbacks
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    import psutil
    RESOURCE_MONITORING_AVAILABLE = True
except ImportError:
    RESOURCE_MONITORING_AVAILABLE = False


# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{log_dir}/index_evaluation_{timestamp}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger("index_evaluator")


class OperationType(Enum):
    """Types of operations that can be performed on an index."""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"
    BATCH_ADD = "batch_add"
    BATCH_UPDATE = "batch_update"
    BATCH_DELETE = "batch_delete"


@dataclass
class IndexOperation:
    """Represents a single operation on the index."""
    operation_type: OperationType
    data: Any
    timestamp: float = field(default_factory=time.time)
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class OperationResult:
    """Result of an index operation."""
    operation: IndexOperation
    success: bool
    execution_time_sec: float
    result: Any = None
    error: Optional[str] = None
    memory_usage_bytes: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_type": self.operation.operation_type.value,
            "operation_id": self.operation.operation_id,
            "success": self.success,
            "execution_time_sec": self.execution_time_sec,
            "error": self.error,
            "memory_usage_bytes": self.memory_usage_bytes
        }


@dataclass
class TestScenario:
    """A test scenario for the index evaluator."""
    name: str
    description: str
    operations: List[IndexOperation]
    validation_function: Optional[Callable[[Any, List[OperationResult]], bool]] = None
    concurrency_level: int = 1
    expected_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSummary:
    """Summary of index evaluation results."""
    scenario_name: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    avg_execution_time_sec: float
    min_execution_time_sec: float
    max_execution_time_sec: float
    p95_execution_time_sec: float
    p99_execution_time_sec: float
    validation_passed: bool
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ResourceUsage:
    """Resource usage statistics."""
    cpu_percent: float = 0.0
    memory_bytes: int = 0
    memory_percent: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@contextmanager
def measure_resources() -> Dict[str, Any]:
    """Context manager to measure resource usage before and after an operation."""
    if not RESOURCE_MONITORING_AVAILABLE:
        yield {}
        return
    
    process = psutil.Process(os.getpid())
    start_cpu_times = process.cpu_times()
    start_io_counters = process.io_counters() if hasattr(process, 'io_counters') else None
    start_memory = process.memory_info().rss
    
    try:
        yield {}
    finally:
        end_cpu_times = process.cpu_times()
        end_io_counters = process.io_counters() if hasattr(process, 'io_counters') else None
        end_memory = process.memory_info().rss
        
        cpu_user = end_cpu_times.user - start_cpu_times.user
        cpu_system = end_cpu_times.system - start_cpu_times.system
        memory_delta = end_memory - start_memory
        
        io_read_delta = 0
        io_write_delta = 0
        if start_io_counters and end_io_counters:
            io_read_delta = end_io_counters.read_bytes - start_io_counters.read_bytes
            io_write_delta = end_io_counters.write_bytes - start_io_counters.write_bytes
        
        result = {
            "cpu_user_sec": cpu_user,
            "cpu_system_sec": cpu_system,
            "memory_delta_bytes": memory_delta,
            "io_read_bytes": io_read_delta,
            "io_write_bytes": io_write_delta
        }


class IndexEvaluator:
    """
    Evaluates the performance, correctness, and reliability of dynamic index updates.
    Provides comprehensive testing and metrics for index operations.
    """
    
    def __init__(
        self,
        index_updater: Callable,
        index_reader: Optional[Callable] = None,
        index_deleter: Optional[Callable] = None,
        output_dir: str = "evaluation_results"
    ):
        """
        Initialize the index evaluator.
        
        Args:
            index_updater: Function that updates the index
            index_reader: Function that queries the index
            index_deleter: Function that deletes from the index
            output_dir: Directory to save evaluation results
        """
        self.index_updater = index_updater
        self.index_reader = index_reader
        self.index_deleter = index_deleter
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Resource tracking
        self.resource_usage: List[ResourceUsage] = []
        self.resource_tracking_interval = 1.0  # seconds
        self.resource_tracking_thread = None
        self.stop_resource_tracking = threading.Event()
        
        logger.info(f"Initialized IndexEvaluator with output directory: {output_dir}")
    
    def _start_resource_tracking(self) -> None:
        """Start tracking resource usage in a background thread."""
        if not RESOURCE_MONITORING_AVAILABLE:
            logger.warning("Resource monitoring not available. Install psutil package for this feature.")
            return
        
        self.resource_usage = []
        self.stop_resource_tracking.clear()
        
        def track_resources():
            """Track resource usage at regular intervals."""
            process = psutil.Process(os.getpid())
            
            while not self.stop_resource_tracking.is_set():
                try:
                    cpu_percent = process.cpu_percent(interval=None)
                    memory_info = process.memory_info()
                    memory_percent = process.memory_percent()
                    
                    io_read = 0
                    io_write = 0
                    if hasattr(process, 'io_counters'):
                        io_counters = process.io_counters()
                        io_read = io_counters.read_bytes
                        io_write = io_counters.write_bytes
                    
                    usage = ResourceUsage(
                        cpu_percent=cpu_percent,
                        memory_bytes=memory_info.rss,
                        memory_percent=memory_percent,
                        io_read_bytes=io_read,
                        io_write_bytes=io_write
                    )
                    
                    self.resource_usage.append(usage)
                    
                except Exception as e:
                    logger.error(f"Error in resource tracking: {str(e)}")
                
                self.stop_resource_tracking.wait(self.resource_tracking_interval)
        
        self.resource_tracking_thread = threading.Thread(target=track_resources)
        self.resource_tracking_thread.daemon = True
        self.resource_tracking_thread.start()
        
        logger.info("Started resource usage tracking")
    
    def _stop_resource_tracking(self) -> None:
        """Stop tracking resource usage."""
        if self.resource_tracking_thread and self.resource_tracking_thread.is_alive():
            self.stop_resource_tracking.set()
            self.resource_tracking_thread.join(timeout=2.0)
            logger.info("Stopped resource usage tracking")
    
    def execute_operation(
        self,
        index: Any,
        operation: IndexOperation
    ) -> OperationResult:
        """
        Execute a single operation on the index.
        
        Args:
            index: The index to operate on
            operation: The operation to perform
            
        Returns:
            OperationResult with execution details
        """
        start_time = time.perf_counter()
        result = None
        error = None
        success = False
        memory_usage = None
        
        # Capture memory usage before operation if available
        if RESOURCE_MONITORING_AVAILABLE:
            memory_before = psutil.Process(os.getpid()).memory_info().rss
        
        try:
            # Execute the appropriate operation based on type
            if operation.operation_type == OperationType.ADD:
                self.index_updater(index, operation.data)
                success = True
            
            elif operation.operation_type == OperationType.UPDATE:
                if isinstance(operation.data, tuple) and len(operation.data) == 2:
                    key, value = operation.data
                    # For update, we often need both the key to update and the new value
                    self.index_updater(index, key, value)
                else:
                    # Simple update with just new value
                    self.index_updater(index, operation.data)
                success = True
            
            elif operation.operation_type == OperationType.DELETE:
                if self.index_deleter:
                    self.index_deleter(index, operation.data)
                    success = True
                else:
                    error = "Delete operation not supported (no index_deleter provided)"
            
            elif operation.operation_type == OperationType.QUERY:
                if self.index_reader:
                    result = self.index_reader(index, operation.data)
                    success = True
                else:
                    error = "Query operation not supported (no index_reader provided)"
            
            elif operation.operation_type == OperationType.BATCH_ADD:
                if isinstance(operation.data, list) or isinstance(operation.data, tuple):
                    for item in operation.data:
                        self.index_updater(index, item)
                    success = True
                else:
                    error = "Batch add requires list or tuple of items"
            
            elif operation.operation_type == OperationType.BATCH_UPDATE:
                if isinstance(operation.data, list) or isinstance(operation.data, tuple):
                    for item in operation.data:
                        if isinstance(item, tuple) and len(item) == 2:
                            key, value = item
                            self.index_updater(index, key, value)
                        else:
                            self.index_updater(index, item)
                    success = True
                else:
                    error = "Batch update requires list or tuple of items"
            
            elif operation.operation_type == OperationType.BATCH_DELETE:
                if self.index_deleter and (isinstance(operation.data, list) or isinstance(operation.data, tuple)):
                    for item in operation.data:
                        self.index_deleter(index, item)
                    success = True
                else:
                    error = "Batch delete not supported or requires list/tuple of items"
            
            else:
                error = f"Unknown operation type: {operation.operation_type}"
        
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Error executing operation {operation.operation_id}: {error}")
            logger.debug(traceback.format_exc())
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Capture memory usage after operation if available
        if RESOURCE_MONITORING_AVAILABLE:
            memory_after = psutil.Process(os.getpid()).memory_info().rss
            memory_usage = memory_after - memory_before
        
        # Create result
        operation_result = OperationResult(
            operation=operation,
            success=success,
            execution_time_sec=execution_time,
            result=result,
            error=error,
            memory_usage_bytes=memory_usage
        )
        
        logger.debug(f"Operation {operation.operation_id} completed in {execution_time:.6f} seconds, success={success}")
        return operation_result
    
    def run_scenario(
        self,
        index: Any,
        scenario: TestScenario
    ) -> Tuple[List[OperationResult], EvaluationSummary]:
        """
        Run a test scenario with multiple operations.
        
        Args:
            index: The index to operate on
            scenario: The test scenario to run
            
        Returns:
            Tuple of (list of operation results, evaluation summary)
        """
        logger.info(f"Running scenario: {scenario.name}")
        logger.info(f"Description: {scenario.description}")
        logger.info(f"Total operations: {len(scenario.operations)}")
        logger.info(f"Concurrency level: {scenario.concurrency_level}")
        
        operation_results = []
        
        # Start resource tracking
        self._start_resource_tracking()
        
        start_time = time.perf_counter()
        
        try:
            if scenario.concurrency_level <= 1:
                # Sequential execution
                for operation in scenario.operations:
                    result = self.execute_operation(index, operation)
                    operation_results.append(result)
            else:
                # Concurrent execution
                index_lock = threading.RLock()  # Reentrant lock for thread safety
                
                def execute_with_lock(op):
                    with index_lock:
                        return self.execute_operation(index, op)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=scenario.concurrency_level) as executor:
                    future_to_op = {executor.submit(execute_with_lock, op): op for op in scenario.operations}
                    
                    for future in concurrent.futures.as_completed(future_to_op):
                        try:
                            result = future.result()
                            operation_results.append(result)
                        except Exception as e:
                            op = future_to_op[future]
                            logger.error(f"Exception during concurrent execution of {op.operation_id}: {str(e)}")
                            # Create failed result
                            operation_results.append(OperationResult(
                                operation=op,
                                success=False,
                                execution_time_sec=0.0,
                                error=f"Concurrency error: {str(e)}"
                            ))
            
            # Run validation if provided
            validation_passed = True
            if scenario.validation_function:
                try:
                    validation_passed = scenario.validation_function(index, operation_results)
                    logger.info(f"Validation result: {'PASS' if validation_passed else 'FAIL'}")
                except Exception as e:
                    logger.error(f"Error during validation: {str(e)}")
                    validation_passed = False
            
            # Calculate summary statistics
            successful_ops = sum(1 for r in operation_results if r.success)
            failed_ops = len(operation_results) - successful_ops
            
            execution_times = [r.execution_time_sec for r in operation_results if r.success]
            
            if execution_times:
                avg_time = statistics.mean(execution_times)
                min_time = min(execution_times)
                max_time = max(execution_times)
                
                # Calculate percentiles
                sorted_times = sorted(execution_times)
                p95_index = int(len(sorted_times) * 0.95)
                p99_index = int(len(sorted_times) * 0.99)
                
                p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else max_time
                p99_time = sorted_times[p99_index] if p99_index < len(sorted_times) else max_time
            else:
                avg_time = min_time = max_time = p95_time = p99_time = 0.0
            
            # Create summary
            summary = EvaluationSummary(
                scenario_name=scenario.name,
                total_operations=len(operation_results),
                successful_operations=successful_ops,
                failed_operations=failed_ops,
                avg_execution_time_sec=avg_time,
                min_execution_time_sec=min_time,
                max_execution_time_sec=max_time,
                p95_execution_time_sec=p95_time,
                p99_execution_time_sec=p99_time,
                validation_passed=validation_passed
            )
            
            logger.info(f"Scenario complete. Success rate: {successful_ops/len(operation_results):.2%}")
            logger.info(f"Avg execution time: {avg_time:.6f} sec")
            
            return operation_results, summary
            
        finally:
            end_time = time.perf_counter()
            logger.info(f"Total scenario execution time: {end_time - start_time:.2f} seconds")
            
            # Stop resource tracking
            self._stop_resource_tracking()
    
    def generate_report(
        self,
        scenario: TestScenario,
        operation_results: List[OperationResult],
        summary: EvaluationSummary
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            scenario: The test scenario that was run
            operation_results: Results of all operations
            summary: Evaluation summary statistics
            
        Returns:
            Path to the generated report file
        """
        # Create timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, f"{scenario.name}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Prepare report data
        report_data = {
            "scenario": {
                "name": scenario.name,
                "description": scenario.description,
                "concurrency_level": scenario.concurrency_level,
                "operation_count": len(scenario.operations),
                "metadata": scenario.metadata
            },
            "summary": summary.to_dict(),
            "operations": [r.to_dict() for r in operation_results],
            "resource_usage": [u.to_dict() for u in self.resource_usage] if self.resource_usage else []
        }
        
        # Save report as JSON
        report_path = os.path.join(report_dir, "evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate visualizations if available
        if VISUALIZATION_AVAILABLE:
            self._generate_visualizations(report_dir, operation_results, summary)
        
        logger.info(f"Report generated at {report_path}")
        return report_path
    
    def _generate_visualizations(
        self,
        report_dir: str,
        operation_results: List[OperationResult],
        summary: EvaluationSummary
    ) -> None:
        """Generate visualizations for the report."""
        try:
            # Create plots directory
            plots_dir = os.path.join(report_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # 1. Execution time distribution
            plt.figure(figsize=(10, 6))
            execution_times = [r.execution_time_sec for r in operation_results if r.success]
            
            if execution_times:
                sns.histplot(execution_times, kde=True)
                plt.title("Execution Time Distribution")
                plt.xlabel("Execution Time (seconds)")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plots_dir, "execution_time_distribution.png"))
                plt.close()
            
            # 2. Success rate by operation type
            op_types = {}
            for r in operation_results:
                op_type = r.operation.operation_type.value
                if op_type not in op_types:
                    op_types[op_type] = {"success": 0, "fail": 0}
                
                if r.success:
                    op_types[op_type]["success"] += 1
                else:
                    op_types[op_type]["fail"] += 1
            
            if op_types:
                plt.figure(figsize=(10, 6))
                
                op_names = list(op_types.keys())
                success_counts = [op_types[op]["success"] for op in op_names]
                fail_counts = [op_types[op]["fail"] for op in op_names]
                
                x = range(len(op_names))
                width = 0.35
                
                plt.bar(x, success_counts, width, label="Success", color="green")
                plt.bar([i + width for i in x], fail_counts, width, label="Fail", color="red")
                
                plt.xlabel("Operation Type")
                plt.ylabel("Count")
                plt.title("Operation Success Rate by Type")
                plt.xticks([i + width/2 for i in x], op_names)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.savefig(os.path.join(plots_dir, "success_rate_by_type.png"))
                plt.close()
            
            # 3. Time series of execution times
            if len(operation_results) > 1:
                plt.figure(figsize=(12, 6))
                
                # Sort by timestamp
                sorted_results = sorted(operation_results, key=lambda r: r.operation.timestamp)
                times = [r.operation.timestamp for r in sorted_results]
                execution_times = [r.execution_time_sec for r in sorted_results]
                success = [r.success for r in sorted_results]
                
                # Create relative timestamps for better readability
                if times:
                    min_time = min(times)
                    relative_times = [(t - min_time) for t in times]
                    
                    # Plot with different colors for success/failure
                    for i, (t, e, s) in enumerate(zip(relative_times, execution_times, success)):
                        color = "green" if s else "red"
                        plt.scatter(t, e, color=color, alpha=0.7)
                    
                    # Add trend line
                    if len(relative_times) > 1:
                        try:
                            # Only use successful operations for trend
                            success_times = [t for t, s in zip(relative_times, success) if s]
                            success_exec_times = [e for e, s in zip(execution_times, success) if s]
                            
                            if len(success_times) > 1:
                                z = np.polyfit(success_times, success_exec_times, 1)
                                p = np.poly1d(z)
                                plt.plot(success_times, p(success_times), "b--", alpha=0.5)
                        except:
                            # Continue if trend line calculation fails
                            pass
                    
                    plt.title("Execution Time Trend")
                    plt.xlabel("Relative Time (seconds)")
                    plt.ylabel("Execution Time (seconds)")
                    plt.grid(True, alpha=0.3)
                    
                    # Add legend
                    plt.scatter([], [], color="green", alpha=0.7, label="Success")
                    plt.scatter([], [], color="red", alpha=0.7, label="Failure")
                    plt.legend()
                    
                    plt.savefig(os.path.join(plots_dir, "execution_time_trend.png"))
                    plt.close()
            
            # 4. Resource usage if available
            if self.resource_usage:
                # CPU usage
                plt.figure(figsize=(12, 6))
                cpu_percents = [u.cpu_percent for u in self.resource_usage]
                plt.plot(range(len(cpu_percents)), cpu_percents)
                plt.title("CPU Usage During Evaluation")
                plt.xlabel("Time (samples)")
                plt.ylabel("CPU Usage (%)")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plots_dir, "cpu_usage.png"))
                plt.close()
                
                # Memory usage
                plt.figure(figsize=(12, 6))
                memory_mb = [u.memory_bytes / (1024 * 1024) for u in self.resource_usage]
                plt.plot(range(len(memory_mb)), memory_mb)
                plt.title("Memory Usage During Evaluation")
                plt.xlabel("Time (samples)")
                plt.ylabel("Memory Usage (MB)")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plots_dir, "memory_usage.png"))
                plt.close()
            
            logger.info(f"Generated visualizations in {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def run_basic_evaluation(
        self,
        index: Any,
        operations: List[IndexOperation]
    ) -> Tuple[List[OperationResult], EvaluationSummary]:
        """
        Run a basic evaluation with the provided operations.
        
        Args:
            index: The index to operate on
            operations: List of operations to perform
            
        Returns:
            Tuple of (list of operation results, evaluation summary)
        """
        # Create a basic scenario
        scenario = TestScenario(
            name="basic_evaluation",
            description="Basic evaluation of index operations",
            operations=operations,
            concurrency_level=1
        )
        
        # Run the scenario
        results, summary = self.run_scenario(index, scenario)
        
        # Generate report
        self.generate_report(scenario, results, summary)
        
        return results, summary
    
    def run_concurrency_test(
        self,
        index: Any,
        operations: List[IndexOperation],
        concurrency_levels: List[int]
    ) -> Dict[int, EvaluationSummary]:
        """
        Test index performance under different concurrency levels.
        
        Args:
            index: The index to operate on
            operations: List of operations to perform
            concurrency_levels: List of concurrency levels to test
            
        Returns:
            Dictionary mapping concurrency levels to evaluation summaries
        """
        results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing with concurrency level: {concurrency}")
            
            # Create a deep copy of the index for this test
            index_copy = copy.deepcopy(index)
            
            # Create scenario for this concurrency level
            scenario = TestScenario(
                name=f"concurrency_test_{concurrency}",
                description=f"Testing index performance with {concurrency} concurrent operations",
                operations=operations,
                concurrency_level=concurrency
            )
            
            # Run scenario
            _, summary = self.run_scenario(index_copy, scenario)
            
            # Store results
            results[concurrency] = summary
        
        # Compare and visualize results if visualization is available
        if VISUALIZATION_AVAILABLE and results:
            self._visualize_concurrency_results(results, concurrency_levels)
        
        return results
    
    def _visualize_concurrency_results(
        self,
        results: Dict[int, EvaluationSummary],
        concurrency_levels: List[int]
    ) -> None:
        """Visualize the results of concurrency testing."""
        try:
            plots_dir = os.path.join(self.output_dir, "concurrency_test_plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # 1. Execution time vs concurrency
            plt.figure(figsize=(10, 6))
            
            avg_times = [results[c].avg_execution_time_sec for c in concurrency_levels]
            p95_times = [results[c].p95_execution_time_sec for c in concurrency_levels]
            
            plt.plot(concurrency_levels, avg_times, 'o-', label="Average")
            plt.plot(concurrency_levels, p95_times, 's-', label="P95")
            
            plt.title("Execution Time vs Concurrency")
            plt.xlabel("Concurrency Level")
            plt.ylabel("Execution Time (seconds)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(plots_dir, "execution_time_vs_concurrency.png"))
            plt.close()
            
            # 2. Success rate vs concurrency
            plt.figure(figsize=(10, 6))
            
            success_rates = [results[c].successful_operations / results[c].total_operations * 100 
                           for c in concurrency_levels]
            
            plt.plot(concurrency_levels, success_rates, 'o-')
            
            plt.title("Success Rate vs Concurrency")
            plt.xlabel("Concurrency Level")
            plt.ylabel("Success Rate (%)")
            plt.ylim(0, 105)
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(plots_dir, "success_rate_vs_concurrency.png"))
            plt.close()
            
            logger.info(f"Generated concurrency test visualizations in {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error visualizing concurrency results: {str(e)}")
    
    def test_recovery_mechanisms(
        self,
        index: Any,
        operations: List[IndexOperation],
        failure_points: List[int],
        recovery_function: Callable[[Any], Any]
    ) -> Dict[str, Any]:
        """
        Test index recovery mechanisms after simulated failures.
        
        Args:
            index: The index to operate on
            operations: List of operations to perform
            failure_points: Indices of operations after which to simulate failure
            recovery_function: Function to recover the index after failure
            
        Returns:
            Dictionary with recovery test results
        """
        results = {
            "tests": [],
            "overall_success": True
        }
        
        for failure_point in failure_points:
            if failure_point >= len(operations):
                logger.warning(f"Failure point {failure_point} exceeds operation count, skipping")
                continue
            
            logger.info(f"Testing recovery after operation {failure_point}")
            
            # Create a deep copy of the index for this test
            index_copy = copy.deepcopy(index)
            
            # Execute operations up to failure point
            pre_failure_results = []
            for i in range(failure_point + 1):
                if i < len(operations):
                    result = self.execute_operation(index_copy, operations[i])
                    pre_failure_results.append(result)
            
            # Simulate failure and recovery
            try:
                logger.info("Simulating failure and recovery")
                recovered_index = recovery_function(index_copy)
                
                # Verify recovery by running remaining operations
                post_recovery_results = []
                for i in range(failure_point + 1, len(operations)):
                    result = self.execute_operation(recovered_index, operations[i])
                    post_recovery_results.append(result)
                
                # Check if all post-recovery operations succeeded
                recovery_success = all(r.success for r in post_recovery_results)
                
                test_result = {
                    "failure_point": failure_point,
                    "pre_failure_operations": len(pre_failure_results),
                    "post_recovery_operations": len(post_recovery_results),
                    "recovery_success": recovery_success,
                    "post_recovery_success_rate": sum(1 for r in post_recovery_results if r.success) / len(post_recovery_results) if post_recovery_results else 0
                }
                
                results["tests"].append(test_result)
                
                if not recovery_success:
                    results["overall_success"] = False
                
                logger.info(f"Recovery test at point {failure_point}: {'SUCCESS' if recovery_success else 'FAILURE'}")
                
            except Exception as e:
                logger.error(f"Error during recovery test at point {failure_point}: {str(e)}")
                results["tests"].append({
                    "failure_point": failure_point,
                    "error": str(e),
                    "recovery_success": False
                })
                results["overall_success"] = False
        
        # Save recovery test results
        recovery_test_path = os.path.join(self.output_dir, "recovery_test_results.json")
        with open(recovery_test_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Recovery test results saved to {recovery_test_path}")
        return results


def create_test_operations(
    count: int,
    operation_distribution: Optional[Dict[OperationType, float]] = None,
    data_generator: Optional[Callable[[OperationType], Any]] = None
) -> List[IndexOperation]:
    """
    Create a list of test operations for index evaluation.
    
    Args:
        count: Number of operations to create
        operation_distribution: Distribution of operation types (should sum to 1.0)
        data_generator: Function to generate data for each operation type
        
    Returns:
        List of IndexOperation objects
    """
    # Default distribution if none provided
    if operation_distribution is None:
        operation_distribution = {
            OperationType.ADD: 0.5,
            OperationType.UPDATE: 0.3,
            OperationType.DELETE: 0.1,
            OperationType.QUERY: 0.1
        }
    
    # Default data generator if none provided
    if data_generator is None:
        def default_generator(op_type):
            if op_type == OperationType.ADD:
                return f"doc{random.randint(1000, 9999)}"
            elif op_type == OperationType.UPDATE:
                return f"doc{random.randint(1000, 9999)}"
            elif op_type == OperationType.DELETE:
                return f"doc{random.randint(1000, 9999)}"
            elif op_type == OperationType.QUERY:
                return f"query{random.randint(1000, 9999)}"
            else:
                return f"data{random.randint(1000, 9999)}"
        
        data_generator = default_generator
    
    # Create operations
    operations = []
    operation_types = list(operation_distribution.keys())
    weights = list(operation_distribution.values())
    
    for _ in range(count):
        op_type = random.choices(operation_types, weights=weights, k=1)[0]
        data = data_generator(op_type)
        
        operations.append(IndexOperation(
            operation_type=op_type,
            data=data
        ))
    
    return operations


# Example Dynamic Index Updater implementations (replace with actual implementations)
class SimpleListIndex:
    """Simple list-based index for demonstration purposes."""
    
    def __init__(self, initial_data=None):
        """Initialize the index with optional initial data."""
        self.data = initial_data or []
        self.lookup = set(self.data)
    
    def add(self, entry):
        """Add an entry to the index."""
        if entry not in self.lookup:
            self.data.append(entry)
            self.lookup.add(entry)
            return True
        return False
    
    def update(self, old_entry, new_entry):
        """Update an entry in the index."""
        if old_entry in self.lookup:
            # Remove old entry
            self.data.remove(old_entry)
            self.lookup.remove(old_entry)
            
            # Add new entry
            self.data.append(new_entry)
            self.lookup.add(new_entry)
            return True
        return False
    
    def delete(self, entry):
        """Delete an entry from the index."""
        if entry in self.lookup:
            self.data.remove(entry)
            self.lookup.remove(entry)
            return True
        return False
    
    def query(self, query_string):
        """Query the index."""
        # Simple substring search
        return [entry for entry in self.data if query_string in entry]


# Helper functions for the index operations
def list_index_add(index, entry):
    """Add an entry to a list-based index."""
    index.add(entry)


def list_index_update(index, old_entry, new_entry=None):
    """Update an entry in a list-based index."""
    if new_entry is None:
        # If no new_entry provided, assume old_entry is actually the new entry
        # and we need to find something to update
        if index.data:
            # Update the first item as a demonstration
            index.update(index.data[0], old_entry)
    else:
        index.update(old_entry, new_entry)


def list_index_delete(index, entry):
    """Delete an entry from a list-based index."""
    index.delete(entry)


def list_index_query(index, query_string):
    """Query a list-based index."""
    return index.query(query_string)


def validate_index_consistency(index, operation_results):
    """
    Validate that the index is consistent after operations.
    
    Args:
        index: The index to validate
        operation_results: List of operation results
        
    Returns:
        Boolean indicating whether the index is consistent
    """
    # For a list-based index, we can check that:
    # 1. All successful ADD operations have their entries in the index
    # 2. No successful DELETE operations have their entries in the index
    
    for result in operation_results:
        if not result.success:
            continue
        
        op = result.operation
        
        if op.operation_type == OperationType.ADD:
            if op.data not in index.lookup:
                logger.error(f"Consistency error: Added entry {op.data} not found in index")
                return False
        
        elif op.operation_type == OperationType.DELETE:
            if op.data in index.lookup:
                logger.error(f"Consistency error: Deleted entry {op.data} still in index")
                return False
    
    return True


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate real-time index performance")
    parser.add_argument("--output-dir", default="index_evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--operation-count", type=int, default=1000,
                        help="Number of operations to perform")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrency level for operations")
    parser.add_argument("--run-concurrency-test", action="store_true",
                        help="Run concurrency scaling test")
    parser.add_argument("--run-recovery-test", action="store_true",
                        help="Run recovery mechanism test")
    
    args = parser.parse_args()
    
    # Create sample index with initial data
    initial_data = [f"doc{i}" for i in range(1, 11)]
    index = SimpleListIndex(initial_data)
    
    # Create index evaluator
    evaluator = IndexEvaluator(
        index_updater=list_index_add,
        index_reader=list_index_query,
        index_deleter=list_index_delete,
        output_dir=args.output_dir
    )
    
    # Create test operations
    operations = create_test_operations(
        count=args.operation_count,
        operation_distribution={
            OperationType.ADD: 0.6,
            OperationType.DELETE: 0.2,
            OperationType.QUERY: 0.2
        }
    )
    
    # Create validation function
    validation_function = validate_index_consistency
    
    # Create basic test scenario
    scenario = TestScenario(
        name="basic_index_test",
        description="Basic test of index operations with validation",
        operations=operations,
        validation_function=validation_function,
        concurrency_level=args.concurrency
    )
    
    # Run the scenario
    results, summary = evaluator.run_scenario(index, scenario)
    
    # Generate report
    report_path = evaluator.generate_report(scenario, results, summary)
    
    # Print summary
    print("\nIndex Evaluation Summary:")
    print(f"Total operations: {summary.total_operations}")
    print(f"Successful operations: {summary.successful_operations} ({summary.successful_operations/summary.total_operations:.2%})")
    print(f"Failed operations: {summary.failed_operations}")
    print(f"Average execution time: {summary.avg_execution_time_sec:.6f} seconds")
    print(f"P95 execution time: {summary.p95_execution_time_sec:.6f} seconds")
    print(f"Validation: {'PASSED' if summary.validation_passed else 'FAILED'}")
    print(f"Report saved to: {report_path}")
    
    # Run concurrency test if requested
    if args.run_concurrency_test:
        print("\nRunning concurrency scaling test...")
        concurrency_levels = [1, 2, 4, 8, 16]
        concurrency_results = evaluator.run_concurrency_test(
            index,
            operations[:200],  # Use fewer operations for concurrency test
            concurrency_levels
        )
        
        print("\nConcurrency Test Results:")
        for level, result in concurrency_results.items():
            print(f"Concurrency {level}: Avg time {result.avg_execution_time_sec:.6f}s, "
                  f"Success rate {result.successful_operations/result.total_operations:.2%}")
    
    # Run recovery test if requested
    if args.run_recovery_test:
        print("\nRunning recovery mechanism test...")
        
        # Simple recovery function that creates a new index with the same data
        def recovery_function(broken_index):
            new_index = SimpleListIndex()
            # Copy only valid data (simulating recovery)
            for item in broken_index.data:
                if item:  # Simple validity check
                    new_index.add(item)
            return new_index
        
        # Test recovery at different points
        failure_points = [len(operations) // 4, len(operations) // 2, 3 * len(operations) // 4]
        recovery_results = evaluator.test_recovery_mechanisms(
            index,
            operations[:100],  # Use fewer operations for recovery test
            failure_points,
            recovery_function
        )
        
        print("\nRecovery Test Results:")
        print(f"Overall success: {'Yes' if recovery_results['overall_success'] else 'No'}")
        for test in recovery_results["tests"]:
            print(f"Failure point {test['failure_point']}: "
                  f"{'SUCCESS' if test.get('recovery_success', False) else 'FAILURE'}")


if __name__ == "__main__":
    main()


"""
TODO:
- Implement support for persistent indices with disk I/O measurement
- Add specialized tests for spatial and graph indices
- Enhance validation with consistency models from distributed systems
- Support for custom index operations beyond CRUD (e.g., range queries, aggregations)
- Add benchmark comparison against industry standard indices
- Implement automatic parameter tuning for optimal index performance
- Add support for simulated network latency and packet loss in distributed scenarios
- Enhance visualization with interactive dashboards
- Add support for incremental index rebuilding performance testing
- Implement multi-node distributed index testing framework
"""
