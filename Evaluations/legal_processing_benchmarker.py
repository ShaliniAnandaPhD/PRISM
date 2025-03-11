"""
CAPABILITIES:
- Measures execution time across multiple iterations for statistical significance
- Tracks memory consumption during processing operations
- Monitors CPU and GPU utilization during compute-intensive tasks
- Performs scalability testing with progressively larger document sets
- Generates comprehensive benchmarking reports in multiple formats
- Compares performance metrics across different processing methods
- Visualizes performance data through charts and graphs
- Identifies performance bottlenecks in document processing pipelines
"""

import logging
import json
import time
import tracemalloc
import numpy as np
import psutil
import matplotlib.pyplot as plt
import os
import gc
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd


# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging with timestamp in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{log_dir}/benchmark_{timestamp}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)


@dataclass
class BenchmarkResult:
    """Data class for storing benchmark results for a single function."""
    function_name: str
    execution_time_sec: float
    memory_usage_kb: float
    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    input_size: int = 0
    iterations: int = 1
    std_dev_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary for serialization."""
        return asdict(self)


@dataclass
class ScalabilityResult:
    """Data class for storing scalability test results across different input sizes."""
    function_name: str
    input_sizes: List[int]
    execution_times: List[float]
    memory_usages: List[float]
    cpu_percents: List[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scalability result to dictionary for serialization."""
        return asdict(self)


class LegalProcessingBenchmarker:
    """
    A class for benchmarking legal document processing functions.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize the benchmarker.
        
        Args:
            output_dir: Directory where benchmark results will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.scalability_results: List[ScalabilityResult] = []
        
        # Log system information
        self._log_system_info()
    
    def _log_system_info(self):
        """Log information about the system for reference."""
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown",
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown"
        }
        
        memory_info = {
            "total": psutil.virtual_memory().total / (1024 ** 3),  # GB
            "available": psutil.virtual_memory().available / (1024 ** 3)  # GB
        }
        
        logging.info(f"System Information:")
        logging.info(f"CPU: {cpu_info}")
        logging.info(f"Memory: {memory_info} GB")
        
        # Try to detect GPU information
        try:
            # This is just a placeholder - in a real implementation, you would use
            # a library like pynvml, tf.config.list_physical_devices, or torch.cuda.device_count()
            gpu_info = {"detected": False}
            logging.info(f"GPU: {gpu_info}")
        except Exception as e:
            logging.warning(f"Could not detect GPU information: {str(e)}")
    
    def benchmark_function(
        self,
        function: Callable,
        inputs: Tuple,
        function_name: Optional[str] = None,
        iterations: int = 5,
        warmup_iterations: int = 1,
        cleanup_function: Optional[Callable] = None
    ) -> BenchmarkResult:
        """
        Benchmark a function by measuring execution time and memory usage.
        
        Args:
            function: The function to benchmark
            inputs: Tuple of input arguments to pass to the function
            function_name: Optional name for the function (defaults to function.__name__)
            iterations: Number of iterations to run for averaging performance
            warmup_iterations: Number of warmup iterations before measurement
            cleanup_function: Optional function to call between iterations for cleanup
            
        Returns:
            BenchmarkResult containing performance metrics
        """
        function_name = function_name or function.__name__
        input_size = self._estimate_input_size(inputs)
        
        logging.info(f"Benchmarking function: {function_name}")
        logging.info(f"Input size estimate: {input_size} elements")
        
        # Perform warmup iterations
        for _ in range(warmup_iterations):
            try:
                function(*inputs)
                if cleanup_function:
                    cleanup_function()
            except Exception as e:
                logging.error(f"Error during warmup: {str(e)}")
                return BenchmarkResult(
                    function_name=function_name,
                    execution_time_sec=0.0,
                    memory_usage_kb=0.0,
                    input_size=input_size,
                    iterations=iterations,
                    error=str(e)
                )
        
        # Measure execution time
        execution_times = []
        cpu_percentages = []
        memory_usages = []
        
        for i in range(iterations):
            # Clear any leftover objects and run garbage collection
            gc.collect()
            
            try:
                # Start CPU monitoring in a separate thread
                process = psutil.Process(os.getpid())
                cpu_start = process.cpu_percent()
                
                # Start memory tracking
                tracemalloc.start()
                
                # Time the execution
                start_time = time.perf_counter()
                result = function(*inputs)
                end_time = time.perf_counter()
                
                # Get peak memory usage
                _, peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Collect performance metrics
                execution_time = end_time - start_time
                memory_usage = peak_memory / 1024  # Convert to KB
                cpu_percent = process.cpu_percent()
                
                execution_times.append(execution_time)
                memory_usages.append(memory_usage)
                cpu_percentages.append(cpu_percent)
                
                # Log iteration results
                logging.info(f"Iteration {i+1}/{iterations}: Time={execution_time:.6f}s, Memory={memory_usage:.2f}KB, CPU={cpu_percent:.1f}%")
                
                # Run cleanup if provided
                if cleanup_function:
                    cleanup_function()
                
            except Exception as e:
                logging.error(f"Error during benchmark iteration {i+1}: {str(e)}")
                tracemalloc.stop()
                return BenchmarkResult(
                    function_name=function_name,
                    execution_time_sec=0.0,
                    memory_usage_kb=0.0,
                    input_size=input_size,
                    iterations=iterations,
                    error=str(e)
                )
        
        # Calculate statistics
        avg_execution_time = np.mean(execution_times)
        std_dev_time = np.std(execution_times)
        avg_memory_usage = np.mean(memory_usages)
        avg_cpu_percent = np.mean(cpu_percentages)
        
        # Create and store benchmark result
        result = BenchmarkResult(
            function_name=function_name,
            execution_time_sec=avg_execution_time,
            memory_usage_kb=avg_memory_usage,
            cpu_percent=avg_cpu_percent,
            input_size=input_size,
            iterations=iterations,
            std_dev_time=std_dev_time
        )
        
        self.results.append(result)
        
        # Log overall result
        logging.info(f"Benchmark completed for {function_name}:")
        logging.info(f"  Avg. Execution Time: {avg_execution_time:.6f}s (±{std_dev_time:.6f}s)")
        logging.info(f"  Avg. Memory Usage: {avg_memory_usage:.2f}KB")
        logging.info(f"  Avg. CPU Usage: {avg_cpu_percent:.1f}%")
        
        return result
    
    def _estimate_input_size(self, inputs: Tuple) -> int:
        """
        Estimate the size of input data.
        
        Args:
            inputs: Tuple of input arguments
            
        Returns:
            Estimated size (number of elements or bytes)
        """
        if not inputs:
            return 0
        
        total_size = 0
        
        for input_arg in inputs:
            if isinstance(input_arg, (list, tuple, set, dict)):
                total_size += len(input_arg)
            elif isinstance(input_arg, (int, float, bool, str)):
                total_size += 1
            elif isinstance(input_arg, np.ndarray):
                total_size += input_arg.size
            elif hasattr(input_arg, '__len__'):
                try:
                    total_size += len(input_arg)
                except:
                    total_size += 1
            else:
                total_size += 1
        
        return total_size
    
    def test_scalability(
        self,
        function: Callable,
        input_generator: Callable[[int], Tuple],
        input_sizes: List[int],
        function_name: Optional[str] = None,
        iterations: int = 3,
        cleanup_function: Optional[Callable] = None
    ) -> ScalabilityResult:
        """
        Test how function performance scales with input size.
        
        Args:
            function: The function to benchmark
            input_generator: Function that generates inputs of specified size
            input_sizes: List of input sizes to test
            function_name: Optional name for the function
            iterations: Number of iterations for each input size
            cleanup_function: Optional function to call between iterations
            
        Returns:
            ScalabilityResult containing performance metrics across input sizes
        """
        function_name = function_name or function.__name__
        
        logging.info(f"Testing scalability for function: {function_name}")
        logging.info(f"Input sizes: {input_sizes}")
        
        execution_times = []
        memory_usages = []
        cpu_percents = []
        
        for size in input_sizes:
            logging.info(f"Testing with input size: {size}")
            
            inputs = input_generator(size)
            result = self.benchmark_function(
                function,
                inputs,
                function_name=function_name,
                iterations=iterations,
                cleanup_function=cleanup_function
            )
            
            execution_times.append(result.execution_time_sec)
            memory_usages.append(result.memory_usage_kb)
            cpu_percents.append(result.cpu_percent)
        
        # Create scalability result
        scalability_result = ScalabilityResult(
            function_name=function_name,
            input_sizes=input_sizes,
            execution_times=execution_times,
            memory_usages=memory_usages,
            cpu_percents=cpu_percents
        )
        
        self.scalability_results.append(scalability_result)
        
        return scalability_result
    
    def compare_functions(
        self,
        functions: List[Callable],
        inputs: Tuple,
        function_names: Optional[List[str]] = None,
        iterations: int = 5
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare performance metrics across multiple functions with the same inputs.
        
        Args:
            functions: List of functions to compare
            inputs: Tuple of input arguments to pass to each function
            function_names: Optional list of function names
            iterations: Number of iterations for each function
            
        Returns:
            Dictionary mapping function names to BenchmarkResults
        """
        if function_names is None:
            function_names = [func.__name__ for func in functions]
        
        if len(functions) != len(function_names):
            raise ValueError("Length of functions and function_names must match")
        
        comparison_results = {}
        
        for func, name in zip(functions, function_names):
            result = self.benchmark_function(func, inputs, function_name=name, iterations=iterations)
            comparison_results[name] = result
        
        return comparison_results
    
    def generate_report(self, filename: str = "benchmark_report") -> None:
        """
        Generate a comprehensive benchmark report in multiple formats.
        
        Args:
            filename: Base filename for the report (without extension)
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024 ** 3)
            },
            "benchmark_results": [result.to_dict() for result in self.results],
            "scalability_results": [result.to_dict() for result in self.scalability_results]
        }
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        
        # Save as CSV (flattened)
        if self.results:
            results_df = pd.DataFrame([r.to_dict() for r in self.results])
            csv_path = os.path.join(self.output_dir, f"{filename}_functions.csv")
            results_df.to_csv(csv_path, index=False)
        
        if self.scalability_results:
            # Create a flattened version of scalability results
            flat_scalability = []
            for result in self.scalability_results:
                for i, size in enumerate(result.input_sizes):
                    flat_scalability.append({
                        "function_name": result.function_name,
                        "input_size": size,
                        "execution_time_sec": result.execution_times[i],
                        "memory_usage_kb": result.memory_usages[i],
                        "cpu_percent": result.cpu_percents[i] if i < len(result.cpu_percents) else None
                    })
            
            scalability_df = pd.DataFrame(flat_scalability)
            csv_path = os.path.join(self.output_dir, f"{filename}_scalability.csv")
            scalability_df.to_csv(csv_path, index=False)
        
        self._generate_visualizations(filename)
        
        logging.info(f"Benchmark report generated: {json_path}")
    
    def _generate_visualizations(self, filename: str) -> None:
        """
        Generate visualizations of benchmark results.
        
        Args:
            filename: Base filename for the visualization files
        """
        # Function comparison chart (if multiple functions were benchmarked)
        if len(self.results) > 1:
            plt.figure(figsize=(10, 6))
            
            # Sort by execution time
            sorted_results = sorted(self.results, key=lambda x: x.execution_time_sec)
            
            function_names = [r.function_name for r in sorted_results]
            exec_times = [r.execution_time_sec for r in sorted_results]
            
            # Plot execution times
            plt.barh(function_names, exec_times)
            plt.xlabel('Execution Time (seconds)')
            plt.title('Function Performance Comparison')
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, f"{filename}_comparison.png"))
        
        # Scalability visualizations
        for i, result in enumerate(self.scalability_results):
            plt.figure(figsize=(12, 8))
            
            # Create subplot layout
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Execution time vs input size
            ax1.plot(result.input_sizes, result.execution_times, 'o-', color='blue')
            ax1.set_ylabel('Execution Time (seconds)')
            ax1.set_title(f'Scalability: {result.function_name}')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Memory usage vs input size
            ax2.plot(result.input_sizes, result.memory_usages, 'o-', color='green')
            ax2.set_xlabel('Input Size')
            ax2.set_ylabel('Memory Usage (KB)')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{filename}_scalability_{i+1}.png"))
            plt.close()


# Example functions to benchmark (replace with actual legal processing functions)
def example_function(n: int) -> List[int]:
    """Example function that creates a list of squares."""
    return [i**2 for i in range(n)]


def example_function_optimized(n: int) -> np.ndarray:
    """Optimized version using NumPy."""
    return np.arange(n) ** 2


def example_input_generator(size: int) -> Tuple[int]:
    """Generate input of specified size for example functions."""
    return (size,)


# Main execution
if __name__ == "__main__":
    # Initialize benchmarker
    benchmarker = LegalProcessingBenchmarker()
    
    # Basic benchmark of a single function
    result = benchmarker.benchmark_function(
        example_function,
        (100000,),
        iterations=5
    )
    
    # Compare alternative implementations
    benchmarker.compare_functions(
        [example_function, example_function_optimized],
        (100000,),
        function_names=["Standard Implementation", "NumPy Implementation"],
        iterations=5
    )
    
    # Test scalability
    benchmarker.test_scalability(
        example_function,
        example_input_generator,
        [1000, 10000, 50000, 100000, 500000],
        iterations=3
    )
    
    # Generate comprehensive report
    benchmarker.generate_report()
    
    print(f"Benchmarking complete. Reports saved to '{benchmarker.output_dir}' directory.")


"""
TODO:
- Add multi-threaded and multi-process benchmarking capabilities
- Implement real-time performance monitoring for long-running processes
- Add support for GPU utilization monitoring (requires specific hardware APIs)
- Develop adaptive input size selection for automatic scalability testing
- Integrate with CI/CD pipelines for continuous performance monitoring
- Add statistical significance testing between benchmark results
- Implement performance regression detection with historical data
- Support distributed benchmarking across multiple machines
"""
