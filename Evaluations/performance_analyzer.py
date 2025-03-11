"""
PERFORMANCE & SCALABILITY EVALUATION SCRIPT
This script measures execution time, memory usage, and scalability of a given function or model.
The goal is to ensure efficiency, detect performance bottlenecks, and assess scalability under different loads.
"""
import time
import logging
import tracemalloc
import json
import numpy as np
import matplotlib.pyplot as plt
import psutil
import os
import gc
import argparse
import cProfile
import pstats
import io
from datetime import datetime
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, Dict, List, Tuple, Union, Any

# Configure logging for auditability
logging.basicConfig(filename='performance_scalability.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def timer_decorator(func):
    """Decorator to measure execution time of any function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        logging.debug(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds")
        return result, execution_time
    return wrapper

def measure_execution_time(function, inputs, iterations=10, warmup_iterations=2):
    """
    Measure the execution time of a function over multiple iterations.
    
    :param function: The function to be tested.
    :param inputs: A tuple of input arguments.
    :param iterations: Number of times to run the function for averaging.
    :param warmup_iterations: Number of warm-up runs to perform before measurement.
    :return: Dictionary with execution time statistics (mean, median, min, max, std).
    """
    # Perform warm-up runs to account for any JIT compilation or caching effects
    for _ in range(warmup_iterations):
        function(*inputs)
    
    # Measure execution times
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        function(*inputs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Calculate statistics
    times_array = np.array(times)
    stats = {
        "mean": np.mean(times_array),
        "median": np.median(times_array),
        "min": np.min(times_array),
        "max": np.max(times_array),
        "std": np.std(times_array),
        "percentile_95": np.percentile(times_array, 95),
        "percentile_99": np.percentile(times_array, 99),
        "iterations": iterations
    }
    
    return stats

def measure_memory_usage(function, inputs, iterations=3):
    """
    Measure peak memory usage of a function execution.
    
    :param function: The function to be tested.
    :param inputs: A tuple of input arguments.
    :param iterations: Number of times to run the measurement for reliability.
    :return: Dictionary with memory usage statistics (peak, increase).
    """
    memory_peaks = []
    memory_increases = []
    
    for _ in range(iterations):
        # Clear any garbage before measurement
        gc.collect()
        
        # Get baseline memory
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[1]
        
        # Execute function and measure peak
        function(*inputs)
        current, peak = tracemalloc.get_traced_memory()
        
        # Calculate increase from baseline
        memory_increase = peak - baseline
        
        tracemalloc.stop()
        
        # Store in KB
        memory_peaks.append(peak / 1024)
        memory_increases.append(memory_increase / 1024)
    
    return {
        "peak_kb_mean": np.mean(memory_peaks),
        "peak_kb_max": np.max(memory_peaks),
        "increase_kb_mean": np.mean(memory_increases),
        "increase_kb_max": np.max(memory_increases)
    }

def measure_cpu_utilization(function, inputs, iterations=3):
    """
    Measure CPU utilization during function execution.
    
    :param function: The function to be tested.
    :param inputs: A tuple of input arguments.
    :param iterations: Number of times to run the measurement for reliability.
    :return: Dictionary with CPU utilization statistics.
    """
    cpu_percent_readings = []
    
    for _ in range(iterations):
        # Clear any pending tasks
        gc.collect()
        
        # Start CPU monitoring in a separate thread
        process = psutil.Process(os.getpid())
        
        # Get baseline
        process.cpu_percent(interval=None)  # First call returns 0.0
        time.sleep(0.1)  # Short delay to get initial reading
        
        # Execute function
        start_time = time.perf_counter()
        function(*inputs)
        end_time = time.perf_counter()
        
        # Get CPU percent (this returns the CPU time since last call as a percentage)
        cpu_percent = process.cpu_percent(interval=None)
        cpu_percent_readings.append(cpu_percent)
    
    return {
        "cpu_percent_mean": np.mean(cpu_percent_readings),
        "cpu_percent_max": np.max(cpu_percent_readings)
    }

def profile_function(function, inputs, output_file=None):
    """
    Profile a function to identify bottlenecks.
    
    :param function: The function to be profiled.
    :param inputs: A tuple of input arguments.
    :param output_file: Optional file to save profile results.
    :return: String with profiling statistics.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    function(*inputs)
    profiler.disable()
    
    # Get string output
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions
    
    # Save to file if requested
    if output_file:
        ps.dump_stats(output_file)
    
    return s.getvalue()

def test_scalability(function, input_generator, input_sizes, iterations=3):
    """
    Test scalability by measuring performance across different input sizes.
    
    :param function: The function to be tested.
    :param input_generator: Function that takes a size parameter and returns inputs.
    :param input_sizes: List of input sizes to test.
    :param iterations: Number of iterations for each size.
    :return: Dictionary with scalability data.
    """
    results = {
        "input_sizes": input_sizes,
        "execution_times": [],
        "memory_usage": [],
        "cpu_utilization": []
    }
    
    for size in input_sizes:
        inputs = input_generator(size)
        
        # Measure execution time
        time_stats = measure_execution_time(function, inputs, iterations=iterations)
        results["execution_times"].append(time_stats["mean"])
        
        # Measure memory usage
        memory_stats = measure_memory_usage(function, inputs)
        results["memory_usage"].append(memory_stats["peak_kb_mean"])
        
        # Measure CPU utilization
        cpu_stats = measure_cpu_utilization(function, inputs)
        results["cpu_utilization"].append(cpu_stats["cpu_percent_mean"])
        
        logging.info(f"Input size {size}: Time {time_stats['mean']:.6f}s, Memory {memory_stats['peak_kb_mean']:.2f}KB, CPU {cpu_stats['cpu_percent_mean']:.2f}%")
    
    return results

def test_threading_impact(function, inputs, max_threads=8, iterations=3):
    """
    Test impact of multi-threading on performance.
    
    :param function: The function to be tested.
    :param inputs: A tuple of input arguments.
    :param max_threads: Maximum number of threads to test.
    :param iterations: Number of iterations for each thread count.
    :return: Dictionary with threading performance data.
    """
    thread_counts = list(range(1, max_threads + 1))
    results = {
        "thread_counts": thread_counts,
        "execution_times": []
    }
    
    def run_in_threads(n_threads):
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(function, *inputs) for _ in range(n_threads)]
            # Wait for all to complete
            for future in futures:
                future.result()
                
        end_time = time.perf_counter()
        return end_time - start_time
    
    for n_threads in thread_counts:
        times = []
        for _ in range(iterations):
            execution_time = run_in_threads(n_threads)
            times.append(execution_time)
        
        avg_time = np.mean(times)
        results["execution_times"].append(avg_time)
        logging.info(f"Threads: {n_threads}, Avg execution time: {avg_time:.6f}s")
    
    return results

def plot_scalability_results(results, output_dir="performance_results"):
    """
    Generate plots of scalability test results.
    
    :param results: Dictionary with scalability data.
    :param output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Execution time plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["input_sizes"], results["execution_times"], marker='o')
    plt.title("Execution Time vs Input Size")
    plt.xlabel("Input Size")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "execution_time_scalability.png"))
    plt.close()
    
    # Memory usage plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["input_sizes"], results["memory_usage"], marker='o', color='green')
    plt.title("Memory Usage vs Input Size")
    plt.xlabel("Input Size")
    plt.ylabel("Memory Usage (KB)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "memory_usage_scalability.png"))
    plt.close()
    
    # CPU utilization plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["input_sizes"], results["cpu_utilization"], marker='o', color='red')
    plt.title("CPU Utilization vs Input Size")
    plt.xlabel("Input Size")
    plt.ylabel("CPU Utilization (%)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "cpu_utilization_scalability.png"))
    plt.close()
    
    # Attempt to determine complexity (log-log plot)
    plt.figure(figsize=(10, 6))
    log_sizes = np.log(results["input_sizes"])
    log_times = np.log(results["execution_times"])
    
    # Linear regression to estimate complexity
    if len(log_sizes) > 1:  # Need at least 2 points for regression
        slope, intercept = np.polyfit(log_sizes, log_times, 1)
        plt.scatter(log_sizes, log_times, color='blue')
        plt.plot(log_sizes, slope * log_sizes + intercept, color='red', 
                linestyle='--', label=f'Slope: {slope:.2f} (O(n^{slope:.2f}))')
        plt.title("Algorithmic Complexity Analysis (Log-Log Plot)")
        plt.xlabel("Log(Input Size)")
        plt.ylabel("Log(Execution Time)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "complexity_analysis.png"))
        plt.close()
    
def plot_threading_results(results, output_dir="performance_results"):
    """
    Generate plots of threading test results.
    
    :param results: Dictionary with threading performance data.
    :param output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Threading impact plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["thread_counts"], results["execution_times"], marker='o')
    plt.title("Execution Time vs Thread Count")
    plt.xlabel("Thread Count")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "threading_performance.png"))
    plt.close()
    
    # Calculate and plot speedup relative to single thread
    if results["execution_times"] and results["execution_times"][0] > 0:
        plt.figure(figsize=(10, 6))
        base_time = results["execution_times"][0]
        speedups = [base_time / time for time in results["execution_times"]]
        
        plt.plot(results["thread_counts"], speedups, marker='o', color='green')
        plt.plot(results["thread_counts"], results["thread_counts"], linestyle='--', color='gray', label='Ideal Speedup')
        plt.title("Speedup vs Thread Count")
        plt.xlabel("Thread Count")
        plt.ylabel("Speedup (relative to 1 thread)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "threading_speedup.png"))
        plt.close()

def analyze_performance(function, test_params, output_dir="performance_results"):
    """
    Comprehensive performance analysis of a function.
    
    :param function: The function to be analyzed.
    :param test_params: Dictionary with test parameters.
    :param output_dir: Directory to save results.
    :return: Dictionary with all performance data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        "function_name": function.__name__,
        "timestamp": datetime.now().isoformat(),
        "basic_metrics": {},
        "scalability_data": {},
        "threading_data": {},
        "profiling_data": ""
    }
    
    # Basic performance metrics
    logging.info(f"Measuring basic performance metrics for {function.__name__}")
    
    # Get base input for initial tests
    base_input = test_params.get("base_input", (1000,))
    iterations = test_params.get("iterations", 5)
    
    # Execution time
    time_stats = measure_execution_time(function, base_input, iterations)
    results["basic_metrics"]["execution_time"] = time_stats
    
    # Memory usage
    memory_stats = measure_memory_usage(function, base_input)
    results["basic_metrics"]["memory_usage"] = memory_stats
    
    # CPU utilization
    cpu_stats = measure_cpu_utilization(function, base_input)
    results["basic_metrics"]["cpu_utilization"] = cpu_stats
    
    # Profiling
    profile_output = profile_function(function, base_input, 
                                     os.path.join(output_dir, "profile_results.prof"))
    results["profiling_data"] = profile_output
    
    # Save profiling output to file
    with open(os.path.join(output_dir, "profile_results.txt"), "w") as f:
        f.write(profile_output)
    
    # Scalability testing
    if "input_generator" in test_params and "input_sizes" in test_params:
        logging.info(f"Testing scalability for {function.__name__}")
        scalability_results = test_scalability(
            function, 
            test_params["input_generator"],
            test_params["input_sizes"],
            test_params.get("scalability_iterations", 3)
        )
        results["scalability_data"] = scalability_results
        
        # Plot scalability results
        plot_scalability_results(scalability_results, output_dir)
    
    # Threading impact testing
    if test_params.get("test_threading", False):
        logging.info(f"Testing threading impact for {function.__name__}")
        threading_results = test_threading_impact(
            function,
            base_input,
            test_params.get("max_threads", 8),
            test_params.get("threading_iterations", 3)
        )
        results["threading_data"] = threading_results
        
        # Plot threading results
        plot_threading_results(threading_results, output_dir)
    
    # Save complete results to JSON
    with open(os.path.join(output_dir, "performance_scalability_summary.json"), "w") as f:
        # Convert numpy values to native Python types for JSON serialization
        json_safe_results = convert_to_json_serializable(results)
        json.dump(json_safe_results, f, indent=4)
    
    return results

def convert_to_json_serializable(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_json_serializable(obj.tolist())
    else:
        return obj

def print_summary(results):
    """Print a summary of the performance analysis results."""
    print("\n" + "=" * 80)
    print(f"PERFORMANCE & SCALABILITY ANALYSIS SUMMARY: {results['function_name']}")
    print("=" * 80)
    
    # Basic metrics
    if "basic_metrics" in results:
        basic = results["basic_metrics"]
        
        if "execution_time" in basic:
            time_stats = basic["execution_time"]
            print(f"\nExecution Time:")
            print(f"  Mean:      {time_stats['mean']:.6f} seconds")
            print(f"  Median:    {time_stats['median']:.6f} seconds")
            print(f"  Min/Max:   {time_stats['min']:.6f} / {time_stats['max']:.6f} seconds")
            print(f"  Std Dev:   {time_stats['std']:.6f} seconds")
            print(f"  95th/99th: {time_stats['percentile_95']:.6f} / {time_stats['percentile_99']:.6f} seconds")
        
        if "memory_usage" in basic:
            memory_stats = basic["memory_usage"]
            print(f"\nMemory Usage:")
            print(f"  Peak:      {memory_stats['peak_kb_mean']:.2f} KB")
            print(f"  Increase:  {memory_stats['increase_kb_mean']:.2f} KB")
        
        if "cpu_utilization" in basic:
            cpu_stats = basic["cpu_utilization"]
            print(f"\nCPU Utilization:")
            print(f"  Mean:      {cpu_stats['cpu_percent_mean']:.2f}%")
            print(f"  Max:       {cpu_stats['cpu_percent_max']:.2f}%")
    
    # Scalability data
    if "scalability_data" in results and results["scalability_data"]:
        print("\nScalability Analysis:")
        scalability = results["scalability_data"]
        if "input_sizes" in scalability and len(scalability["input_sizes"]) > 0:
            # Print first and last few data points
            sizes = scalability["input_sizes"]
            times = scalability["execution_times"]
            
            print("  Input Size -> Execution Time")
            for i in range(min(3, len(sizes))):
                print(f"  {sizes[i]} -> {times[i]:.6f} seconds")
            
            if len(sizes) > 6:
                print("  ...")
                
            for i in range(max(3, len(sizes)-3), len(sizes)):
                print(f"  {sizes[i]} -> {times[i]:.6f} seconds")
    
    # Threading data
    if "threading_data" in results and results["threading_data"]:
        print("\nThreading Analysis:")
        threading = results["threading_data"]
        if "thread_counts" in threading and len(threading["thread_counts"]) > 0:
            print("  Threads -> Execution Time")
            for i in range(len(threading["thread_counts"])):
                print(f"  {threading['thread_counts'][i]} -> {threading['execution_times'][i]:.6f} seconds")
    
    print("\n" + "=" * 80)
    print(f"Detailed results saved to performance_results directory")
    print("=" * 80 + "\n")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Performance & Scalability Analysis")
    
    parser.add_argument("--iterations", type=int, default=5,
                      help="Number of iterations for time measurements")
    
    parser.add_argument("--output-dir", default="performance_results",
                      help="Directory to save results and plots")
    
    parser.add_argument("--test-threading", action="store_true",
                      help="Test threading impact on performance")
    
    parser.add_argument("--max-threads", type=int, default=8,
                      help="Maximum number of threads to test")
    
    parser.add_argument("--max-input-size", type=int, default=1000000,
                      help="Maximum input size for scalability testing")
    
    return parser.parse_args()

# Example function to be tested
def example_function(n):
    """Example function that creates a list of squares."""
    return [i**2 for i in range(n)]

# Example input generator for scalability testing
def input_generator(size):
    """Generate input tuple based on size parameter."""
    return (size,)

# Run performance evaluation
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Define test parameters
    test_params = {
        "base_input": (10000,),
        "iterations": args.iterations,
        "input_generator": input_generator,
        "input_sizes": [100, 1000, 10000, 100000, args.max_input_size],
        "scalability_iterations": 3,
        "test_threading": args.test_threading,
        "max_threads": args.max_threads,
        "threading_iterations": 3
    }
    
    # Run comprehensive analysis
    results = analyze_performance(example_function, test_params, args.output_dir)
    
    # Print summary
    print_summary(results)
    
    print("Performance & Scalability evaluation complete.")
    print(f"Results saved to '{args.output_dir}/performance_scalability_summary.json'")
    print(f"Visualizations saved to '{args.output_dir}'")

"""
TODO:
- Extend to measure GPU utilization and performance for GPU-accelerated functions.
- Implement distributed computing performance analysis for cluster deployments.
- Add automated bottleneck detection and optimization recommendations.
- Support performance comparison between different implementations or algorithm versions.
- Integrate with continuous integration workflows for regression testing.
- Add network performance analysis for distributed or client-server applications.
- Implement power consumption measurement for energy efficiency analysis.
- Add support for database query performance assessment.
- Extend visualization capabilities with interactive dashboards.
- Implement anomaly detection for identifying performance outliers across test runs.
"""
