"""
performance.py - Performance benchmarking for Phi-4 + PRISM fusion model

This module provides performance benchmarking utilities to evaluate the
speed, memory usage, and efficiency of the fusion model.

"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from fusion import FusionModel


def measure_inference_speed(model: FusionModel,
                           queries: List[str],
                           document: Optional[str] = None,
                           num_runs: int = 5,
                           warm_up_runs: int = 2) -> Dict[str, float]:
    """
    Measure inference speed of the model.
    
    Args:
        model: FusionModel instance
        queries: List of test queries
        document: Optional document to process
        num_runs: Number of measurement runs
        warm_up_runs: Number of warm-up runs (not measured)
        
    Returns:
        Dictionary with speed metrics
    """
    logging.info(f"Measuring inference speed with {len(queries)} queries")
    
    # Warm up
    logging.info(f"Running {warm_up_runs} warm-up iterations")
    for i in range(warm_up_runs):
        _ = model.generate(queries[0], document=document)
    
    # Measurements
    latencies = []
    tokens_per_second = []
    
    for query in queries:
        start_time = time.time()
        result = model.generate(query, document=document)
        end_time = time.time()
        
        # Calculate latency
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        # Estimate tokens in response
        response_text = result["response"]
        approx_tokens = len(response_text.split())
        
        # Calculate tokens per second
        if latency_ms > 0:
            tps = approx_tokens / (latency_ms / 1000)
            tokens_per_second.append(tps)
    
    # Calculate statistics
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))
    
    avg_tps = float(np.mean(tokens_per_second))
    
    return {
        "tokens_per_second": avg_tps,
        "latency": avg_latency,
        "p95_latency": p95_latency,
        "p99_latency": p99_latency,
        "num_queries": len(queries)
    }


def measure_memory_usage(model: FusionModel,
                        device: str = "cuda",
                        sequence_lengths: List[int] = [256, 512, 1024, 2048, 4096],
                        document_size: Optional[int] = None) -> Dict[str, Dict[int, float]]:
    """
    Measure memory usage for different sequence lengths.
    
    Args:
        model: FusionModel instance
        device: Device to measure (cuda or cpu)
        sequence_lengths: List of sequence lengths to test
        document_size: Optional document size to include
        
    Returns:
        Dictionary with memory usage metrics
    """
    logging.info(f"Measuring memory usage for sequence lengths: {sequence_lengths}")
    
    memory_usage = {
        "gpu": {},
        "cpu": {}
    }
    
    # Skip if not using CUDA
    if device != "cuda" or not torch.cuda.is_available():
        logging.warning("CUDA not available, skipping GPU memory measurement")
        return memory_usage
    
    # Create a dummy query
    base_query = "Analyze the following legal document and identify potential risks:"
    
    for seq_len in sequence_lengths:
        # Create padding to reach desired sequence length
        if seq_len > len(base_query.split()):
            padding_words = ["document"] * (seq_len - len(base_query.split()))
            query = base_query + " " + " ".join(padding_words)
        else:
            query = base_query
        
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        # Measure baseline memory
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.max_memory_allocated()
        
        # Process query
        result = model.generate(query)
        
        # Measure peak memory
        peak_memory = torch.cuda.max_memory_allocated()
        
        # Calculate memory usage in MB
        memory_used = (peak_memory - baseline_memory) / (1024 * 1024)
        
        # Store result
        memory_usage["gpu"][seq_len] = float(memory_used)
        
        # CPU memory is harder to measure accurately, 
        # would need a process-level monitoring tool
        memory_usage["cpu"][seq_len] = 0.0  # Placeholder
    
    return memory_usage


def measure_throughput_scaling(model: FusionModel,
                              queries: List[str],
                              batch_sizes: List[int] = [1, 2, 4, 8],
                              document: Optional[str] = None) -> Dict[str, Dict[int, float]]:
    """
    Measure throughput scaling with batch size.
    
    Note: This is a simplified simulation as the actual model may not support batching.
    
    Args:
        model: FusionModel instance
        queries: List of test queries
        batch_sizes: List of batch sizes to test
        document: Optional document to process
        
    Returns:
        Dictionary with throughput metrics
    """
    logging.info(f"Measuring throughput scaling for batch sizes: {batch_sizes}")
    
    throughput = {}
    
    for batch_size in batch_sizes:
        logging.info(f"Testing batch size: {batch_size}")
        
        # Select queries for this batch size
        if len(queries) < batch_size:
            batch_queries = queries * (batch_size // len(queries) + 1)
            batch_queries = batch_queries[:batch_size]
        else:
            batch_queries = queries[:batch_size]
        
        # Process serially (simulating batch)
        start_time = time.time()
        
        for query in batch_queries:
            _ = model.generate(query, document=document)
        
        end_time = time.time()
        
        # Calculate throughput (queries per second)
        total_time = end_time - start_time
        qps = batch_size / total_time if total_time > 0 else 0
        
        throughput[batch_size] = qps
    
    return {
        "queries_per_second": throughput
    }


def run_performance_benchmark(model: FusionModel,
                             benchmark_path: str = "benchmarks/data") -> Dict[str, Any]:
    """
    Run comprehensive performance benchmark.
    
    Args:
        model: FusionModel instance
        benchmark_path: Path to benchmark data directory
        
    Returns:
        Dictionary with benchmark results
    """
    logging.info("Starting performance benchmark")
    
    # Load benchmark queries
    queries_path = os.path.join(benchmark_path, "benchmark_queries.json")
    
    if not os.path.exists(queries_path):
        logging.warning(f"Benchmark queries file not found: {queries_path}")
        # Create sample queries
        queries = [
            "Analyze this contract for potential liability issues.",
            "What are the key provisions in this agreement?",
            "Identify any compliance risks in this document.",
            "Summarize the main points of this legal text.",
            "What are the termination clauses in this agreement?"
        ]
    else:
        with open(queries_path, "r") as f:
            queries = json.load(f)
    
    # Load benchmark document
    document_path = os.path.join(benchmark_path, "benchmark_document.txt")
    document = None
    
    if os.path.exists(document_path):
        with open(document_path, "r") as f:
            document = f.read()
    
    # Run benchmarks
    speed_results = measure_inference_speed(
        model=model,
        queries=queries,
        document=document
    )
    
    memory_results = measure_memory_usage(
        model=model,
        sequence_lengths=[256, 512, 1024, 2048]
    )
    
    throughput_results = measure_throughput_scaling(
        model=model,
        queries=queries,
        document=document
    )
    
    # Combine results
    results = {
        "speed": speed_results,
        "memory": memory_results,
        "throughput": throughput_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fusion_ratio": model.fusion_ratio
    }
    
    # Log summary
    logging.info(f"Performance benchmark completed")
    logging.info(f"Tokens per second: {speed_results['tokens_per_second']:.2f}")
    logging.info(f"Average latency: {speed_results['latency']:.2f} ms")
    
    return results


"""
SUMMARY:
- Provides performance benchmarking utilities for the fusion model
- Measures inference speed, memory usage, and throughput scaling
- Supports customizable benchmark parameters
- Handles GPU memory profiling for different sequence lengths
- Includes simulated batch processing for throughput tests

TODO:
- Add support for parallel inference testing
- Implement more detailed memory profiling for CPU usage
- Add power consumption measurement
- Support for distributed inference benchmarking
- Add comparison with baseline models
- Implement visualization of performance results
- Add support for custom benchmark scenarios
"""
