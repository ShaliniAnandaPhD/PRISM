"""
main.py - Main CLI entry point for Phi-4 + PRISM fusion model

This module provides the command-line interface for interacting with
the Phi-4 + PRISM fusion model for legal tasks. It parses arguments,
configures the model, and manages the inference workflow.

Usage:
    python main.py --query "Analyze this contract for potential risks" --file contract.pdf
    python main.py --interactive --fusion-ratio 0.7 0.3
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import torch

from config import ModelConfig, load_config
from fusion import FusionModel
from models.phi4_model import Phi4Model
from models.prism_model import PRISMModel
from utils.document_processor import DocumentProcessor
from utils.logger import setup_logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phi-4 + PRISM fusion model for legal analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--query", type=str, help="Single query mode - provide your legal question"
    )
    mode_group.add_argument(
        "--interactive", action="store_true", help="Start interactive shell mode"
    )
    mode_group.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks"
    )

    # Input options
    input_group = parser.add_argument_group("Input options")
    input_group.add_argument(
        "--file", type=str, help="Path to legal document for analysis"
    )
    input_group.add_argument(
        "--image", type=str, help="Path to image of document (for multimodal analysis)"
    )

    # Model configuration
    model_group = parser.add_argument_group("Model configuration")
    model_group.add_argument(
        "--fusion-ratio", type=float, nargs=2, default=[0.7, 0.3],
        help="Fusion ratio between Phi-4 and PRISM (must sum to 1.0)"
    )
    model_group.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to config file"
    )
    model_group.add_argument(
        "--use-gpu", action="store_true", help="Use GPU for inference if available"
    )
    model_group.add_argument(
        "--precision", type=str, choices=["fp16", "fp32", "int8"], default="fp16",
        help="Model precision for inference"
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output", type=str, help="Save output to specified file"
    )
    output_group.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase verbosity (can be used multiple times)"
    )
    output_group.add_argument(
        "--visualize", action="store_true", help="Generate fusion visualization"
    )

    args = parser.parse_args()

    # Validate arguments
    if sum(args.fusion_ratio) != 1.0:
        parser.error("Fusion ratio must sum to 1.0")

    return args


def setup_models(config: ModelConfig, device: str, fusion_ratio: List[float]
                ) -> Tuple[Phi4Model, PRISMModel, FusionModel]:
    """
    Initialize and configure the models.
    
    Args:
        config: Model configuration parameters
        device: Device to run models on (cuda or cpu)
        fusion_ratio: Ratio for fusion layer [phi4_weight, prism_weight]
        
    Returns:
        Tuple of initialized models (phi4, prism, fusion)
    """
    logging.info("Initializing Phi-4 model...")
    phi4 = Phi4Model(
        model_path=config.phi4_model_path,
        device=device,
        precision=config.precision
    )
    
    logging.info("Initializing PRISM model...")
    prism = PRISMModel(
        model_path=config.prism_model_path,
        index_path=config.document_index_path,
        device=device,
        precision=config.precision
    )
    
    logging.info(f"Creating fusion model with ratio {fusion_ratio}...")
    fusion = FusionModel(
        phi4_model=phi4,
        prism_model=prism,
        fusion_ratio=fusion_ratio,
        lora_path=config.lora_weights_path,
        device=device
    )
    
    return phi4, prism, fusion


def process_query(query: str, 
                 fusion_model: FusionModel, 
                 doc_processor: Optional[DocumentProcessor] = None,
                 document_path: Optional[str] = None,
                 image_path: Optional[str] = None,
                 output_path: Optional[str] = None,
                 visualize: bool = False) -> Dict:
    """
    Process a single query through the fusion model.
    
    Args:
        query: User query text
        fusion_model: Initialized fusion model
        doc_processor: Document processor for handling files
        document_path: Optional path to document to analyze
        image_path: Optional path to image to analyze
        output_path: Optional path to save results
        visualize: Whether to generate attention visualizations
        
    Returns:
        Dictionary containing query results
    """
    start_time = time.time()
    
    # Process any input documents or images
    document_content = None
    image_content = None
    
    if document_path and doc_processor:
        logging.info(f"Processing document: {document_path}")
        document_content = doc_processor.process_document(document_path)
    
    if image_path:
        logging.info(f"Processing image: {image_path}")
        image_content = fusion_model.phi4_model.process_image(image_path)
    
    # Run inference
    logging.info("Running inference with fusion model...")
    results = fusion_model.generate(
        query=query,
        document=document_content,
        image=image_content,
        generate_visualization=visualize
    )
    
    elapsed_time = time.time() - start_time
    results["metadata"] = {
        "processing_time": elapsed_time,
        "query": query,
        "document": document_path,
        "image": image_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save output if requested
    if output_path:
        import json
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {output_path}")
    
    return results


def interactive_mode(fusion_model: FusionModel, doc_processor: DocumentProcessor):
    """
    Run an interactive shell for the model.
    
    Args:
        fusion_model: Initialized fusion model
        doc_processor: Document processor for handling files
    """
    import readline  # Enable command history and editing
    
    print("\n==== Phi-4 + PRISM Legal Assistant ====")
    print("Type 'help' for commands, 'exit' to quit")
    print("Current fusion ratio:", fusion_model.fusion_ratio)
    
    document_path = None
    image_path = None
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ("exit", "quit"):
                break
                
            elif user_input.lower() == "help":
                print("\nCommands:")
                print("  help                   - Show this help message")
                print("  exit, quit             - Exit the program")
                print("  load document <path>   - Load a document for analysis")
                print("  load image <path>      - Load an image for analysis")
                print("  ratio <phi4> <prism>   - Change fusion ratio")
                print("  clear                  - Clear current document and image")
                print("  Any other input will be treated as a query")
                
            elif user_input.lower().startswith("load document "):
                document_path = user_input[14:].strip()
                if os.path.exists(document_path):
                    print(f"Document loaded: {document_path}")
                else:
                    print(f"File not found: {document_path}")
                    document_path = None
                    
            elif user_input.lower().startswith("load image "):
                image_path = user_input[11:].strip()
                if os.path.exists(image_path):
                    print(f"Image loaded: {image_path}")
                else:
                    print(f"File not found: {image_path}")
                    image_path = None
                    
            elif user_input.lower().startswith("ratio "):
                try:
                    parts = user_input.split()
                    if len(parts) != 3:
                        print("Usage: ratio <phi4_weight> <prism_weight>")
                        continue
                        
                    phi4_ratio = float(parts[1])
                    prism_ratio = float(parts[2])
                    
                    if phi4_ratio + prism_ratio != 1.0:
                        print("Ratios must sum to 1.0")
                        continue
                        
                    fusion_model.update_fusion_ratio([phi4_ratio, prism_ratio])
                    print(f"Fusion ratio updated: Phi-4={phi4_ratio}, PRISM={prism_ratio}")
                    
                except ValueError:
                    print("Invalid ratio values")
                    
            elif user_input.lower() == "clear":
                document_path = None
                image_path = None
                print("Document and image cleared")
                
            elif user_input.strip():
                # Process as a query
                results = process_query(
                    query=user_input,
                    fusion_model=fusion_model,
                    doc_processor=doc_processor,
                    document_path=document_path,
                    image_path=image_path
                )
                
                print("\n" + "-" * 50)
                print(results["response"])
                print("-" * 50)
                print(f"Processing time: {results['metadata']['processing_time']:.2f} seconds")
                
                # Show citations if available
                if "citations" in results and results["citations"]:
                    print("\nCitations:")
                    for citation in results["citations"]:
                        print(f"- {citation}")
                
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except Exception as e:
            print(f"Error: {e}")


def run_benchmarks(fusion_model: FusionModel, config: ModelConfig):
    """
    Run performance benchmarks on the model.
    
    Args:
        fusion_model: Initialized fusion model
        config: Model configuration
    """
    from benchmarks.performance import run_performance_benchmark
    from benchmarks.accuracy import run_accuracy_benchmark
    
    print("\n==== Running Performance Benchmarks ====")
    
    # Test different fusion ratios
    ratios_to_test = [
        [0.9, 0.1],
        [0.7, 0.3],
        [0.5, 0.5],
        [0.3, 0.7]
    ]
    
    perf_results = {}
    accuracy_results = {}
    
    for ratio in ratios_to_test:
        ratio_str = f"{ratio[0]}/{ratio[1]}"
        print(f"\nTesting ratio: {ratio_str}")
        
        # Update model ratio
        fusion_model.update_fusion_ratio(ratio)
        
        # Run benchmarks
        perf_results[ratio_str] = run_performance_benchmark(
            fusion_model, 
            benchmark_path=config.benchmark_path
        )
        
        accuracy_results[ratio_str] = run_accuracy_benchmark(
            fusion_model,
            benchmark_path=config.benchmark_path
        )
    
    # Print summary
    print("\n==== Benchmark Results ====")
    
    print("\nPerformance (tokens/sec):")
    for ratio, result in perf_results.items():
        print(f"  {ratio}: {result['tokens_per_second']:.2f} tokens/sec, " +
              f"latency: {result['latency']:.2f} ms")
    
    print("\nAccuracy:")
    for ratio, result in accuracy_results.items():
        print(f"  {ratio}: F1 score: {result['f1_score']:.4f}, " +
              f"Hallucination rate: {result['hallucination_rate']:.2%}")
    
    # Determine optimal ratio
    best_ratio = max(accuracy_results.items(), 
                     key=lambda x: x[1]['f1_score'])
    
    print(f"\nRecommended optimal ratio: {best_ratio[0]}")
    print(f"  F1 score: {best_ratio[1]['f1_score']:.4f}")
    print(f"  Hallucination rate: {best_ratio[1]['hallucination_rate']:.2%}")
    
    # Save results
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump({
            "performance": perf_results,
            "accuracy": accuracy_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recommended_ratio": best_ratio[0]
        }, f, indent=2)
    
    print("\nResults saved to benchmark_results.json")


def main():
    """Main entry point for the CLI application."""
    args = parse_arguments()
    
    # Set up logging
    log_level = max(logging.WARNING - args.verbose * 10, logging.DEBUG)
    setup_logger(log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    logging.info(f"Using device: {device}")
    
    # Set up models
    phi4, prism, fusion_model = setup_models(
        config=config,
        device=device,
        fusion_ratio=args.fusion_ratio
    )
    
    # Initialize document processor
    doc_processor = DocumentProcessor()
    
    try:
        if args.benchmark:
            # Run benchmarks
            run_benchmarks(fusion_model, config)
            
        elif args.interactive:
            # Interactive mode
            interactive_mode(fusion_model, doc_processor)
            
        else:
            # Single query mode
            results = process_query(
                query=args.query,
                fusion_model=fusion_model,
                doc_processor=doc_processor,
                document_path=args.file,
                image_path=args.image,
                output_path=args.output,
                visualize=args.visualize
            )
            
            # Print response
            print("\n" + "-" * 50)
            print(results["response"])
            print("-" * 50)
            
            # Show metadata if verbose
            if args.verbose > 0:
                print(f"\nProcessing time: {results['metadata']['processing_time']:.2f} seconds")
                
                if "citations" in results and results["citations"]:
                    print("\nCitations:")
                    for citation in results["citations"]:
                        print(f"- {citation}")
    
    finally:
        # Clean up resources
        logging.info("Cleaning up resources...")
        fusion_model.cleanup()
        phi4.cleanup()
        prism.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
SUMMARY:
- Implements CLI interface for the Phi-4 + PRISM fusion model
- Supports three modes: single query, interactive shell, and benchmarking
- Handles document and image processing for multimodal analysis
- Configurable fusion ratio between models
- Provides comprehensive output with citations

TODO:
- Add more advanced visualization options for model fusion
- Implement caching for repeated document analysis
- Add API server mode for remote access
- Enhance error handling for malformed documents
- Support batch processing mode for multiple queries
- Add export options for different output formats
- Implement progressive generation with streaming output
"""
