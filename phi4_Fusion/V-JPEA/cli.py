#  Command-line interface for the V-JEPA to PRISM pipeline

import os
import sys
import argparse
import logging
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Import components from other files
from config_manager import Config
from vjepa_extractor import VJEPALatentExtractor
from sequence_decoder import SequenceLatentToTextModel
from dataset_loaders import get_dataset_loader
from evaluation_metrics import TextGenerationEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("vjepa_to_prism.log"), logging.StreamHandler()]
)
logger = logging.getLogger("VJEPAtoPRISM-CLI")

def extract_command(args):
    """
    Extract latent vectors from videos using V-JEPA.
    """
    logger.info("Starting latent extraction")
    
    # Load configuration
    config = Config.load(args.config)
    
    # Initialize V-JEPA extractor
    extractor = VJEPALatentExtractor(
        model_path=config.vjepa.model_path,
        config_path=config.vjepa.config_path,
        device=args.device or config.vjepa.device,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir or config.data.cache_dir
    )
    
    # Get list of videos to process
    if args.video:
        # Single video mode
        video_paths = [args.video]
    elif args.video_dir:
        # Directory mode
        video_dir = Path(args.video_dir)
        video_paths = [str(p) for p in video_dir.glob("**/*.mp4")]
        video_paths += [str(p) for p in video_dir.glob("**/*.avi")]
        video_paths += [str(p) for p in video_dir.glob("**/*.mov")]
        logger.info(f"Found {len(video_paths)} videos in directory")
    else:
        logger.error("No video or video directory specified")
        return
    
    # Extract latents
    latents = extractor.batch_extract_latents(
        video_paths,
        batch_size=args.batch_size
    )
    
    # Save results if output path is provided
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save as PyTorch file
        results = {
            'video_paths': video_paths,
            'latents': [l.cpu() for l in latents],
            'timestamp': datetime.now().isoformat()
        }
        torch.save(results, args.output)
        logger.info(f"Saved {len(latents)} latent vectors to {args.output}")
    
    logger.info("Latent extraction complete")

def generate_command(args):
    """
    Generate text descriptions from latent vectors.
    """
    logger.info("Starting text generation")
    
    # Load configuration
    config = Config.load(args.config)
    
    # Set device
    device = torch.device(args.device or config.vjepa.device
                         if torch.cuda.is_available() and "cuda" in (args.device or config.vjepa.device)
                         else "cpu")
    
    # Initialize model
    model = SequenceLatentToTextModel(
        latent_dim=config.vjepa.embed_dim,
        model_name=args.model_name or "gpt2",
        num_layers=config.decoder.num_layers,
        hidden_dim=config.decoder.hidden_dim,
        dropout=config.decoder.dropout
    )
    
    # Load model weights
    model_path = args.model_path or config.training.output_model_path
    if not os.path.exists(model_path):
        logger.error(f"Model weights not found: {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load latents
    if not args.latents_file:
        logger.error("No latents file specified")
        return
    
    if not os.path.exists(args.latents_file):
        logger.error(f"Latents file not found: {args.latents_file}")
        return
    
    latents_data = torch.load(args.latents_file)
    video_paths = latents_data['video_paths']
    latents = latents_data['latents']
    
    logger.info(f"Loaded {len(latents)} latent vectors")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name or "gpt2")
    
    # Generate descriptions
    descriptions = []
    
    logger.info("Generating descriptions")
    with torch.no_grad():
        for i, latent in enumerate(tqdm(latents)):
            # Move latent to device
            latent = latent.to(device)
            
            # Generate text
            output_ids = model.generate_from_latent(
                latent.unsqueeze(0),
                max_length=args.max_length or config.decoder.max_length,
                num_beams=args.num_beams or config.decoder.beam_size,
                temperature=args.temperature or config.decoder.temperature
            )
            
            # Decode text
            description = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            descriptions.append(description)
            
            # Log occasional examples
            if i % 10 == 0:
                logger.info(f"Example {i}: {description}")
    
    # Save results if output path is provided
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Create results data
        results = []
        for path, desc in zip(video_paths, descriptions):
            results.append({
                'video': path,
                'description': desc
            })
        
        # Save as JSON
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved {len(descriptions)} descriptions to {args.output}")
    
    logger.info("Text generation complete")

def evaluate_command(args):
    """
    Evaluate generated descriptions against ground truth.
    """
    logger.info("Starting evaluation")
    
    # Check files
    if not args.predictions or not args.references:
        logger.error("Both predictions and references files must be specified")
        return
    
    if not os.path.exists(args.predictions):
        logger.error(f"Predictions file not found: {args.predictions}")
        return
    
    if not os.path.exists(args.references):
        logger.error(f"References file not found: {args.references}")
        return
    
    # Load predictions
    with open(args.predictions, 'r') as f:
        predictions_data = json.load(f)
    
    # Load references
    with open(args.references, 'r') as f:
        references_data = json.load(f)
    
    # Organize data
    video_to_pred = {item['video']: item['description'] for item in predictions_data}
    video_to_refs = {}
    
    for item in references_data:
        video = item['video']
        caption = item['caption']
        
        if video not in video_to_refs:
            video_to_refs[video] = []
        
        video_to_refs[video].append(caption)
    
    # Find videos in both sets
    common_videos = set(video_to_pred.keys()) & set(video_to_refs.keys())
    
    if not common_videos:
        logger.error("No common videos found between predictions and references")
        return
    
    logger.info(f"Found {len(common_videos)} videos with both predictions and references")
    
    # Prepare evaluation data
    hypotheses = []
    references_list = []
    
    for video in common_videos:
        hypotheses.append(video_to_pred[video])
        references_list.append(video_to_refs[video])
    
    # Initialize evaluator
    evaluator = TextGenerationEvaluator()
    
    # Compute metrics
    metrics = evaluator.evaluate_batch(references_list, hypotheses)
    
    # Print results
    logger.info("Evaluation results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results if output path is provided
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved evaluation results to {args.output}")
    
    logger.info("Evaluation complete")

def main():
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description="V-JEPA to PRISM Command Line Interface")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--device", help="Device to use (e.g., 'cuda:0', 'cpu')")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract latents from videos")
    extract_parser.add_argument("--video", help="Path to a video file")
    extract_parser.add_argument("--video-dir", help="Path to a directory of videos")
    extract_parser.add_argument("--output", help="Path to save extracted latents")
    extract_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    extract_parser.add_argument("--no-cache", action="store_true", help="Disable caching of latents")
    extract_parser.add_argument("--cache-dir", help="Directory for caching latents")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate descriptions from latents")
    generate_parser.add_argument("--latents-file", required=True, help="Path to file with extracted latents")
    generate_parser.add_argument("--model-path", help="Path to trained model weights")
    generate_parser.add_argument("--model-name", help="Name of pretrained model to use")
    generate_parser.add_argument("--output", help="Path to save generated descriptions")
    generate_parser.add_argument("--max-length", type=int, help="Maximum length of generated text")
    generate_parser.add_argument("--num-beams", type=int, help="Number of beams for beam search")
    generate_parser.add_argument("--temperature", type=float, help="Temperature for generation")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate generated descriptions")
    evaluate_parser.add_argument("--predictions", required=True, help="Path to file with generated descriptions")
    evaluate_parser.add_argument("--references", required=True, help="Path to file with reference descriptions")
    evaluate_parser.add_argument("--output", help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    if args.command == "extract":
        extract_command(args)
    elif args.command == "generate":
        generate_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
