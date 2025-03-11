# Main CLI Entry Point

import os
import sys
import time
import logging
import argparse
import yaml
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress

# Import PRISM modules
from prism.case_manager import CaseManager
from prism.audio_analyzer import AudioAnalyzer
from prism.video_analyzer import VideoAnalyzer
from prism.evidence_correlator import EvidenceCorrelator
from prism.document_generator import DocumentGenerator
from prism.media_extractor import MediaExtractor
from prism.utils import setup_logging, print_header, print_success, print_info, print_warning, print_error

# Global variables
VERSION = "3.4.2.128"
console = Console()

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="PRISM - Protection-order Resource for Integrated Scanning and Management")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # init command
    init_parser = subparsers.add_parser("init", help="Initialize a new case")
    init_parser.add_argument("name", help="Case name")
    init_parser.add_argument("--jurisdiction", choices=["CA", "NY", "TX", "FL", "IL"], default="CA", help="Legal jurisdiction")
    init_parser.add_argument("--detailed-logging", action="store_true", help="Enable detailed logging")
    
    # import command
    import_parser = subparsers.add_parser("import", help="Import evidence files")
    import_parser.add_argument("path", help="Path to evidence files")
    import_parser.add_argument("--tags", help="Tags to apply to imported files")
    import_parser.add_argument("--case-type", default="protection-order", help="Case type")
    import_parser.add_argument("--chain-of-custody", action="store_true", help="Enable chain of custody tracking")
    import_parser.add_argument("--detailed-scan", action="store_true", help="Perform detailed file scanning")
    
    # analyze-audio command
    audio_parser = subparsers.add_parser("analyze-audio", help="Analyze audio files")
    audio_parser.add_argument("path", help="Path to audio files")
    audio_parser.add_argument("--transcribe", action="store_true", help="Transcribe audio")
    audio_parser.add_argument("--emotional-markers", action="store_true", help="Detect emotional markers")
    audio_parser.add_argument("--speaker-id", action="store_true", help="Perform speaker identification")
    audio_parser.add_argument("--vocal-stress", action="store_true", help="Detect vocal stress")
    audio_parser.add_argument("--detailed-analysis", action="store_true", help="Perform detailed analysis")
    audio_parser.add_argument("--quality-enhancement", action="store_true", help="Enhance audio quality")
    
    # analyze-video command
    video_parser = subparsers.add_parser("analyze-video", help="Analyze video files")
    video_parser.add_argument("path", help="Path to video files")
    video_parser.add_argument("--object-detection", action="store_true", help="Perform object detection")
    video_parser.add_argument("--facial-recognition", action="store_true", help="Perform facial recognition")
    video_parser.add_argument("--incident-detection", action="store_true", help="Detect incidents")
    video_parser.add_argument("--extract-frames", action="store_true", help="Extract key frames")
    video_parser.add_argument("--technical-forensics", action="store_true", help="Perform technical forensic analysis")
    video_parser.add_argument("--background-removal", action="store_true", help="Apply background removal")
    
    # correlate-evidence command
    correlate_parser = subparsers.add_parser("correlate-evidence", help="Correlate evidence")
    correlate_parser.add_argument("--modalities", choices=["all", "audio", "video", "text"], default="all", help="Evidence modalities to correlate")
    correlate_parser.add_argument("--timeline", action="store_true", help="Create evidence timeline")
    correlate_parser.add_argument("--threshold", type=float, default=0.7, help="Correlation confidence threshold")
    correlate_parser.add_argument("--output", help="Output file path")
    correlate_parser.add_argument("--detailed-analysis", action="store_true", help="Perform detailed analysis")
    
    # extract-multimedia-segments command
    extract_parser = subparsers.add_parser("extract-multimedia-segments", help="Extract multimedia segments")
    extract_parser.add_argument("--config", required=True, help="Configuration file for segments to extract")
    extract_parser.add_argument("--output", required=True, help="Output directory")
    extract_parser.add_argument("--forensic-packaging", action="store_true", help="Create forensic packaging")
    
    # generate-protection-order command
    generate_parser = subparsers.add_parser("generate-protection-order", help="Generate protection order documents")
    generate_parser.add_argument("--template", required=True, help="Document template to use")
    generate_parser.add_argument("--court", required=True, help="Court name")
    generate_parser.add_argument("--evidence-integration", choices=["low", "medium", "high"], default="medium", help="Level of evidence integration")
    generate_parser.add_argument("--expedited", action="store_true", help="Mark as expedited")
    generate_parser.add_argument("--incident-details", action="store_true", help="Include detailed incident information")
    
    # generate-case-summary command
    summary_parser = subparsers.add_parser("generate-case-summary", help="Generate case summary")
    summary_parser.add_argument("--comprehensive", action="store_true", help="Create comprehensive summary")
    summary_parser.add_argument("--include-technical", action="store_true", help="Include technical details")
    summary_parser.add_argument("--evidence-strengths", action="store_true", help="Include evidence strength assessment")
    
    # processing-summary command
    subparsers.add_parser("processing-summary", help="Show processing summary")
    
    return parser.parse_args()

def handle_init(args, config):
    """Handle the init command"""
    case_manager = CaseManager(config)
    print_header(f"[PRISM v{VERSION} - Protection Order Module Activated]")
    
    start_time = time.time()
    result = case_manager.initialize_case(args.name, args.jurisdiction, args.detailed_logging)
    elapsed_time = time.time() - start_time
    
    if result:
        print_success(f"Initialized PRISM workspace for case \"{args.name}\" [Case ID: {result}]")
        print_info(f"Workspace created at: {os.path.abspath(args.name)}")
        print_info(f"Using {case_manager.get_jurisdiction_name(args.jurisdiction)} jurisdiction templates and legal references")
        
        if args.detailed_logging:
            print_info("Created standard protection order folder structure with detailed logging enabled")
        else:
            print_info("Created standard protection order folder structure")
            
        print(case_manager.get_folder_structure())
    else:
        print_error("Failed to initialize case workspace")
    
    return True

def handle_import(args, config):
    """Handle the import command"""
    case_manager = CaseManager(config)
    if not case_manager.is_valid_case_directory():
        print_error("Not in a valid PRISM case directory. Initialize a case first.")
        return False
    
    print_header(f"[PRISM v{VERSION} - Protection Order Module Activated]")
    print_info(f"Scanning directory {args.path}")
    
    # Perform file scanning and importing
    case_manager.import_evidence(
        args.path, 
        tags=args.tags.split(",") if args.tags else [], 
        case_type=args.case_type,
        chain_of_custody=args.chain_of_custody,
        detailed_scan=args.detailed_scan
    )
    
    return True

def handle_analyze_audio(args, config):
    """Handle the analyze-audio command"""
    case_manager = CaseManager(config)
    if not case_manager.is_valid_case_directory():
        print_error("Not in a valid PRISM case directory. Initialize a case first.")
        return False
    
    print_header("[PRISM Audio Analysis Module v2.8.1.42]")
    print_info("Initializing comprehensive audio analysis pipeline")
    
    audio_analyzer = AudioAnalyzer(config)
    audio_analyzer.analyze(
        args.path,
        transcribe=args.transcribe,
        emotional_markers=args.emotional_markers,
        speaker_id=args.speaker_id,
        vocal_stress=args.vocal_stress,
        detailed_analysis=args.detailed_analysis,
        quality_enhancement=args.quality_enhancement
    )
    
    return True

def handle_analyze_video(args, config):
    """Handle the analyze-video command"""
    case_manager = CaseManager(config)
    if not case_manager.is_valid_case_directory():
        print_error("Not in a valid PRISM case directory. Initialize a case first.")
        return False
    
    print_header("[PRISM Video Analysis Module v3.2.4.87]")
    print_info("Initializing comprehensive video analysis pipeline")
    
    video_analyzer = VideoAnalyzer(config)
    video_analyzer.analyze(
        args.path,
        object_detection=args.object_detection,
        facial_recognition=args.facial_recognition,
        incident_detection=args.incident_detection,
        extract_frames=args.extract_frames,
        technical_forensics=args.technical_forensics,
        background_removal=args.background_removal
    )
    
    return True

def handle_correlate_evidence(args, config):
    """Handle the correlate-evidence command"""
    case_manager = CaseManager(config)
    if not case_manager.is_valid_case_directory():
        print_error("Not in a valid PRISM case directory. Initialize a case first.")
        return False
    
    print_header("[PRISM Evidence Correlation Engine v2.9.3.64]")
    print_info("Initializing comprehensive evidence correlation")
    
    correlator = EvidenceCorrelator(config)
    correlator.correlate(
        modalities=args.modalities,
        timeline=args.timeline,
        threshold=args.threshold,
        output=args.output,
        detailed_analysis=args.detailed_analysis
    )
    
    return True

def handle_extract_multimedia_segments(args, config):
    """Handle the extract-multimedia-segments command"""
    case_manager = CaseManager(config)
    if not case_manager.is_valid_case_directory():
        print_error("Not in a valid PRISM case directory. Initialize a case first.")
        return False
    
    print_header("[PRISM Media Processing Module v2.2.1.35]")
    print_info(f"Loading configuration from {args.config}")
    
    extractor = MediaExtractor(config)
    extractor.extract_segments(
        config_file=args.config,
        output_dir=args.output,
        forensic_packaging=args.forensic_packaging
    )
    
    return True

def handle_generate_protection_order(args, config):
    """Handle the generate-protection-order command"""
    case_manager = CaseManager(config)
    if not case_manager.is_valid_case_directory():
        print_error("Not in a valid PRISM case directory. Initialize a case first.")
        return False
    
    print_header("[PRISM Legal Document Generation v3.5.2.128]")
    print_info("Initializing protection order document generation")
    
    generator = DocumentGenerator(config)
    generator.generate_protection_order(
        template=args.template,
        court=args.court,
        evidence_integration=args.evidence_integration,
        expedited=args.expedited,
        incident_details=args.incident_details
    )
    
    return True

def handle_generate_case_summary(args, config):
    """Handle the generate-case-summary command"""
    case_manager = CaseManager(config)
    if not case_manager.is_valid_case_directory():
        print_error("Not in a valid PRISM case directory. Initialize a case first.")
        return False
    
    print_header("[PRISM Case Documentation Module v2.7.2.53]")
    print_info("Generating comprehensive case summary with evidence assessment")
    
    generator = DocumentGenerator(config)
    generator.generate_case_summary(
        comprehensive=args.comprehensive,
        include_technical=args.include_technical,
        evidence_strengths=args.evidence_strengths
    )
    
    return True

def handle_processing_summary(args, config):
    """Handle the processing-summary command"""
    case_manager = CaseManager(config)
    if not case_manager.is_valid_case_directory():
        print_error("Not in a valid PRISM case directory. Initialize a case first.")
        return False
    
    print_header("[PRISM System Summary v1.3.4]")
    case_manager.show_processing_summary()
    
    return True

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(verbose=True)
    
    # Handle commands
    handlers = {
        "init": handle_init,
        "import": handle_import,
        "analyze-audio": handle_analyze_audio,
        "analyze-video": handle_analyze_video,
        "correlate-evidence": handle_correlate_evidence,
        "extract-multimedia-segments": handle_extract_multimedia_segments,
        "generate-protection-order": handle_generate_protection_order,
        "generate-case-summary": handle_generate_case_summary,
        "processing-summary": handle_processing_summary
    }
    
    if args.command in handlers:
        result = handlers[args.command](args, config)
        return 0 if result else 1
    else:
        print_error(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
