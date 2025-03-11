"""
PRISM - Utility Functions
Shared utility functions for the PRISM system.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    """Set up logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure rich handler for console output
    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    rich_handler.setLevel(log_level)
    
    # Configure formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Configure file handler if log_file is provided
    handlers = [rich_handler]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers
    )

def print_header(message: str):
    """Print a header message"""
    console.print(message, style="bold purple")

def print_success(message: str):
    """Print a success message"""
    console.print(f"✓ {message}", style="bold green")

def print_info(message: str):
    """Print an info message"""
    console.print(f"ℹ {message}", style="bold cyan")

def print_warning(message: str):
    """Print a warning message"""
    console.print(f"⚠ {message}", style="bold yellow")

def print_error(message: str):
    """Print an error message"""
    console.print(f"! {message}", style="bold red")

def print_section(title: str):
    """Print a section title"""
    console.print(title, style="bold magenta")

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML file"""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print_error(f"Failed to load YAML file: {file_path}")
        print_error(str(e))
        return {}

def save_yaml(data: Dict[str, Any], file_path: str) -> bool:
    """Save data to YAML file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        return True
    except Exception as e:
        print_error(f"Failed to save YAML file: {file_path}")
        print_error(str(e))
        return False

def create_directory(path: str) -> bool:
    """Create directory if it doesn't exist"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print_error(f"Failed to create directory: {path}")
        print_error(str(e))
        return False

def create_progress_bar() -> Progress:
    """Create a progress bar"""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.percentage:>3.0f}%"),
        TimeElapsedColumn()
    )

def get_timestamp() -> str:
    """Get current timestamp in standard format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def format_time_interval(seconds: float) -> str:
    """Format time interval in mm:ss.xx format"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{int(seconds):02d}.{int((seconds - int(seconds)) * 100):02d}"

def calculate_sha256(file_path: str) -> str:
    """Calculate SHA-256 hash of a file"""
    import hashlib
    
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print_error(f"Failed to calculate SHA-256 hash for {file_path}: {str(e)}")
        return ""

def get_file_extension(file_path: str) -> str:
    """Get file extension in lowercase"""
    return os.path.splitext(file_path)[1].lower()

def is_audio_file(file_path: str) -> bool:
    """Check if file is an audio file"""
    audio_extensions = ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg']
    return get_file_extension(file_path) in audio_extensions

def is_video_file(file_path: str) -> bool:
    """Check if file is a video file"""
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv']
    return get_file_extension(file_path) in video_extensions

def is_image_file(file_path: str) -> bool:
    """Check if file is an image file"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    return get_file_extension(file_path) in image_extensions

def is_document_file(file_path: str) -> bool:
    """Check if file is a document file"""
    document_extensions = ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt']
    return get_file_extension(file_path) in document_extensions

def validate_evidence_file(file_path: str) -> Dict[str, Any]:
    """Validate an evidence file and extract basic metadata"""
    result = {
        "path": file_path,
        "filename": os.path.basename(file_path),
        "size": os.path.getsize(file_path),
        "modified_time": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S"),
        "checksum": calculate_sha256(file_path),
        "type": None,
        "valid": True,
        "error": None
    }
    
    # Determine file type
    if is_audio_file(file_path):
        result["type"] = "audio"
    elif is_video_file(file_path):
        result["type"] = "video"
    elif is_image_file(file_path):
        result["type"] = "image"
    elif is_document_file(file_path):
        result["type"] = "document"
    else:
        result["type"] = "other"
    
    # Check if file is valid
    try:
        with open(file_path, 'rb') as f:
            # Just read a small part to verify the file is accessible
            f.read(1024)
    except Exception as e:
        result["valid"] = False
        result["error"] = str(e)
    
    return result

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} {unit}"
