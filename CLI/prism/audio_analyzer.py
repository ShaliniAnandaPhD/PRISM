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
