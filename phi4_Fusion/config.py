"""
config.py - Configuration management for Phi-4 + PRISM fusion system

This module handles loading and validating configuration parameters
for the fusion model system.

Author: AI Legal Tech Team
Date: March 16, 2025
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import yaml


@dataclass
class ModelConfig:
    """Configuration parameters for the fusion model system."""
    
    # Model paths
    phi4_model_path: str
    prism_model_path: str
    lora_weights_path: Optional[str] = None
    
    # Document processing
    document_index_path: Optional[str] = None
    
    # Model parameters
    precision: str = "fp16"
    phi4_max_length: int = 4096
    prism_max_length: int = 2048
    
    # Generation parameters
    max_output_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Performance
    use_flash_attention: bool = True
    use_kv_cache: bool = True
    
    # Benchmarking
    benchmark_path: str = "benchmarks/data"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not os.path.exists(self.phi4_model_path):
            logging.warning(f"Phi-4 model path does not exist: {self.phi4_model_path}")
        
        if not os.path.exists(self.prism_model_path):
            logging.warning(f"PRISM model path does not exist: {self.prism_model_path}")
        
        if self.lora_weights_path and not os.path.exists(self.lora_weights_path):
            logging.warning(f"LoRA weights path does not exist: {self.lora_weights_path}")
        
        if self.document_index_path and not os.path.exists(self.document_index_path):
            logging.warning(f"Document index path does not exist: {self.document_index_path}")
        
        if self.precision not in ["fp16", "fp32", "int8"]:
            raise ValueError(f"Invalid precision: {self.precision}")


def load_config(config_path: str) -> ModelConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        ModelConfig object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Create ModelConfig object
        config = ModelConfig(**config_dict)
        
        logging.info(f"Loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise


def create_default_config(output_path: str):
    """
    Create a default configuration file.
    
    Args:
        output_path: Path to save the config file
    """
    default_config = {
        "phi4_model_path": "models/phi-4-mini",
        "prism_model_path": "models/prism-legal",
        "lora_weights_path": "models/lora/phi4_prism_lora.pt",
        "document_index_path": "data/document_index",
        "precision": "fp16",
        "phi4_max_length": 4096,
        "prism_max_length": 2048,
        "max_output_length": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "use_flash_attention": True,
        "use_kv_cache": True,
        "benchmark_path": "benchmarks/data"
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    logging.info(f"Created default configuration at {output_path}")


"""
SUMMARY:
- Provides configuration management for the fusion model system
- Uses dataclasses for type-safe configuration parameters
- Implements validation for configuration values
- Supports loading from YAML files
- Includes default configuration generation

TODO:
- Add support for environment variable overrides
- Implement configuration versioning
- Add schema validation for config files
- Support for multiple configuration profiles
- Add configuration encryption for sensitive values
"""
