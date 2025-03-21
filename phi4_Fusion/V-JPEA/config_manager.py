#  Central configuration system for V-JEPA to PRISM integration

import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

"""
SUMMARY:

This module provides a centralized configuration system for the entire V-JEPA to PRISM 
integration pipeline. It uses Python dataclasses to organize settings into logical groups:
- VJEPA settings (model paths, dimensions)
- Text decoder settings (model type, dimensions)
- Dataset settings (which dataset to use, file paths)
- Training settings (batch size, learning rate, etc.)

The main Config class includes methods to save and load these settings from YAML files,
making it easy to reproduce experiments and maintain consistent settings across
different runs of the pipeline.

TODO:

- TODO: Add validation for configuration parameters to catch errors early
- TODO: Add support for command-line overrides of configuration values
- TODO: Include more detailed documentation for each parameter
- TODO: Add experiment tracking integration (e.g., with MLflow or Weights & Biases)
- TODO: Create pre-configured settings for different common use cases
- TODO: Add support for loading multiple configuration files and merging them
"""

logger = logging.getLogger("ConfigManager")

# V-JEPA model configuration settings
@dataclass
class VJEPAConfig:
    """Configuration for the V-JEPA video understanding model"""
    model_path: str = "models/vjepa_base.pth"
    config_path: str = "configs/vjepa_base.yaml"
    device: str = "cuda:0"    # Use CPU if no GPU is available
    mask_ratio: float = 0.75  # How much of the video to mask during prediction
    patch_size: int = 16      # Size of video patches for processing
    embed_dim: int = 1024     # Dimension of latent space
    depth: int = 12           # Number of transformer layers
    num_heads: int = 16       # Number of attention heads
    
# Text decoder configuration settings
@dataclass
class DecoderConfig:
    """Configuration for the text generation model"""
    model_type: str = "sequence"  # "simple" or "sequence"
    hidden_dim: int = 512         # Size of hidden layers
    num_layers: int = 4           # Number of transformer layers
    dropout: float = 0.1          # Dropout rate for regularization
    max_length: int = 64          # Maximum text generation length
    beam_size: int = 5            # Beam size for beam search generation
    temperature: float = 1.0      # Temperature for controlling generation randomness
    
# Dataset configuration settings
@dataclass
class DataConfig:
    """Configuration for dataset loading and processing"""
    dataset_name: str = "MSR-VTT"  # Which video-caption dataset to use
    data_root: str = "data/videos" # Directory containing videos
    annotations_path: str = "data/annotations.json"
    cache_dir: str = "data/cache"  # Where to cache processed data
    val_split: float = 0.1         # Portion of data to use for validation
    use_cache: bool = True         # Whether to use cached processed data
    
# Training configuration settings
@dataclass
class TrainingConfig:
    """Configuration for model training"""
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01     # Regularization strength
    num_epochs: int = 10
    warmup_ratio: float = 0.1      # Portion of training for learning rate warmup
    checkpoint_dir: str = "checkpoints"
    output_model_path: str = "vjepa_decoder.pt"
    val_frequency: int = 1         # Validate every N epochs
    patience: int = 3              # Early stopping patience
    gradient_clip: float = 1.0     # Maximum gradient norm
    use_mixed_precision: bool = True  # Whether to use mixed precision training
    
# Main configuration class that contains all the above
@dataclass
class Config:
    """
    Main configuration class that combines all configuration groups.
    
    This class can load/save configurations from/to YAML files, making it
    easy to reproduce experiments and share configurations.
    """
    vjepa: VJEPAConfig = field(default_factory=VJEPAConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def save(self, config_path):
        """
        Save configuration to a YAML file
        
        TODO: Add option to save only non-default values
        TODO: Add versioning for configuration files
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_path}")
    
    @classmethod
    def load(cls, config_path):
        """
        Load configuration from a YAML file
        
        TODO: Add validation of loaded values
        TODO: Add backward compatibility for older config formats
        """
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        
        # Update nested configurations
        if 'vjepa' in config_dict:
            config.vjepa = VJEPAConfig(**config_dict['vjepa'])
        if 'decoder' in config_dict:
            config.decoder = DecoderConfig(**config_dict['decoder'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
