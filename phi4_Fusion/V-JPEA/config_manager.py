import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("ConfigManager")

@dataclass
class VJEPAConfig:
    model_path: str = "models/vjepa_base.pth"
    config_path: str = "configs/vjepa_base.yaml"
    device: str = "cuda:0"
    mask_ratio: float = 0.75
    patch_size: int = 16
    embed_dim: int = 1024
    depth: int = 12
    num_heads: int = 16
    
@dataclass
class DecoderConfig:
    model_type: str = "sequence"  # "simple" or "sequence"
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    max_length: int = 64
    beam_size: int = 5
    temperature: float = 1.0
    
@dataclass
class DataConfig:
    dataset_name: str = "MSR-VTT"  # Options: MSR-VTT, ActivityNet, WebVid, custom
    data_root: str = "data/videos"
    annotations_path: str = "data/annotations.json"
    cache_dir: str = "data/cache"
    val_split: float = 0.1
    use_cache: bool = True
    
@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    checkpoint_dir: str = "checkpoints"
    output_model_path: str = "vjepa_decoder.pt"
    val_frequency: int = 1
    patience: int = 3
    gradient_clip: float = 1.0
    use_mixed_precision: bool = True
    
@dataclass
class Config:
    vjepa: VJEPAConfig = field(default_factory=VJEPAConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def save(self, config_path):
        """Save configuration to a YAML file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_path}")
    
    @classmethod
    def load(cls, config_path):
        """Load configuration from a YAML file"""
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
