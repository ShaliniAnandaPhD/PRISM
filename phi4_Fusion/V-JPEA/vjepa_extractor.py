import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision.io import read_video
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger("VJEPAExtractor")

class VJEPALatentExtractor:
    """
    Extracts latent representations from videos using V-JEPA.
    
    This class handles loading videos, preprocessing frames, and
    running them through the V-JEPA model to extract latent features.
    """
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cuda:0",
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        use_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the V-JEPA latent extractor.
        
        Args:
            model_path: Path to the pretrained V-JEPA model weights
            config_path: Path to the V-JEPA model configuration
            device: Device to run inference on
            patch_size: Patch size used by the V-JEPA model
            mask_ratio: Masking ratio for V-JEPA prediction
            use_cache: Whether to cache extracted latents
            cache_dir: Directory to store cached latents
        """
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Create cache directory if needed
        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Initializing V-JEPA latent extractor on {self.device}")
        
        # Load the V-JEPA model
        self.model = self._load_model(model_path, config_path)
        self.model.eval()
        
        logger.info("V-JEPA latent extractor initialized")
    
    def _load_model(self, model_path: str, config_path: str) -> torch.nn.Module:
        """
        Load the V-JEPA model from disk.
        
        Args:
            model_path: Path to the pretrained V-JEPA model weights
            config_path: Path to the V-JEPA model configuration
            
        Returns:
            Loaded V-JEPA model
        """
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"V-JEPA model weights not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"V-JEPA config not found: {config_path}")
        
        # TODO: Replace with actual V-JEPA model loading code
        # This is a placeholder - in a real implementation, we would load
        # the actual V-JEPA model using its specific loading function
        logger.warning("Using placeholder V-JEPA model - replace with actual implementation")
        
        # Dummy model for development purposes
        class DummyVJEPA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(1024, 1024)
                
            def forward(self, x):
                return self.backbone(x)
                
            def extract_features(self, video_tensor, mask_ratio=0.75):
                # Pretend to extract features
                batch_size = video_tensor.shape[0]
                # Return random latent vector of the appropriate dimension
                return torch.randn(batch_size, 1024)
        
        model = DummyVJEPA()
        
        # Move model to the appropriate device
        model = model.to(self.device)
        
        return model
    
    def load_video(self, video_path: str, target_frames: int = 32) -> torch.Tensor:
        """
        Load and preprocess a video for V-JEPA.
        
        Args:
            video_path: Path to the video file
            target_frames: Number of frames to extract
            
        Returns:
            Preprocessed video tensor [T, C, H, W]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Read the video file
            video, _, _ = read_video(video_path, pts_unit='sec')
            
            # Check if video is empty
            if video.shape[0] == 0:
                raise ValueError(f"Empty video: {video_path}")
            
            # Convert to float and normalize
            video = video.float() / 255.0
            
            # Subsample or pad frames to reach target_frames
            if video.shape[0] > target_frames:
                indices = torch.linspace(0, video.shape[0] - 1, target_frames).long()
                video = video[indices]
            elif video.shape[0] < target_frames:
                padding = target_frames - video.shape[0]
                last_frame = video[-1:].repeat(padding, 1, 1, 1)
                video = torch.cat([video, last_frame], dim=0)
            
            # Permute dimensions from [T, H, W, C] to [T, C, H, W]
            video = video.permute(0, 3, 1, 2)
            
            return video
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            raise
    
    def extract_latents(
        self, 
        video_path: str,
        use_cache: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Extract latent representations from a video using V-JEPA.
        
        Args:
            video_path: Path to the video file
            use_cache: Whether to use cached latents (overrides instance setting)
            
        Returns:
            Latent representation of the video
        """
        # Determine whether to use cache
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        # If caching is enabled, check if latents are already cached
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{Path(video_path).stem}_latent.pt"
            
            if cache_path.exists():
                try:
                    logger.debug(f"Loading cached latents for {video_path}")
                    return torch.load(cache_path)
                except Exception as e:
                    logger.warning(f"Failed to load cached latents: {e}, recomputing")
        
        # Load and preprocess the video
        video_tensor = self.load_video(video_path)
        
        # Move to the appropriate device
        video_tensor = video_tensor.to(self.device)
        
        # Extract latents using V-JEPA
        with torch.no_grad():
            latents = self.model.extract_features(
                video_tensor.unsqueeze(0),  # Add batch dimension
                mask_ratio=self.mask_ratio
            )
        
        # Cache the latents if caching is enabled
        if use_cache and self.cache_dir:
            try:
                torch.save(latents, cache_path)
                logger.debug(f"Cached latents to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache latents: {e}")
        
        return latents.squeeze(0)  # Remove batch dimension
    
    def batch_extract_latents(
        self,
        video_paths: List[str],
        batch_size: int = 8,
        use_cache: Optional[bool] = None
    ) -> List[torch.Tensor]:
        """
        Extract latents from multiple videos in batches.
        
        Args:
            video_paths: List of paths to video files
            batch_size: Number of videos to process at once
            use_cache: Whether to use cached latents
            
        Returns:
            List of latent representations
        """
        logger.info(f"Extracting latents from {len(video_paths)} videos")
        
        latents = []
        
        # Process videos in batches
        for i in tqdm(range(0, len(video_paths), batch_size), desc="Extracting latents"):
            # Get current batch
            batch_paths = video_paths[i:i+batch_size]
            batch_latents = []
            
            # Process each video in the batch
            for video_path in batch_paths:
                try:
                    latent = self.extract_latents(video_path, use_cache=use_cache)
                    batch_latents.append(latent)
                except Exception as e:
                    logger.error(f"Error extracting latents from {video_path}: {e}")
                    # Use a zero tensor as a placeholder for failed extractions
                    batch_latents.append(torch.zeros(1024, device=self.device))
            
            # Add batch latents to results
            latents.extend(batch_latents)
        
        logger.info(f"Extracted latents from {len(latents)} videos")
        return latents
