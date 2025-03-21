#Bridging CLASS TO v-jepa

import os
import torch
import numpy as np
from typing import Union
from torchvision.io import read_video

# Assumes JEPA modules are locally available or pip-installed from jepa repo
from app.vjepa import load_model_from_config, run_forward_pass
from src.utils.visual_decoder import decode_latents_to_text  # optional text decoder


class VJEPAtoPRISM:
    """
    Bridge class to integrate V-JEPA with PRISM Fusion pipeline.
    This class extracts latent representations from videos and optionally decodes them into natural language.
    """

    def __init__(self, config_path: str, device: str = "cuda:0"):
        """
        Initializes V-JEPA encoder and predictor from config file.
        
        Args:
            config_path: Path to JEPA YAML config file (e.g., 'configs/pretrain/vitl16.yaml')
            device: Device to load model onto (e.g., 'cuda:0' or 'cpu')
        """
        self.device = torch.device(device)
        self.model = load_model_from_config(config_path=config_path, device=self.device)
        self.model.eval()

    def load_video(self, video_path: str) -> torch.Tensor:
        """
        Loads and preprocesses a video into a tensor format compatible with V-JEPA.

        Args:
            video_path: Path to video file (.mp4, .webm, etc.)

        Returns:
            video_tensor: Preprocessed video tensor with shape [T, H, W, C]
        """
        video, _, _ = read_video(video_path, pts_unit='sec')  # returns video as [T, H, W, C] uint8
        video = video.float() / 255.0  # normalize to [0, 1]
        return video.to(self.device)

    def extract_latents(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Runs the video tensor through V-JEPA to obtain feature-space predictions.

        Args:
            video_tensor: Preprocessed video tensor [T, H, W, C]

        Returns:
            latents: Feature predictions from predictor model
        """
        with torch.no_grad():
            latents = run_forward_pass(self.model, video_tensor)
        return latents

    def latents_to_text(self, latents: torch.Tensor) -> str:
        """
        Optionally decode latent features into natural language using a trained decoder (e.g., diffusion model).

        Args:
            latents: Latent tensor from V-JEPA

        Returns:
            A natural language description of the video
        """
        return decode_latents_to_text(latents)

    def video_to_context(self, video_path: str, as_text: bool = True) -> Union[torch.Tensor, str]:
        """
        Master method to load video, run through V-JEPA, and return output for PRISM.

        Args:
            video_path: File path to the input video
            as_text: If True, returns a natural language summary; else returns latent tensor

        Returns:
            Either a text string or raw latent tensor
        """
        video_tensor = self.load_video(video_path)
        latents = self.extract_latents(video_tensor)
        if as_text:
            return self.latents_to_text(latents)
        return latents
