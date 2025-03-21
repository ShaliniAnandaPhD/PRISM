

import os
import json
import csv
import torch
import logging
import pandas as pd
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from pathlib import Path

logger = logging.getLogger("DatasetLoaders")

def load_msrvtt_annotations(annotations_path: str) -> List[Dict]:
    """
    Load annotations from MSR-VTT dataset.
    
    Args:
        annotations_path: Path to the MSR-VTT JSON file
        
    Returns:
        List of annotation dictionaries with 'video' and 'caption' keys
    """
    logger.info(f"Loading MSR-VTT annotations from {annotations_path}")
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"MSR-VTT annotations file not found: {annotations_path}")
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    annotations = []
    
    # Process videos
    video_id_to_name = {}
    for video in data['videos']:
        video_id = video['video_id']
        video_name = f"{video_id}.mp4"
        video_id_to_name[video_id] = video_name
    
    # Process captions
    for sentence in data['sentences']:
        video_id = sentence['video_id']
        caption = sentence['caption']
        
        if video_id in video_id_to_name:
            annotations.append({
                'video': video_id_to_name[video_id],
                'caption': caption
            })
    
    logger.info(f"Loaded {len(annotations)} MSR-VTT annotations")
    return annotations

def load_activitynet_annotations(annotations_path: str) -> List[Dict]:
    """
    Load annotations from ActivityNet Captions dataset.
    
    Args:
        annotations_path: Path to the ActivityNet JSON file
        
    Returns:
        List of annotation dictionaries with 'video' and 'caption' keys
    """
    logger.info(f"Loading ActivityNet annotations from {annotations_path}")
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"ActivityNet annotations file not found: {annotations_path}")
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    annotations = []
    
    # Process each video
    for video_id, video_data in data.items():
        if 'segments' not in video_data:
            continue
            
        video_name = f"{video_id}.mp4"
        
        # Process each segment
        for segment in video_data['segments']:
            caption = segment['sentence']
            
            annotations.append({
                'video': video_name,
                'caption': caption,
                'segment': segment['segment']  # Start and end times
            })
    
    logger.info(f"Loaded {len(annotations)} ActivityNet annotations")
    return annotations

def load_webvid_annotations(annotations_path: str) -> List[Dict]:
    """
    Load annotations from WebVid dataset.
    
    Args:
        annotations_path: Path to the WebVid CSV file
        
    Returns:
        List of annotation dictionaries with 'video' and 'caption' keys
    """
    logger.info(f"Loading WebVid annotations from {annotations_path}")
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"WebVid annotations file not found: {annotations_path}")
    
    # Read CSV file
    df = pd.read_csv(annotations_path)
    
    annotations = []
    
    # Process each row
    for _, row in df.iterrows():
        video_name = f"{row['videoid']}.mp4"
        caption = row['name'] if 'name' in row else row['title']
        
        annotations.append({
            'video': video_name,
            'caption': caption,
            'url': row.get('videoUrl', None)
        })
    
    logger.info(f"Loaded {len(annotations)} WebVid annotations")
    return annotations

def load_custom_annotations(annotations_path: str) -> List[Dict]:
    """
    Load annotations from a custom JSON file.
    
    Args:
        annotations_path: Path to the custom JSON file
        
    Returns:
        List of annotation dictionaries with 'video' and 'caption' keys
    """
    logger.info(f"Loading custom annotations from {annotations_path}")
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Custom annotations file not found: {annotations_path}")
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Validate the format
    for i, ann in enumerate(annotations[:10]):
        if 'video' not in ann or 'caption' not in ann:
            logger.warning(f"Annotation {i} missing required fields: {ann}")
    
    logger.info(f"Loaded {len(annotations)} custom annotations")
    return annotations

def get_dataset_loader(dataset_name: str, annotations_path: str) -> List[Dict]:
    """
    Get the appropriate loader function for a dataset.
    
    Args:
        dataset_name: Name of the dataset ('MSR-VTT', 'ActivityNet', 'WebVid', 'custom')
        annotations_path: Path to the annotations file
        
    Returns:
        List of annotation dictionaries
    """
    loaders = {
        'MSR-VTT': load_msrvtt_annotations,
        'ActivityNet': load_activitynet_annotations,
        'WebVid': load_webvid_annotations,
        'custom': load_custom_annotations
    }
    
    if dataset_name not in loaders:
        logger.warning(f"Unknown dataset name: {dataset_name}, falling back to custom loader")
        dataset_name = 'custom'
    
    return loaders[dataset_name](annotations_path)
