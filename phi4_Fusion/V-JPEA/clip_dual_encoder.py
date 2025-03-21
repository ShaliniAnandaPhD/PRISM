import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
import numpy as np
import logging
import json
from tqdm import tqdm
from pathlib import Path
import os

# Set up logging for better debugging and progress tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LatentTextMapping")

# Configuration parameters - these would typically go in a config file
# but are defined here for clarity
class Config:
    """
    Configuration for the latent-to-text mapping models.
    
    This defines all the parameters needed for the model architecture,
    training process, and data handling.
    """
    # Model architecture
    LATENT_DIM = 1024        # Dimension of V-JEPA's latent vectors
    HIDDEN_DIM = 512         # Internal dimension for our mapping networks
    TEXT_ENCODER_MODEL = 'xlm-roberta-base'  # Pretrained text model
    
    # Training parameters
    BATCH_SIZE = 64          # How many samples to process at once
    LEARNING_RATE = 2e-5     # How quickly the model learns
    WEIGHT_DECAY = 0.01      # Regularization to prevent overfitting
    NUM_EPOCHS = 20          # How many times to go through the training data
    WARMUP_STEPS = 100       # Gradual learning rate increase at start
    TEMPERATURE = 0.07       # Controls sharpness of similarity distribution
    
    # Data handling
    MAX_TEXT_LENGTH = 128    # Maximum number of tokens in text descriptions
    VALIDATION_SPLIT = 0.1   # Portion of data to use for validation
    
    # Filesystem
    CHECKPOINT_DIR = "./checkpoints"  # Where to save trained models
    RESULTS_DIR = "./results"         # Where to save evaluation results


class LatentEncoder(nn.Module):
    """
    Transforms V-JEPA's video latent vectors into a shared embedding space.
    
    This neural network takes the complex, high-dimensional representation
    of a video from V-JEPA and maps it into a simpler space where it can
    be directly compared with text embeddings.
    
    Think of this as translating "video understanding" into a format that
    can be matched with language.
    """
    def __init__(self, latent_dim=Config.LATENT_DIM, embed_dim=Config.HIDDEN_DIM):
        """
        Set up the neural network layers.
        
        Args:
            latent_dim: Dimension of the input latent vectors from V-JEPA
            embed_dim: Dimension of the output embeddings to match with text
        """
        super().__init__()
        
        # A more sophisticated architecture than the original:
        # - First reduces dimension with a larger layer
        # - Uses layer normalization for more stable training
        # - Applies dropout to prevent overfitting
        # - Has a residual connection to help training deep networks
        self.net = nn.Sequential(
            # First layer: reduce dimensions 
            nn.Linear(latent_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),  # Normalize activations
            nn.GELU(),  # Modern activation function (smoother than ReLU)
            nn.Dropout(0.1),  # Randomly turn off 10% of neurons during training
            
            # Second layer: further processing
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # Final layer: project to final dimension (with residual connection)
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Initialize weights properly for better training
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights for better training convergence"""
        if isinstance(module, nn.Linear):
            # Common initialization method for transformers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Process a batch of latent vectors.
        
        Args:
            x: Tensor of shape [batch_size, latent_dim] containing V-JEPA latents
            
        Returns:
            Tensor of shape [batch_size, embed_dim] containing mapped embeddings
        """
        return self.net(x)


class TextEncoder(nn.Module):
    """
    Encodes text descriptions into the shared embedding space.
    
    This uses a pre-trained language model (like XLM-RoBERTa) to understand
    the meaning of text descriptions, then maps those understandings into
    the same space as the video embeddings.
    
    The model supports multiple languages, making it useful for global
    applications of security video analysis.
    """
    def __init__(self, model_name=Config.TEXT_ENCODER_MODEL, max_length=Config.MAX_TEXT_LENGTH):
        """
        Set up the text encoder using a pre-trained model.
        
        Args:
            model_name: Which pretrained model to use (from Hugging Face)
            max_length: Maximum number of tokens in text descriptions
        """
        super().__init__()
        
        logger.info(f"Loading text encoder model: {model_name}")
        
        # Load the pretrained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        
        # Add a projection layer to map from the model's dimension to our embedding dimension
        # This helps ensure text and video embeddings have compatible dimensions
        model_dim = self.encoder.config.hidden_size
        self.projection = nn.Linear(model_dim, Config.HIDDEN_DIM)
        
    def encode_text(self, texts):
        """
        Convert text descriptions into embeddings.
        
        Args:
            texts: List of text descriptions (strings)
            
        Returns:
            Tensor of shape [batch_size, embed_dim] containing text embeddings
        """
        # Tokenize the texts (convert words to numbers the model understands)
        tokens = self.tokenizer(
            texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        ).to(self.encoder.device)
        
        # Run the texts through the language model
        with torch.no_grad():  # Don't compute gradients for the pretrained model
            outputs = self.encoder(**tokens)
        
        # Extract the [CLS] token embedding (special token that represents the whole text)
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # Project to our embedding dimension
        projected = self.projection(pooled)
        
        return projected


class CLIPStyleDualEncoder(nn.Module):
    """
    Main model that connects video latents to text descriptions.
    
    This model is inspired by OpenAI's CLIP architecture, which learns to
    match images with text. Here, we're matching V-JEPA video latents with
    text descriptions instead.
    
    The model is trained to maximize similarity between matching video-text
    pairs while minimizing similarity between non-matching pairs.
    """
    def __init__(self, embed_dim=Config.HIDDEN_DIM, temp=Config.TEMPERATURE):
        """
        Set up the dual encoder model.
        
        Args:
            embed_dim: Dimension of the shared embedding space
            temp: Temperature parameter for scaling similarity scores
        """
        super().__init__()
        
        # Sub-networks for encoding each modality
        self.latent_encoder = LatentEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder()
        
        # Learnable temperature parameter for scaling similarity scores
        # Lower values make the model more confident/certain
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temp))
        
        logger.info(f"Initialized dual encoder with embedding dimension {embed_dim}")
        
    def forward(self, latents, texts):
        """
        Process a batch of video latents and text descriptions.
        
        Args:
            latents: Tensor of shape [batch_size, latent_dim] with V-JEPA latents
            texts: List of text descriptions (strings)
            
        Returns:
            Tuple of similarity matrices (video-to-text and text-to-video)
        """
        # Encode both modalities into the shared embedding space
        latent_embeds = self.latent_encoder(latents)
        text_embeds = self.text_encoder.encode_text(texts)
        
        # Normalize the embeddings to use cosine similarity
        latent_embeds = latent_embeds / latent_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Calculate similarity scores between all pairs of videos and texts
        # Higher score means the video and text are more likely to match
        logit_scale = self.logit_scale.exp()
        logits_per_video = torch.matmul(latent_embeds, text_embeds.t()) * logit_scale
        logits_per_text = logits_per_video.t()
        
        return logits_per_video, logits_per_text


class LatentTextDataset(Dataset):
    """
    Dataset for training the latent-to-text mapping.
    
    This handles loading and preprocessing paired data of V-JEPA latents
    and their corresponding text descriptions.
    """
    def __init__(self, data_file, max_samples=None):
        """
        Initialize the dataset from a JSON data file.
        
        Args:
            data_file: Path to JSON file with latent-text pairs
            max_samples: Maximum number of samples to load (for debugging)
        """
        super().__init__()
        
        logger.info(f"Loading dataset from {data_file}")
        
        # Load the dataset
        with open(data_file, 'r') as f:
            self.data = json.load(f)
            
        # Optionally limit the number of samples
        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data[:max_samples]
            
        logger.info(f"Loaded {len(self.data)} latent-text pairs")
            
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary with 'latent' and 'text' keys
        """
        item = self.data[idx]
        
        # Convert the latent vector from list to tensor
        latent = torch.tensor(item['latent'], dtype=torch.float32)
        
        return {
            'latent': latent,
            'text': item['text']
        }


def contrastive_loss(logits_per_video, logits_per_text):
    """
    Calculate the contrastive loss used for training.
    
    This loss encourages the model to give high similarity scores to matching
    video-text pairs and low scores to non-matching pairs.
    
    Args:
        logits_per_video: Similarity scores from videos to texts
        logits_per_text: Similarity scores from texts to videos
        
    Returns:
        Average loss value for the batch
    """
    # Number of samples in the batch
    batch_size = logits_per_video.shape[0]
    
    # Labels are the diagonal indices (where video i matches with text i)
    labels = torch.arange(batch_size, device=logits_per_video.device)
    
    # Calculate loss in both directions and average them
    loss_video = nn.functional.cross_entropy(logits_per_video, labels)
    loss_text = nn.functional.cross_entropy(logits_per_text, labels)
    
    return (loss_video + loss_text) / 2.0


def train_model(model, train_dataloader, val_dataloader=None, config=Config):
    """
    Train the latent-to-text mapping model.
    
    Args:
        model: The CLIPStyleDualEncoder model to train
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation data
        config: Configuration parameters
        
    Returns:
        Trained model
    """
    logger.info("Starting model training")
    
    # Set up model for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    # Set up optimizer with weight decay (to prevent overfitting)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Set up learning rate scheduler for better training dynamics
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS * len(train_dataloader)
    )
    
    # Create directory for saving checkpoints
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        # Track metrics
        total_loss = 0
        correct_pairs = 0
        total_pairs = 0
        
        # Process each batch
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}"):
            # Move data to the right device
            latents = batch['latent'].to(device)
            texts = batch['text']
            
            # Zero out gradients from previous batch
            optimizer.zero_grad()
            
            # Forward pass
            logits_per_video, logits_per_text = model(latents, texts)
            
            # Calculate loss
            loss = contrastive_loss(logits_per_video, logits_per_text)
            
            # Backward pass
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate accuracy (how often the top prediction is correct)
            video_preds = logits_per_video.argmax(dim=1)
            text_preds = logits_per_text.argmax(dim=1)
            labels = torch.arange(logits_per_video.shape[0], device=device)
            
            correct_pairs += (video_preds == labels).sum().item()
            correct_pairs += (text_preds == labels).sum().item()
            total_pairs += 2 * logits_per_video.shape[0]
        
        # Calculate average metrics for the epoch
        avg_loss = total_loss / len(train_dataloader)
        accuracy = correct_pairs / total_pairs
        
        logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
                    f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Validation
        if val_dataloader is not None:
            val_loss, val_accuracy = evaluate_model(model, val_dataloader)
            logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")
        
        # Save checkpoint for the epoch
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }, checkpoint_path)
    
    logger.info("Training completed")
    return model


def evaluate_model(model, dataloader):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The trained model to evaluate
        dataloader: DataLoader for evaluation data
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    device = next(model.parameters()).device
    model.eval()
    
    total_loss = 0
    correct_pairs = 0
    total_pairs = 0
    
    with torch.no_grad():  # Don't compute gradients for evaluation
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move data to the right device
            latents = batch['latent'].to(device)
            texts = batch['text']
            
            # Forward pass
            logits_per_video, logits_per_text = model(latents, texts)
            
            # Calculate loss
            loss = contrastive_loss(logits_per_video, logits_per_text)
            total_loss += loss.item()
            
            # Calculate accuracy
            video_preds = logits_per_video.argmax(dim=1)
            text_preds = logits_per_text.argmax(dim=1)
            labels = torch.arange(logits_per_video.shape[0], device=device)
            
            correct_pairs += (video_preds == labels).sum().item()
            correct_pairs += (text_preds == labels).sum().item()
            total_pairs += 2 * logits_per_video.shape[0]
    
    # Calculate average metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_pairs / total_pairs
    
    model.train()  # Set model back to training mode
    return avg_loss, accuracy


def latent_to_text(model, latent, top_k=5):
    """
    Find the most similar text descriptions for a video latent.
    
    This function takes a trained model and a video latent vector,
    then finds the text descriptions in a database that best match
    the video content.
    
    Args:
        model: Trained CLIPStyleDualEncoder model
        latent: V-JEPA latent vector for a video
        top_k: Number of text descriptions to return
        
    Returns:
        List of top-k text descriptions
    """
    # TODO: Implement this using FAISS for efficient similarity search
    # in large databases of text descriptions
    
    # For now, this is a placeholder that would be implemented in the full system
    return ["Person enters through front door with normal pace and movement.",
            "Individual scans the room before proceeding further inside.",
            "Subject exhibits hesitation when approaching the entry point.",
            "Person carrying items enters the premises without unusual behavior.",
            "Individual tests multiple entry points before gaining access."]


def get_data_loaders(data_file, batch_size=Config.BATCH_SIZE, val_split=Config.VALIDATION_SPLIT):
    """
    Create data loaders for training and validation.
    
    Args:
        data_file: Path to JSON file with latent-text pairs
        batch_size: Number of samples in each batch
        val_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load the full dataset
    full_dataset = LatentTextDataset(data_file)
    
    # Split into train and validation sets
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader


def main():
    """
    Main function to demonstrate the workflow.
    """
    logger.info("Starting latent-to-text mapping workflow")
    
    # Initialize model
    model = CLIPStyleDualEncoder()
    
    # Example: Train the model (would use actual data in practice)
    # train_loader, val_loader = get_data_loaders("data/latent_text_pairs.json")
    # model = train_model(model, train_loader, val_loader)
    
    # Example: Use the model to convert a latent to text
    # latent = torch.randn(1, Config.LATENT_DIM)  # Example random latent
    # descriptions = latent_to_text(model, latent)
    # for desc in descriptions:
    #     print(f"- {desc}")
    
    logger.info("Workflow completed")


if __name__ == "__main__":
    main()
