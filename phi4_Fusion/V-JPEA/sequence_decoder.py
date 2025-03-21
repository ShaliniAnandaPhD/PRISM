
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

class SequenceLatentToTextModel(nn.Module):
    """
    Advanced model that generates text sequences from V-JEPA latents.
    
    This model projects V-JEPA latents into the embedding space of a
    pretrained language model, then uses that model to generate
    coherent text descriptions.
    """
    def __init__(
        self,
        latent_dim=1024,
        model_name="gpt2",
        num_layers=4,
        hidden_dim=512,
        dropout=0.1
    ):
        super().__init__()
        
        # Load pretrained model configuration
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Create projection from V-JEPA latent space to language model embedding space
        self.latent_projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.config.hidden_size)
        )
        
        # Load pretrained language model
        self.lm = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Freeze most of the language model weights except the last few layers
        for param in self.lm.parameters():
            param.requires_grad = False
            
        # Unfreeze the last few transformer layers for fine-tuning
        for i in range(1, num_layers + 1):
            for param in self.lm.transformer.h[-i].parameters():
                param.requires_grad = True
        
        # Always unfreeze the language model head
        for param in self.lm.lm_head.parameters():
            param.requires_grad = True
            
    def forward(self, latents, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for training or inference.
        
        Args:
            latents: [batch_size, latent_dim] tensor of V-JEPA latents
            input_ids: [batch_size, seq_len] tensor of input token ids
            attention_mask: [batch_size, seq_len] attention mask
            labels: [batch_size, seq_len] target token ids
            
        Returns:
            Model outputs for causal language modeling
        """
        # Project latents to language model hidden dimension
        latent_embeddings = self.latent_projector(latents)  # [batch_size, hidden_size]
        
        # Expand to sequence dimension
        latent_embeddings = latent_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Prepare language model inputs
        if input_ids is not None:
            # For training, we prepend the latent embedding to the token embeddings
            # Get token embeddings from the model's embedding layer
            token_embeddings = self.lm.transformer.wte(input_ids)  # [batch_size, seq_len, hidden_size]
            
            # Concatenate latent embeddings with token embeddings along sequence dimension
            combined_embeddings = torch.cat([latent_embeddings, token_embeddings[:, :-1]], dim=1)
            
            # Create new attention mask that includes the latent embedding
            if attention_mask is not None:
                # Add a column of 1s for the latent embedding
                latent_attention = torch.ones((attention_mask.shape[0], 1), 
                                             device=attention_mask.device)
                combined_attention_mask = torch.cat([latent_attention, attention_mask[:, :-1]], dim=1)
            else:
                combined_attention_mask = None
            
            # Shift labels to align with the combined inputs
            if labels is not None:
                combined_labels = torch.cat([
                    # First token after latent should be predicted from latent
                    labels[:, :1],  
                    # Rest of sequence is predicted normally
                    labels[:, 1:],  
                ], dim=1)
            else:
                combined_labels = None
            
            # Forward pass through language model with combined embeddings
            outputs = self.lm(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                labels=combined_labels,
                return_dict=True
            )
            
            return outputs
            
        else:
            # For inference, we generate text conditioned on the latent embedding
            return self.generate_from_latent(latents)
    
    def generate_from_latent(
        self,
        latents,
        max_length=64,
        num_beams=5,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
    ):
        """
        Generate text from latent vectors.
        
        Args:
            latents: [batch_size, latent_dim] tensor of V-JEPA latents
            max_length: Maximum length of generated sequences
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            
        Returns:
            Generated text sequences
        """
        batch_size = latents.shape[0]
        
        # Project latents to language model hidden dimension
        latent_embeddings = self.latent_projector(latents)  # [batch_size, hidden_size]
        
        # Expand to sequence dimension
        latent_embeddings = latent_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Prepare generation parameters
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.lm.config.pad_token_id or self.lm.config.eos_token_id,
            "bos_token_id": self.lm.config.bos_token_id,
            "eos_token_id": self.lm.config.eos_token_id,
        }
        
        # Generate text from the latent embeddings
        output_ids = self.lm.generate(
            inputs_embeds=latent_embeddings,
            **gen_kwargs
        )
        
        return output_ids
