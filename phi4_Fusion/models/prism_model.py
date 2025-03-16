"""
prism_model.py - Implementation of the PRISM legal RAG model

This module provides the PRISM (Precise Retrieval and Inference System for
Legal Materials) model, a specialized legal domain retrieval-augmented
generation system.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)


class PRISMDocumentEmbedder:
    """Document embedding component for PRISM model."""
    
    def __init__(self, 
                model_path: str,
                device: str = "cuda",
                precision: str = "fp16",
                embed_dim: int = 768):
        """
        Initialize the document embedder.
        
        Args:
            model_path: Path to the embedder model
            device: Device to run model on (cuda or cpu)
            precision: Model precision (fp16, fp32, int8)
            embed_dim: Embedding dimension
        """
        self.device = device
        self.precision = precision
        self.embed_dim = embed_dim
        
        # Set precision
        torch_dtype = self._get_torch_dtype(precision)
        
        # Load embedder model and tokenizer
        logging.info(f"Loading PRISM document embedder from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        logging.info(f"Document embedder initialized on {device} with {precision} precision")
    
    def _get_torch_dtype(self, precision: str) -> torch.dtype:
        """Get torch dtype based on precision string."""
        if precision == "fp16":
            return torch.float16
        elif precision == "fp32":
            return torch.float32
        elif precision == "int8":
            return torch.int8
        else:
            raise ValueError(f"Unsupported precision: {precision}")
    
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Embed a text passage.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding tensor
        """
        # Tokenize
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def embed_passages(self, passages: List[str]) -> torch.Tensor:
        """
        Embed multiple text passages.
        
        Args:
            passages: List of text passages
            
        Returns:
            Tensor of embeddings
        """
        # Process in batches for memory efficiency
        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
        
        # Concatenate batches
        return torch.cat(all_embeddings, dim=0)


class PRISMRetriever:
    """Legal document retriever for PRISM model."""
    
    def __init__(self,
                embedder: PRISMDocumentEmbedder,
                index_path: Optional[str] = None,
                device: str = "cuda"):
        """
        Initialize the retriever.
        
        Args:
            embedder: Document embedder instance
            index_path: Path to pre-built document index
            device: Device to run model on (cuda or cpu)
        """
        self.embedder = embedder
        self.device = device
        
        # Document index (will be loaded from index_path)
        self.document_embeddings = None
        self.document_chunks = None
        self.document_metadata = None
        
        # Load index if provided
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
        
        logging.info("PRISM retriever initialized")
    
    def load_index(self, index_path: str):
        """
        Load pre-built document index.
        
        Args:
            index_path: Path to index directory
        """
        try:
            # Load embeddings
            embeddings_path = os.path.join(index_path, "embeddings.pt")
            self.document_embeddings = torch.load(embeddings_path, map_location=self.device)
            
            # Load document chunks
            chunks_path = os.path.join(index_path, "chunks.json")
            with open(chunks_path, "r") as f:
                self.document_chunks = json.load(f)
            
            # Load metadata
            metadata_path = os.path.join(index_path, "metadata.json")
            with open(metadata_path, "r") as f:
                self.document_metadata = json.load(f)
            
            logging.info(f"Loaded document index with {len(self.document_chunks)} chunks")
        except Exception as e:
            logging.error(f"Failed to load document index: {e}")
            raise
    
    def build_index(self, documents: Dict[str, str], output_path: Optional[str] = None):
        """
        Build a document index.
        
        Args:
            documents: Dictionary of {document_id: text}
            output_path: Optional path to save index
        """
        all_chunks = []
        all_metadata = []
        
        # Process each document
        for doc_id, text in documents.items():
            # Split document into chunks (simplified implementation)
            chunks = self._split_document(text)
            
            # Store chunks with metadata
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "document_id": doc_id,
                    "chunk_id": i,
                    "position": i / len(chunks)
                })
        
        # Embed all chunks
        logging.info(f"Embedding {len(all_chunks)} document chunks")
        embeddings = self.embedder.embed_passages(all_chunks)
        
        # Store index
        self.document_embeddings = embeddings
        self.document_chunks = all_chunks
        self.document_metadata = all_metadata
        
        # Save index if output path provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            
            # Save embeddings
            torch.save(embeddings, os.path.join(output_path, "embeddings.pt"))
            
            # Save chunks
            with open(os.path.join(output_path, "chunks.json"), "w") as f:
                json.dump(all_chunks, f)
            
            # Save metadata
            with open(os.path.join(output_path, "metadata.json"), "w") as f:
                json.dump(all_metadata, f)
            
            logging.info(f"Saved document index to {output_path}")
    
    def _split_document(self, text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
        """
        Split document into overlapping chunks.
        
        Args:
            text: Document text
            chunk_size: Target size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Simple character-based chunking (word-based would be better)
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 50:  # Skip very small chunks
                chunks.append(chunk)
        
        return chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with retrieved chunks and metadata
        """
        if self.document_embeddings is None:
            raise ValueError("Document index not loaded")
        
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        
        # Calculate similarity scores
        similarity_scores = torch.matmul(
            query_embedding, 
            self.document_embeddings.T
        ).squeeze()
        
        # Get top-k matches
        if top_k > similarity_scores.size(0):
            top_k = similarity_scores.size(0)
            
        top_scores, top_indices = torch.topk(similarity_scores, k=top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append({
                "text": self.document_chunks[idx],
                "metadata": self.document_metadata[idx],
                "score": score
            })
        
        return results
    
    def retrieve_from_document(self, document: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant sections from a document.
        
        Args:
            document: Document text
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with retrieved chunks and metadata
        """
        # Split document into chunks
        chunks = self._split_document(document)
        
        # Skip if no chunks
        if not chunks:
            return []
        
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        
        # Embed all chunks
        chunk_embeddings = self.embedder.embed_passages(chunks)
        
        # Calculate similarity scores
        similarity_scores = torch.matmul(
            query_embedding, 
            chunk_embeddings.T
        ).squeeze()
        
        # Get top-k matches
        if top_k > similarity_scores.size(0):
            top_k = similarity_scores.size(0)
            
        top_scores, top_indices = torch.topk(similarity_scores, k=top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append({
                "text": chunks[idx],
                "metadata": {
                    "chunk_id": idx,
                    "position": idx / len(chunks)
                },
                "score": score
            })
        
        return results


class PRISMReasoningModule(nn.Module):
    """Legal reasoning module for PRISM model."""
    
    def __init__(self, 
                model_path: str,
                hidden_size: int = 768):
        """
        Initialize the reasoning module.
        
        Args:
            model_path: Path to pre-trained reasoning module
            hidden_size: Hidden size dimension
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Simplified module structure
        self.context_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        
        self.query_projector = nn.Linear(hidden_size, hidden_size)
        self.context_projector = nn.Linear(hidden_size, hidden_size)
        
        # Load pre-trained weights if available
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path)
                self.load_state_dict(state_dict)
                logging.info(f"Loaded reasoning module weights from {model_path}")
            except Exception as e:
                logging.warning(f"Failed to load reasoning module weights: {e}")
    
    def forward(self, 
               query_embedding: torch.Tensor,
               context_embeddings: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process query and context through reasoning module.
        
        Args:
            query_embedding: Embedded query
            context_embeddings: Embedded context passages
            attention_mask: Optional attention mask
            
        Returns:
            Enhanced representation
        """
        # Project query and context
        query_proj = self.query_projector(query_embedding)
        context_proj = self.context_projector(context_embeddings)
        
        # Combine query with context
        # Repeat query for each context element
        batch_size, context_len, _ = context_embeddings.shape
        expanded_query = query_proj.unsqueeze(1).expand(-1, context_len, -1)
        
        # Concatenate along feature dimension
        combined = torch.cat([expanded_query, context_proj], dim=-1)
        
        # Process through transformer layer
        output = self.context_encoder(combined, src_key_padding_mask=attention_mask)
        
        # Aggregate (mean pooling)
        pooled = output.mean(dim=1)
        
        return pooled


class PRISMModel:
    """
    PRISM (Precise Retrieval and Inference System for Legal Materials) model.
    """
    
    def __init__(self,
                model_path: str,
                index_path: Optional[str] = None,
                device: str = "cuda",
                precision: str = "fp16"):
        """
        Initialize the PRISM model.
        
        Args:
            model_path: Path to model directory
            index_path: Path to document index
            device: Device to run model on (cuda or cpu)
            precision: Model precision (fp16, fp32, int8)
        """
        self.device = device
        self.precision = precision
        self.hidden_size = 768  # Default hidden size
        
        # Paths
        embedder_path = os.path.join(model_path, "embedder")
        reasoning_path = os.path.join(model_path, "reasoning.pt")
        
        # Initialize document embedder
        self.embedder = PRISMDocumentEmbedder(
            model_path=embedder_path,
            device=device,
            precision=precision
        )
        
        # Initialize retriever
        self.retriever = PRISMRetriever(
            embedder=self.embedder,
            index_path=index_path,
            device=device
        )
        
        # Initialize reasoning module
        self.reasoning = PRISMReasoningModule(
            model_path=reasoning_path,
            hidden_size=self.hidden_size
        ).to(device)
        
        # Set to evaluation mode
        self.reasoning.eval()
        
        logging.info(f"PRISM model initialized on {device} with {precision} precision")
    
    def retrieve_contexts(self, 
                         query: str,
                         document: Optional[str] = None,
                         max_contexts: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant contexts for a query.
        
        Args:
            query: User query
            document: Optional document text (if provided, search within this document)
            max_contexts: Maximum number of contexts to retrieve
            
        Returns:
            Dictionary with contexts and metadata
        """
        # Retrieve from document or index
        if document:
            retrieved = self.retriever.retrieve_from_document(
                document=document,
                query=query,
                top_k=max_contexts
            )
        else:
            retrieved = self.retriever.retrieve(
                query=query,
                top_k=max_contexts
            )
        
        # Format citations
        citations = []
        contexts = []
        
        for item in retrieved:
            # Add text to contexts
            contexts.append(item["text"])
            
            # Format citation
            citation = {}
            if "metadata" in item:
                citation["metadata"] = item["metadata"]
            if "score" in item:
                citation["relevance"] = f"{item['score']:.4f}"
            
            # Add snippet
            text = item["text"]
            if len(text) > 200:
                citation["snippet"] = text[:200] + "..."
            else:
                citation["snippet"] = text
                
            citations.append(citation)
        
        return {
            "contexts": contexts,
            "citations": citations,
            "query": query
        }
    
    def process_query(self,
                     query: str,
                     contexts: Optional[List[str]] = None,
                     output_hidden_states: bool = True,
                     output_attentions: bool = False) -> Dict[str, Any]:
        """
        Process a query through the PRISM model.
        
        Args:
            query: Text query
            contexts: Optional list of context passages
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention maps
            
        Returns:
            Dictionary with model outputs
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        
        # Process contexts if provided
        if contexts and len(contexts) > 0:
            # Embed contexts
            context_embeddings = self.embedder.embed_passages(contexts)
            
            # Reshape for batch processing
            context_embeddings = context_embeddings.unsqueeze(0)  # [1, num_contexts, hidden_size]
            
            # Create attention mask (all contexts are valid)
            attention_mask = torch.zeros(
                (1, len(contexts)), 
                dtype=torch.bool, 
                device=self.device
            )
            
            # Process through reasoning module
            with torch.no_grad():
                enhanced_embedding = self.reasoning(
                    query_embedding.unsqueeze(0),  # Add batch dimension
                    context_embeddings,
                    attention_mask
                )
            
            # Get attention maps if requested
            if output_attentions:
                # Note: In a real implementation, we would extract attention
                # maps from the reasoning module's transformer layers
                attentions = None
            else:
                attentions = None
                
        else:
            # No contexts provided, just use query embedding
            enhanced_embedding = query_embedding.unsqueeze(0)  # Add batch dimension
            attentions = None
        
        # Return results
        return {
            "hidden_states": enhanced_embedding,
            "attentions": attentions,
            "query_embedding": query_embedding
        }
    
    def cleanup(self):
        """Release resources when done with the model."""
        # No explicit cleanup needed for PyTorch models,
        # but we could add custom resource cleanup here if needed
        pass


"""
SUMMARY:
- Implements PRISM (Precise Retrieval and Inference System for Legal Materials)
- Provides specialized legal domain retrieval and reasoning
- Components include document embedder, retriever, and reasoning module
- Supports both indexed document search and ad-hoc document analysis
- Provides citation information for retrieved contexts

TODO:
- Implement PDF and docx document parsing
- Add support for legal citation parsing and standardization
- Enhance chunking strategy for legal documents
- Implement citation cross-referencing
- Add case law database integration
- Optimize memory usage for large document collections
- Add support for statute and regulation retrieval
"""
