"""
Persistent memory module implementation based on the Titans paper.

This module implements a task-specific persistent memory with
learnable but data-independent parameters.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union


@dataclass
class PersistentMemoryConfig:
    """Configuration for the persistent memory module."""
    num_vectors: int = 64  # Number of persistent memory vectors
    vector_dim: int = 768  # Dimension of each vector
    init_scale: float = 0.1  # Scale for initialization


class PersistentMemory:
    """
    Persistent memory implementation based on learnable but data-independent parameters.
    
    This memory stores task-specific knowledge that doesn't change during inference
    but helps ground the processing of new information.
    """
    
    def __init__(self, config: PersistentMemoryConfig, pretrained_embeddings: Optional[np.ndarray] = None):
        """
        Initialize the persistent memory module.
        
        Args:
            config: Configuration parameters for the memory module
            pretrained_embeddings: Optional pretrained embeddings to use instead of random initialization
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize memory vectors
        if pretrained_embeddings is not None:
            # Use provided embeddings
            assert pretrained_embeddings.shape == (config.num_vectors, config.vector_dim), \
                f"Expected shape {(config.num_vectors, config.vector_dim)}, got {pretrained_embeddings.shape}"
            
            self.embeddings = torch.tensor(pretrained_embeddings, dtype=torch.float32).to(self.device)
        else:
            # Initialize with random values
            self.embeddings = torch.randn(
                config.num_vectors, config.vector_dim, device=self.device
            ) * config.init_scale
            
            # Normalize vectors to unit length
            self.embeddings = nn.functional.normalize(self.embeddings, p=2, dim=1)
        
        # Generate some generic descriptions for these memory vectors
        self.descriptions = [f"Persistent memory vector {i}" for i in range(config.num_vectors)]
    
    def retrieve(self, query_embedding: Union[np.ndarray, torch.Tensor], top_k: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant items from persistent memory for a query.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of items to retrieve
            
        Returns:
            List of retrieved items with their relevance scores
        """
        # Convert query to torch tensor if needed
        if isinstance(query_embedding, np.ndarray):
            query_tensor = torch.tensor(query_embedding, dtype=torch.float32).to(self.device)
        else:
            query_tensor = query_embedding.to(self.device)
            
        # Normalize query
        query_tensor = nn.functional.normalize(query_tensor, p=2, dim=0)
        
        # Compute cosine similarities
        similarities = torch.matmul(self.embeddings, query_tensor).cpu().numpy()
        
        # Sort indices by similarity (descending)
        sorted_indices = np.argsort(-similarities)
        
        # Return top-k results
        results = []
        for idx in sorted_indices[:top_k]:
            results.append({
                "content": self.descriptions[idx],
                "score": float(similarities[idx]),
                "source": "persistent",
                "vector_idx": int(idx)
            })
            
        return results
    
    def get_all_embeddings(self) -> np.ndarray:
        """
        Get all embeddings in the persistent memory.
        
        Returns:
            Array of all embeddings
        """
        return self.embeddings.cpu().numpy()
    
    def update_descriptions(self, descriptions: List[str]) -> None:
        """
        Update the descriptions for persistent memory vectors.
        
        Args:
            descriptions: New descriptions to use
        """
        assert len(descriptions) == self.config.num_vectors, \
            f"Expected {self.config.num_vectors} descriptions, got {len(descriptions)}"
        
        self.descriptions = descriptions
        
    def get_stats(self) -> Dict:
        """
        Get statistics about the memory.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            "size": self.config.num_vectors,
            "vector_dim": self.config.vector_dim
        }
