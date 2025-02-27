"""
Long-term memory module implementation based on the Titans paper.

This module implements a neural long-term memory that learns to memorize
information at test time using a gradient-based approach with momentum
and adaptive forgetting mechanisms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union


@dataclass
class LongTermMemoryConfig:
    """Configuration for the long-term memory module."""
    vector_dim: int = 768
    memory_depth: int = 2  # Number of layers in the memory neural network
    hidden_dim: int = 1024
    learning_rate: float = 0.01
    momentum_factor: float = 0.9
    forget_rate: float = 0.05
    surprise_threshold: float = 0.6
    max_capacity: int = 10000


class LongTermMemory:
    """
    Neural long-term memory module that learns to memorize information
    at test time, as described in the Titans paper.
    
    The memory uses a surprise-based mechanism to decide what to store,
    with momentum to track importance over time and a forgetting mechanism
    to manage capacity.
    """
    
    def __init__(self, config: LongTermMemoryConfig):
        """
        Initialize the long-term memory module.
        
        Args:
            config: Configuration parameters for the memory module
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the neural network for the memory
        self.memory_layers = []
        
        # Input layer
        self.memory_layers.append(nn.Linear(config.vector_dim, config.hidden_dim))
        
        # Hidden layers
        for _ in range(config.memory_depth - 2):
            self.memory_layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
        
        # Output layer
        if config.memory_depth > 1:
            self.memory_layers.append(nn.Linear(config.hidden_dim, config.vector_dim))
        else:
            # If only one layer, connect input directly to output
            self.memory_layers[0] = nn.Linear(config.vector_dim, config.vector_dim)
            
        # Convert to ModuleList for proper parameter management
        self.memory_network = nn.ModuleList(self.memory_layers).to(self.device)
        
        # Storage for content and embeddings
        self.content_map = []  # List of (content, embedding) tuples
        
        # State tracking
        self.momentum = 0
        self.surprise_history = []
        
        # Initialize parameters with small random values
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize the parameters of the memory network."""
        for layer in self.memory_network:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def _forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the memory network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through the memory network
        """
        for i, layer in enumerate(self.memory_network):
            x = layer(x)
            # Apply SiLU activation to all but the last layer
            if i < len(self.memory_network) - 1:
                x = F.silu(x)
        return x
    
    def _compute_surprise(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute the surprise metric based on how different the input is
        from what the memory would produce.
        
        Args:
            embedding: Input embedding
            
        Returns:
            Surprise score
        """
        with torch.no_grad():
            # Get what the memory would output for this input
            memory_output = self._forward_pass(embedding)
            
            # Compute the error (surprise) as the L2 distance
            surprise = F.mse_loss(memory_output, embedding)
            
            return surprise
            
    def _update_memory(self, embedding: torch.Tensor, content: str) -> None:
        """
        Update the memory based on the surprise of the new input.
        
        Args:
            embedding: Input embedding
            content: Text content associated with the embedding
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32).to(self.device)
        
        # Compute surprise
        surprise = self._compute_surprise(embedding)
        self.surprise_history.append(float(surprise.item()))
        
        # Check if the input is surprising enough to store
        if surprise > self.config.surprise_threshold:
            # Update momentum with current surprise
            self.momentum = self.config.momentum_factor * self.momentum + (1 - self.config.momentum_factor) * surprise
            
            # Gradient-based update of memory network
            # We want the memory to better remember this embedding
            
            # Set requires_grad to True for the backward pass
            embedding_grad = embedding.clone().detach().requires_grad_(True)
            
            # Forward pass
            output = self._forward_pass(embedding_grad)
            
            # Loss is MSE between output and target embedding
            loss = F.mse_loss(output, embedding_grad)
            
            # Backward pass
            loss.backward()
            
            # Update memory parameters with gradient descent
            with torch.no_grad():
                for layer in self.memory_network:
                    # Apply forgetting mechanism (weight decay)
                    layer.weight.data *= (1 - self.config.forget_rate)
                    layer.bias.data *= (1 - self.config.forget_rate)
                    
                    # Update weights with gradient descent
                    layer.weight.data -= self.config.learning_rate * layer.weight.grad
                    layer.bias.data -= self.config.learning_rate * layer.bias.grad
                    
                    # Zero gradients
                    layer.weight.grad.zero_()
                    layer.bias.grad.zero_()
            
            # Store the content and its embedding
            self.content_map.append((content, embedding.detach().cpu().numpy()))
            
            # Manage capacity
            if len(self.content_map) > self.config.max_capacity:
                # Remove oldest entries
                self.content_map = self.content_map[-self.config.max_capacity:]
    
    def retrieve(self, query_embedding: Union[np.ndarray, torch.Tensor], top_k: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant items from memory for a query.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of items to retrieve
            
        Returns:
            List of retrieved items with their relevance scores
        """
        if len(self.content_map) == 0:
            return []
            
        # Convert query to torch tensor if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32).to(self.device)
            
        results = []
        
        # Pass the query through the memory network to get a "memory-augmented" representation
        with torch.no_grad():
            memory_enhanced_query = self._forward_pass(query_embedding)
            memory_enhanced_query_np = memory_enhanced_query.cpu().numpy()
        
        # Compare with stored embeddings to find the most relevant items
        similarities = []
        for content, stored_embedding in self.content_map:
            # Compute cosine similarity
            similarity = np.dot(memory_enhanced_query_np, stored_embedding) / (
                np.linalg.norm(memory_enhanced_query_np) * np.linalg.norm(stored_embedding)
            )
            similarities.append((content, similarity))
            
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        for content, similarity in similarities[:top_k]:
            results.append({
                "content": content,
                "score": float(similarity),
                "source": "long_term"
            })
            
        return results
    
    def add(self, content: str, embedding: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Add a new item to the memory.
        
        Args:
            content: Text content to store
            embedding: Embedding of the content
        """
        # Update the memory with the new item
        self._update_memory(embedding, content)
        
    def clear(self) -> None:
        """Clear the memory."""
        self.content_map = []
        self.surprise_history = []
        self.momentum = 0
        self._init_parameters()
        
    def get_stats(self) -> Dict:
        """
        Get statistics about the memory.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            "size": len(self.content_map),
            "avg_surprise": np.mean(self.surprise_history) if self.surprise_history else 0,
            "max_surprise": np.max(self.surprise_history) if self.surprise_history else 0,
            "forget_rate": self.config.forget_rate,
            "momentum": self.momentum
        }
