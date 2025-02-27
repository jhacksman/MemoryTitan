"""
Short-term memory module implementation based on the Titans paper.

This module implements an attention-based short-term memory
that serves as a fixed-size sliding window over recent inputs.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union


@dataclass
class ShortTermMemoryConfig:
    """Configuration for the short-term memory module."""
    window_size: int = 512  # Number of items to keep in short-term memory
    vector_dim: int = 768


class ShortTermMemory:
    """
    Short-term memory implementation based on a fixed-size sliding window.
    
    This memory acts as a precise, attention-based mechanism for recent context,
    maintaining a fixed number of most recent items.
    """
    
    def __init__(self, config: ShortTermMemoryConfig):
        """
        Initialize the short-term memory module.
        
        Args:
            config: Configuration parameters for the memory module
        """
        self.config = config
        
        # Storage for content and embeddings in FIFO order
        self.content_buffer = []  # List of (content, embedding) tuples
    
    def add(self, content: str, embedding: np.ndarray) -> None:
        """
        Add a new item to the short-term memory.
        
        Args:
            content: Text content to store
            embedding: Embedding of the content
        """
        # Add to buffer
        self.content_buffer.append((content, embedding))
        
        # Maintain fixed size by removing oldest items if needed
        if len(self.content_buffer) > self.config.window_size:
            self.content_buffer = self.content_buffer[-self.config.window_size:]
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant items from short-term memory for a query.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of items to retrieve
            
        Returns:
            List of retrieved items with their relevance scores
        """
        if len(self.content_buffer) == 0:
            return []
            
        results = []
        
        # Compute similarities between query and all stored embeddings
        similarities = []
        for content, stored_embedding in self.content_buffer:
            # Compute cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            similarities.append((content, similarity))
            
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        for content, similarity in similarities[:top_k]:
            results.append({
                "content": content,
                "score": float(similarity),
                "source": "short_term"
            })
            
        return results
    
    def get_all_embeddings(self) -> np.ndarray:
        """
        Get all embeddings in the short-term memory.
        
        Returns:
            Array of all embeddings
        """
        if len(self.content_buffer) == 0:
            return np.array([])
            
        return np.vstack([embedding for _, embedding in self.content_buffer])
    
    def get_all_content(self) -> List[str]:
        """
        Get all content in the short-term memory.
        
        Returns:
            List of all content strings
        """
        return [content for content, _ in self.content_buffer]
    
    def clear(self) -> None:
        """Clear the short-term memory."""
        self.content_buffer = []
        
    def get_stats(self) -> Dict:
        """
        Get statistics about the memory.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            "size": len(self.content_buffer),
            "capacity": self.config.window_size,
            "utilization": len(self.content_buffer) / self.config.window_size if self.config.window_size > 0 else 0
        }
