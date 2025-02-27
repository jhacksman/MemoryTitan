"""
Embedders for converting text to vector representations.

This module provides different embedding methods that can be used 
with the Titans memory system.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union, Any

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class BaseEmbedder(ABC):
    """Base class for text embedders."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into vectors.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embedding vectors
        """
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using sentence-transformers models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentence-transformers embedder.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Please install it with `pip install sentence-transformers`"
            )
            
        self.model = SentenceTransformer(model_name)
        
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using sentence-transformers.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embedding vectors
        """
        return self.model.encode(texts, normalize_embeddings=True)


class SimpleAverageEmbedder(BaseEmbedder):
    """
    Simple embedder that averages word vectors.
    
    This is a fallback embedder that doesn't require additional dependencies.
    It provides lower quality embeddings but is useful for testing.
    """
    
    def __init__(self, vocab_size: int = 10000, vector_dim: int = 768, seed: int = 42):
        """
        Initialize the simple average embedder.
        
        Args:
            vocab_size: Size of the vocabulary
            vector_dim: Dimension of the embedding vectors
            seed: Random seed for reproducibility
        """
        # Create a random embedding matrix
        np.random.seed(seed)
        self.embedding_matrix = np.random.randn(vocab_size, vector_dim).astype(np.float32)
        self.embedding_matrix = self.embedding_matrix / np.linalg.norm(
            self.embedding_matrix, axis=1, keepdims=True
        )
        
        # Simple tokenization vocabulary (just lowercase words)
        self.word_to_id = {}
        
    def _tokenize(self, text: str) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        # Simple word tokenization
        words = text.lower().split()
        
        # Convert to IDs, adding unknown words to vocabulary
        ids = []
        for word in words:
            if word not in self.word_to_id:
                # If vocabulary is full, reuse existing IDs
                if len(self.word_to_id) < self.embedding_matrix.shape[0]:
                    self.word_to_id[word] = len(self.word_to_id)
                else:
                    # Hash to an existing ID
                    self.word_to_id[word] = hash(word) % self.embedding_matrix.shape[0]
                    
            ids.append(self.word_to_id[word])
            
        return ids
        
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts by averaging word vectors.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            # Tokenize
            token_ids = self._tokenize(text)
            
            if not token_ids:
                # Empty text, return zero vector
                embeddings.append(np.zeros(self.embedding_matrix.shape[1], dtype=np.float32))
                continue
                
            # Average word vectors
            text_embedding = np.mean(self.embedding_matrix[token_ids], axis=0)
            
            # Normalize
            norm = np.linalg.norm(text_embedding)
            if norm > 0:
                text_embedding = text_embedding / norm
                
            embeddings.append(text_embedding)
            
        return np.array(embeddings)


class CachedEmbedder(BaseEmbedder):
    """
    Wrapper for an embedder that caches results to avoid recomputing embeddings.
    """
    
    def __init__(self, base_embedder: BaseEmbedder, cache_size: int = 10000):
        """
        Initialize the cached embedder.
        
        Args:
            base_embedder: The embedder to wrap
            cache_size: Maximum number of entries to cache
        """
        self.base_embedder = base_embedder
        self.cache_size = cache_size
        self.cache = {}
        
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts with caching.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embedding vectors
        """
        # Find which texts are not in cache
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text not in self.cache:
                uncached_texts.append(text)
                uncached_indices.append(i)
                
        # If there are uncached texts, compute their embeddings
        if uncached_texts:
            uncached_embeddings = self.base_embedder.embed(uncached_texts)
            
            # Add to cache
            for text, embedding in zip(uncached_texts, uncached_embeddings):
                self.cache[text] = embedding
                
                # If cache is full, remove oldest entries
                if len(self.cache) > self.cache_size:
                    # Get the first key (oldest entry)
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    
        # Retrieve all embeddings from cache
        embeddings = np.array([self.cache[text] for text in texts])
        
        return embeddings


def get_default_embedder() -> BaseEmbedder:
    """
    Get a default embedder based on available dependencies.
    
    Returns:
        An embedder instance
    """
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return CachedEmbedder(SentenceTransformerEmbedder())
    else:
        print("Warning: sentence-transformers not available. Using simple average embedder.")
        return SimpleAverageEmbedder()
