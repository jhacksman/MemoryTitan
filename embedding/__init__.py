"""
Embedding module for the MemoryTitan package.

This module provides embedders for converting text to vector representations.
"""

from embedders import (
    BaseEmbedder,
    SentenceTransformerEmbedder, 
    SimpleAverageEmbedder,
    CachedEmbedder,
    get_default_embedder
)

__all__ = [
    "BaseEmbedder",
    "SentenceTransformerEmbedder", 
    "SimpleAverageEmbedder",
    "CachedEmbedder",
    "get_default_embedder"
]
