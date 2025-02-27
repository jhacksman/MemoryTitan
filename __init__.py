"""
MemoryTitan: A vector database implementation inspired by the Titans paper.

This library implements the hierarchical memory system described in 
"Titans: Learning to Memorize at Test Time", with a focus on practical
vector database functionality for managing large context windows.
"""

from .core.titans_manager import TitansManager, ArchitectureType
from .memory.long_term_memory import LongTermMemory, LongTermMemoryConfig
from .memory.short_term_memory import ShortTermMemory, ShortTermMemoryConfig
from .memory.persistent_memory import PersistentMemory, PersistentMemoryConfig
from .embedding.embedders import (
    BaseEmbedder, 
    SentenceTransformerEmbedder,
    SimpleAverageEmbedder,
    CachedEmbedder,
    get_default_embedder
)

__version__ = "0.1.0"

__all__ = [
    "TitansManager",
    "ArchitectureType",
    "LongTermMemory",
    "LongTermMemoryConfig",
    "ShortTermMemory",
    "ShortTermMemoryConfig",
    "PersistentMemory",
    "PersistentMemoryConfig",
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "SimpleAverageEmbedder",
    "CachedEmbedder",
    "get_default_embedder",
]
