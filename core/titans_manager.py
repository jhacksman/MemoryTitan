"""
TitansManager implementation for managing different memory architectures.

This module provides the main interface for using the Titans memory system,
implementing the three architectural variants described in the paper:
Memory as Context (MAC), Memory as Gate (MAG), and Memory as Layer (MAL).
"""

import numpy as np
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union, Any

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from the memory and embedding modules
from memory.long_term_memory import LongTermMemory, LongTermMemoryConfig
from memory.short_term_memory import ShortTermMemory, ShortTermMemoryConfig
from memory.persistent_memory import PersistentMemory, PersistentMemoryConfig
from embedding.embedders import BaseEmbedder


class ArchitectureType(str, Enum):
    """Enum for the different Titans architecture types."""
    MEMORY_AS_CONTEXT = "mac"
    MEMORY_AS_GATE = "mag"
    MEMORY_AS_LAYER = "mal"
    MEMORY_ONLY = "mem"  # Just the long-term memory module


class TitansManager:
    """
    Main manager class for the Titans memory system.
    
    This class provides a unified interface for working with different
    memory architectures and handles the integration between memory types.
    """
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        architecture: Union[str, ArchitectureType] = ArchitectureType.MEMORY_AS_CONTEXT,
        vector_dim: int = 768,
        short_term_size: int = 512,
        long_term_size: int = 2048,
        persistent_size: int = 64,
        long_term_config: Optional[LongTermMemoryConfig] = None,
        short_term_config: Optional[ShortTermMemoryConfig] = None,
        persistent_config: Optional[PersistentMemoryConfig] = None,
        gate_factor: float = 0.5,  # Used for MAG architecture
    ):
        """
        Initialize the TitansManager.
        
        Args:
            embedder: Text embedder to use
            architecture: Which architectural variant to use
            vector_dim: Dimension of the embedding vectors
            short_term_size: Size of the short-term memory window
            long_term_size: Maximum capacity of the long-term memory
            persistent_size: Number of persistent memory vectors
            long_term_config: Optional custom configuration for long-term memory
            short_term_config: Optional custom configuration for short-term memory
            persistent_config: Optional custom configuration for persistent memory
            gate_factor: Weight factor for gating in MAG architecture
        """
        self.embedder = embedder
        
        # Convert string to enum if needed
        if isinstance(architecture, str):
            self.architecture = ArchitectureType(architecture.lower())
        else:
            self.architecture = architecture
            
        self.vector_dim = vector_dim
        self.gate_factor = gate_factor
        
        # Set up the memory modules
        
        # Long-term memory
        if long_term_config is None:
            long_term_config = LongTermMemoryConfig(
                vector_dim=vector_dim,
                max_capacity=long_term_size
            )
        self.long_term_memory = LongTermMemory(long_term_config)
        
        # Short-term memory (not used in MEMORY_ONLY)
        if self.architecture != ArchitectureType.MEMORY_ONLY:
            if short_term_config is None:
                short_term_config = ShortTermMemoryConfig(
                    window_size=short_term_size,
                    vector_dim=vector_dim
                )
            self.short_term_memory = ShortTermMemory(short_term_config)
        else:
            self.short_term_memory = None
            
        # Persistent memory
        if persistent_config is None:
            persistent_config = PersistentMemoryConfig(
                num_vectors=persistent_size,
                vector_dim=vector_dim
            )
        self.persistent_memory = PersistentMemory(persistent_config)
        
    def add_documents(
        self, 
        documents: List[str], 
        memory_type: str = "all",
        consolidate: bool = True
    ) -> None:
        """
        Add documents to the memory system.
        
        Args:
            documents: List of document texts to add
            memory_type: Which memory to add to ("short_term", "long_term", "all")
            consolidate: Whether to consolidate memories after adding
        """
        if not documents:
            return
            
        # Get embeddings for all documents
        embeddings = self.embedder.embed(documents)
        
        # Add to appropriate memories
        if memory_type in ("short_term", "all") and self.short_term_memory is not None:
            for doc, emb in zip(documents, embeddings):
                self.short_term_memory.add(doc, emb)
                
        if memory_type in ("long_term", "all"):
            for doc, emb in zip(documents, embeddings):
                self.long_term_memory.add(doc, emb)
                
        # Consolidate memories if needed
        if consolidate:
            self.consolidate_memories()
            
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        memory_types: List[str] = None
    ) -> List[Dict]:
        """
        Query the memory system with text.
        
        Args:
            query_text: The query text
            top_k: Number of results to return from each memory type
            memory_types: Which memory types to query (default: depends on architecture)
            
        Returns:
            List of relevant items from memory with their scores
        """
        # Get the query embedding
        query_embedding = self.embedder.embed([query_text])[0]
        
        # Determine which memories to query based on architecture
        if memory_types is None:
            if self.architecture == ArchitectureType.MEMORY_ONLY:
                memory_types = ["long_term"]
            else:
                memory_types = ["short_term", "long_term", "persistent"]
        
        results = []
        
        # Query different memory types and combine results based on architecture
        if self.architecture == ArchitectureType.MEMORY_AS_CONTEXT:
            results = self._query_mac(query_embedding, top_k, memory_types)
        elif self.architecture == ArchitectureType.MEMORY_AS_GATE:
            results = self._query_mag(query_embedding, top_k, memory_types)
        elif self.architecture == ArchitectureType.MEMORY_AS_LAYER:
            results = self._query_mal(query_embedding, top_k, memory_types)
        else:  # MEMORY_ONLY
            if "long_term" in memory_types:
                results.extend(self.long_term_memory.retrieve(query_embedding, top_k))
                
        return results
        
    def _query_mac(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        memory_types: List[str]
    ) -> List[Dict]:
        """
        Query using Memory as Context (MAC) architecture.
        
        In MAC, we first query long-term memory to get context,
        then combine with persistent memory and short-term results.
        """
        results = []
        
        # First, get context from long-term memory
        if "long_term" in memory_types:
            long_term_results = self.long_term_memory.retrieve(query_embedding, top_k)
            results.extend(long_term_results)
            
        # Add persistent memory context
        if "persistent" in memory_types:
            persistent_results = self.persistent_memory.retrieve(query_embedding, top_k)
            results.extend(persistent_results)
            
        # Finally, query short-term memory if available
        if "short_term" in memory_types and self.short_term_memory is not None:
            short_term_results = self.short_term_memory.retrieve(query_embedding, top_k)
            results.extend(short_term_results)
            
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
        
    def _query_mag(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        memory_types: List[str]
    ) -> List[Dict]:
        """
        Query using Memory as Gate (MAG) architecture.
        
        In MAG, we query both memories independently and combine 
        with a gating mechanism based on relevance.
        """
        all_results = []
        short_term_results = []
        long_term_results = []
        
        # Get results from both memories
        if "short_term" in memory_types and self.short_term_memory is not None:
            short_term_results = self.short_term_memory.retrieve(query_embedding, top_k)
            
        if "long_term" in memory_types:
            long_term_results = self.long_term_memory.retrieve(query_embedding, top_k)
            
        if "persistent" in memory_types:
            persistent_results = self.persistent_memory.retrieve(query_embedding, top_k)
            all_results.extend(persistent_results)
        
        # Apply gating to combine short-term and long-term results
        # We'll use the gate_factor to weight between them
        combined_results = {}
        
        # Process short-term results
        for result in short_term_results:
            key = result["content"]
            combined_results[key] = {
                "content": key,
                "score": result["score"] * (1 - self.gate_factor),
                "source": "short_term",
                "sources": ["short_term"]
            }
            
        # Process long-term results
        for result in long_term_results:
            key = result["content"]
            if key in combined_results:
                # If already in combined results, add the gated score
                combined_results[key]["score"] += result["score"] * self.gate_factor
                combined_results[key]["sources"].append("long_term")
            else:
                # Otherwise, add a new entry
                combined_results[key] = {
                    "content": key,
                    "score": result["score"] * self.gate_factor,
                    "source": "long_term",
                    "sources": ["long_term"]
                }
                
        # Convert back to list and add to all_results
        all_results.extend(list(combined_results.values()))
        
        # Sort by combined score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        return all_results[:top_k]
        
    def _query_mal(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        memory_types: List[str]
    ) -> List[Dict]:
        """
        Query using Memory as Layer (MAL) architecture.
        
        In MAL, we process through memory layers sequentially.
        """
        results = []
        
        # First, pass through long-term memory
        if "long_term" in memory_types:
            long_term_results = self.long_term_memory.retrieve(query_embedding, top_k * 2)
            
            # Process long-term results
            for result in long_term_results:
                # Add to results
                results.append(result)
                
        # Then, process through short-term memory if available
        if "short_term" in memory_types and self.short_term_memory is not None:
            # For true layered processing, we'd encode the processed long-term results
            # and then query short-term, but we'll simulate by querying with original
            short_term_results = self.short_term_memory.retrieve(query_embedding, top_k)
            
            # Add short-term results
            results.extend(short_term_results)
            
        # Add persistent memory context
        if "persistent" in memory_types:
            persistent_results = self.persistent_memory.retrieve(query_embedding, top_k // 2)
            results.extend(persistent_results)
            
        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
        
    def consolidate_memories(self) -> None:
        """
        Consolidate memories by moving important information from short-term to long-term.
        
        This implements the surprise-based mechanism from the Titans paper, where
        information that is surprising to the long-term memory is preferentially stored.
        """
        if self.short_term_memory is None:
            return
            
        # Get all content from short-term memory
        short_term_content = self.short_term_memory.get_all_content()
        
        if not short_term_content:
            return
            
        # Get embeddings for all short-term content
        embeddings = self.embedder.embed(short_term_content)
        
        # Evaluate each item for surprise and importance
        # The long_term_memory.add method will automatically use the 
        # surprise threshold to determine what to store
        for content, embedding in zip(short_term_content, embeddings):
            # Pass through long-term memory's add method which will
            # compute surprise and store items that exceed the threshold
            self.long_term_memory.add(content, embedding)
                
    def clear_memories(self, memory_types: List[str] = None) -> None:
        """
        Clear specified memories.
        
        Args:
            memory_types: Which memory types to clear (default: all)
        """
        if memory_types is None:
            memory_types = ["short_term", "long_term", "persistent"]
            
        if "short_term" in memory_types and self.short_term_memory is not None:
            self.short_term_memory.clear()
            
        if "long_term" in memory_types:
            self.long_term_memory.clear()
            
        # We don't typically clear persistent memory as it's meant to be constant,
        # but we'll include the option for completeness
        if "persistent" in memory_types:
            # Re-initialize persistent memory
            self.persistent_memory = PersistentMemory(
                PersistentMemoryConfig(
                    num_vectors=self.persistent_memory.config.num_vectors,
                    vector_dim=self.vector_dim
                )
            )
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "architecture": self.architecture.value,
            "vector_dim": self.vector_dim,
        }
        
        # Add stats from each memory type
        if self.short_term_memory is not None:
            stats["short_term"] = self.short_term_memory.get_stats()
            
        stats["long_term"] = self.long_term_memory.get_stats()
        stats["persistent"] = self.persistent_memory.get_stats()
        
        return stats
