#!/usr/bin/env python3
"""
Simple script to run MemoryTitan with a direct import approach.

This script uses direct path manipulations to import the necessary modules
and run a simple example of MemoryTitan.
"""

import os
import sys
import numpy as np
from dotenv import load_dotenv

# Add paths to the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memory"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "embedding"))
sys.path.insert(0, os.path.dirname(__file__))

# Import the necessary modules
from titans_manager import TitansManager, ArchitectureType
from embedders import SimpleAverageEmbedder

def main():
    """Run a simple example of MemoryTitan."""
    print("MemoryTitan Simple Example")
    print("=========================\n")
    
    # Create a simple embedder (no external dependencies)
    print("Creating embedder...")
    embedder = SimpleAverageEmbedder()
    
    # Create a TitansManager instance
    print("Creating TitansManager...")
    
    # Get the embedding dimension from the embedder
    test_embedding = embedder.embed(["test"])[0]
    embedding_dim = test_embedding.shape[0]
    print(f"Embedding dimension: {embedding_dim}")
    
    titans = TitansManager(
        embedder=embedder,
        architecture="mac",
        vector_dim=embedding_dim,  # Use the detected dimension
        short_term_size=10,
        long_term_size=20
    )
    
    # Create memory config with high surprise threshold for demonstration
    from long_term_memory import LongTermMemoryConfig
    ltm_config = LongTermMemoryConfig(
        vector_dim=embedding_dim,
        max_capacity=20,
        surprise_threshold=0.0001  # Very low threshold -> everything should get stored
    )
    
    # Recreate the titans manager with the new config
    titans = TitansManager(
        embedder=embedder,
        architecture="mac",
        vector_dim=embedding_dim,
        short_term_size=10,
        long_term_size=20,
        long_term_config=ltm_config
    )
    
    # Add some documents
    print("\nAdding documents to memory...")
    documents = [
        "Transformers use self-attention mechanisms",
        "The attention mechanism allows models to focus on specific parts",
        "Long-term memory helps models remember over extended contexts",
        "The Titans architecture uses three memory types",
        "Memory as Context (MAC) concatenates memory with input"
    ]
    
    titans.add_documents(documents)
    
    # Get memory stats
    stats = titans.get_stats()
    print("\nMemory stats after initial documents:")
    print(f"  Architecture: {stats['architecture']}")
    print(f"  Short-term memory size: {stats['short_term']['size']}/{stats['short_term']['capacity']}")
    print(f"  Long-term memory size: {stats['long_term']['size']}")
    print(f"  Average surprise: {stats['long_term'].get('avg_surprise', 0):.4f}")
    
    # Now add a very unique document that should exceed the surprise threshold
    print("\nAdding a unique document that should be surprising...")
    unique_doc = "Quantum entanglement enables superdense coding and quantum teleportation protocols"
    titans.add_documents([unique_doc])
    
    # Manually run consolidation
    print("\nConsolidating memories...")
    titans.consolidate_memories()
    
    # Check stats again
    stats = titans.get_stats()
    print("\nMemory stats after unique document:")
    print(f"  Architecture: {stats['architecture']}")
    print(f"  Short-term memory size: {stats['short_term']['size']}/{stats['short_term']['capacity']}")
    print(f"  Long-term memory size: {stats['long_term']['size']}")
    print(f"  Average surprise: {stats['long_term'].get('avg_surprise', 0):.4f}")
    
    # Run a query
    query = "How do transformers work?"
    print(f"\nRunning query: '{query}'")
    
    results = titans.query(query, top_k=3)
    
    print("\nQuery results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['source']}] Score: {result['score']:.4f}")
        print(f"   {result['content']}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()