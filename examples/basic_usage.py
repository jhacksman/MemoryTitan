#!/usr/bin/env python3
"""
Basic usage example for MemoryTitan.

This script demonstrates the core functionality of MemoryTitan
with a simple example of adding documents and querying the memory system.
"""

import os
import sys
from dotenv import load_dotenv

# Add the necessary paths (required due to the current structure)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(parent_dir, "core"))
sys.path.insert(0, os.path.join(parent_dir, "memory"))
sys.path.insert(0, os.path.join(parent_dir, "embedding"))
sys.path.insert(0, parent_dir)

# Import directly from modules
from titans_manager import TitansManager
from embedders import SentenceTransformerEmbedder
from llm_integration import venice_api

# Load environment variables for API keys
load_dotenv()

def main():
    """Run the basic usage example."""
    # Check for API key
    if not os.getenv("VENICE_API_KEY"):
        print("Error: VENICE_API_KEY not found in environment.")
        print("Please set this in a .env file or environment variable.")
        return
    
    print("MemoryTitan Basic Usage Example")
    print("===============================")
    
    # Initialize the embedder
    print("\nInitializing embedder...")
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    
    # Initialize the TitansManager with Memory as Context (MAC) architecture
    print("Initializing TitansManager with MAC architecture...")
    titans = TitansManager(
        embedder=embedder,
        architecture="mac",  # Options: "mac", "mag", "mal"
        short_term_size=10,  # Size of the short-term memory window
        long_term_size=100,  # Capacity of the long-term memory
        persistent_size=5,   # Size of the persistent memory
        vector_dim=384,      # Embedding dimension
    )
    
    # Add documents to the memory system
    print("\nAdding documents to memory...")
    documents = [
        "Transformers are neural network architectures that use self-attention mechanisms.",
        "The attention mechanism allows models to focus on specific parts of the input sequence.",
        "Long-term memory helps models remember information over extended contexts.",
        "The Titans architecture uses three memory types: short-term, long-term, and persistent.",
        "Memory as Context (MAC) concatenates memory with input for processing.",
        "Memory as Gate (MAG) uses a gating mechanism to combine memory with processing.",
        "Memory as Layer (MAL) processes input sequentially through memory layers.",
        "The surprise-based storage mechanism determines what to keep in long-term memory.",
        "Forgetting is an important aspect of memory systems to avoid information overload.",
        "The deepseek-r1-671b model contains thinking outputs in <think></think> tags."
    ]
    
    titans.add_documents(documents)
    
    # Print memory stats
    stats = titans.get_stats()
    print(f"\nMemory stats:")
    print(f"  Architecture: {stats['architecture']}")
    print(f"  Short-term memory size: {stats['short_term']['size']}/{stats['short_term']['capacity']}")
    print(f"  Long-term memory size: {stats['long_term']['size']}")
    
    # Run a few example queries
    example_queries = [
        "How do transformers work?",
        "What are the three memory architectures?",
        "What is surprise-based storage?",
        "How does the MAC architecture work?"
    ]
    
    for query in example_queries:
        print(f"\n\nQUERY: {query}")
        print("=" * 40)
        
        # Query the memory system
        results = titans.query(query, top_k=3)
        
        # Display retrieved results
        print("Retrieved context:")
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['source']}] (score: {result['score']:.4f})")
            print(f"   {result['content']}")
        
        # Generate a response using deepseek-r1-671b via Venice.ai
        print("\nGenerating response...")
        context_chunks = [result['content'] for result in results]
        
        try:
            response = venice_api.generate_with_context(
                query=query,
                context_chunks=context_chunks
            )
            
            print("\nRESPONSE:")
            print(response)
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            print("(You need a valid Venice.ai API key to generate responses)")
    
    print("\nExample completed.")


if __name__ == "__main__":
    main()