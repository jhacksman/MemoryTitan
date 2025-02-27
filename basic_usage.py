"""
Basic usage example for the MemoryTitan library.

This example demonstrates how to use the MemoryTitan library to create
a simple memory system and interact with it.
"""

from memory_titan import TitansManager
from memory_titan.embedding.embedders import SentenceTransformerEmbedder, SimpleAverageEmbedder
from memory_titan.memory.long_term_memory import LongTermMemoryConfig

def main():
    """
    Run a basic example of the MemoryTitan library.
    """
    print("MemoryTitan Basic Example")
    print("-" * 50)
    
    # Choose an embedder based on what's available
    try:
        print("Initializing SentenceTransformer embedder...")
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    except ImportError:
        print("SentenceTransformer not available, using SimpleAverageEmbedder.")
        embedder = SimpleAverageEmbedder()
        
    # Initialize a TitansManager with Memory as Context (MAC) architecture
    print("Initializing TitansManager with MAC architecture...")
    titans = TitansManager(
        embedder=embedder,
        architecture="mac",  # Options: "mac", "mag", "mal", "mem"
        short_term_size=10,  # Small size for the example
        long_term_size=50,
        persistent_size=5,
        vector_dim=embedder.embed(["test"])[0].shape[0],  # Match embedder dimension
        long_term_config=LongTermMemoryConfig(
            surprise_threshold=0.6,
            forget_rate=0.05
        )
    )
    
    # Add documents to the memory system
    print("\nAdding documents to memory...")
    titans.add_documents([
        "Transformers are neural network architectures that use self-attention.",
        "The attention mechanism allows models to focus on specific parts of the input.",
        "Large language models like GPT and BERT are based on transformer architectures.",
        "Long-term memory helps models remember information over extended contexts.",
        "Short-term memory provides precise recall for recent information.",
        "Persistent memory stores task-specific knowledge.",
        "Memory as Context (MAC) uses memory as context for processing inputs.",
        "Memory as Gate (MAG) uses a gating mechanism to combine memory outputs.",
        "Memory as Layer (MAL) processes inputs sequentially through memory layers."
    ])
    
    # Get statistics about the memory system
    print("\nMemory system statistics:")
    stats = titans.get_stats()
    print(f"  Architecture: {stats['architecture']}")
    print(f"  Vector dimension: {stats['vector_dim']}")
    print(f"  Short-term memory size: {stats['short_term']['size']}/{stats['short_term']['capacity']}")
    print(f"  Long-term memory size: {stats['long_term']['size']}")
    print(f"  Persistent memory size: {stats['persistent']['size']}")
    
    # Example queries
    print("\nQuerying the memory system...\n")
    
    queries = [
        "How do transformers work?",
        "What is the purpose of attention?",
        "What are the different memory architectures?",
        "What is the role of long-term memory?"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        results = titans.query(query, top_k=3)
        
        print("Results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['source']}] {result['content']} (score: {result['score']:.4f})")
        print()
    
    # Add more documents and test consolidation
    print("Adding more documents and consolidating memories...")
    titans.add_documents([
        "The Titans architecture combines different types of memory for improved performance.",
        "Neural memory can learn to store information adaptively based on surprise."
    ])
    
    # Consolidate memories
    titans.consolidate_memories()
    
    # Query again
    print("\nQuerying after consolidation...\n")
    query = "What is special about the Titans architecture?"
    print(f"Query: {query}")
    results = titans.query(query, top_k=3)
    
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result['source']}] {result['content']} (score: {result['score']:.4f})")
    
    # Test different architectures
    print("\nTesting different architectures...\n")
    
    architectures = ["mac", "mag", "mal"]
    query = "What types of memory are there?"
    
    for arch in architectures:
        # Create a new TitansManager with the specified architecture
        titans_arch = TitansManager(
            embedder=embedder,
            architecture=arch,
            short_term_size=10,
            long_term_size=50,
            persistent_size=5,
            vector_dim=embedder.embed(["test"])[0].shape[0]
        )
        
        # Add the same documents
        titans_arch.add_documents([
            "Short-term memory provides precise recall for recent information.",
            "Long-term memory helps models remember information over extended contexts.",
            "Persistent memory stores task-specific knowledge."
        ])
        
        # Query
        print(f"Architecture: {arch}")
        print(f"Query: {query}")
        results = titans_arch.query(query, top_k=3)
        
        print("Results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['source']}] {result['content']} (score: {result['score']:.4f})")
        print()
        
    print("Example complete!")


if __name__ == "__main__":
    main()
