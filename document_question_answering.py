"""
Document question answering example using MemoryTitan.

This example demonstrates how to use MemoryTitan to implement a simple
document question answering system that can handle long documents by
managing context windows effectively.
"""

import argparse
import os
import re
from typing import List, Dict, Tuple

from memory_titan import TitansManager
from memory_titan.embedding.embedders import get_default_embedder


def chunk_document(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Split a document into overlapping chunks.
    
    Args:
        text: Document text to split
        chunk_size: Approximate size of each chunk in words
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of document chunks
    """
    # Split text into words
    words = re.findall(r'\S+', text)
    
    chunks = []
    start = 0
    
    while start < len(words):
        # Calculate end position
        end = min(start + chunk_size, len(words))
        
        # Create chunk
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if end < len(words) else end
        
    return chunks


def main():
    """Run the document QA example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MemoryTitan Document QA Example')
    parser.add_argument('--doc', type=str, required=True, help='Path to document file')
    parser.add_argument('--architecture', type=str, default='mac', 
                        choices=['mac', 'mag', 'mal', 'mem'],
                        help='Memory architecture to use')
    args = parser.parse_args()
    
    # Check if document file exists
    if not os.path.isfile(args.doc):
        print(f"Error: Document file '{args.doc}' not found.")
        return
    
    # Load document
    print(f"Loading document from {args.doc}...")
    with open(args.doc, 'r', encoding='utf-8') as f:
        document = f.read()
    
    # Print document stats
    word_count = len(re.findall(r'\S+', document))
    print(f"Document loaded: {len(document)} characters, {word_count} words.")
    
    # Chunk document
    print("Chunking document...")
    chunks = chunk_document(document)
    print(f"Created {len(chunks)} chunks.")
    
    # Initialize embedder
    print("Initializing embedder...")
    embedder = get_default_embedder()
    
    # Initialize TitansManager
    print(f"Initializing TitansManager with {args.architecture} architecture...")
    titans = TitansManager(
        embedder=embedder,
        architecture=args.architecture,
        short_term_size=20,  # Store recent chunks
        long_term_size=1000,  # Store important information from the document
        persistent_size=10,   # Task-specific knowledge
        vector_dim=embedder.embed(["test"])[0].shape[0]
    )
    
    # Add document chunks to memory
    print("Adding document chunks to memory...")
    titans.add_documents(chunks)
    
    # Print memory stats
    stats = titans.get_stats()
    print(f"Memory stats:")
    print(f"  Architecture: {stats['architecture']}")
    print(f"  Short-term memory size: {stats['short_term']['size']}/{stats['short_term']['capacity']}")
    print(f"  Long-term memory size: {stats['long_term']['size']}")
    
    # Interactive Q&A loop
    print("\nDocument loaded into memory. Ask questions or type 'exit' to quit.")
    while True:
        # Get question from user
        question = input("\nQuestion: ")
        
        if question.lower() in ('exit', 'quit', 'q'):
            break
            
        # Query the memory system
        results = titans.query(question, top_k=5)
        
        # Display results
        print("\nRelevant document sections:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result['source']}] (score: {result['score']:.4f})")
            print("-" * 80)
            print(result['content'])
            print("-" * 80)
        
        # In a real system, you would use an LLM to generate an answer based on these sections
        print("\nTo generate an actual answer, you would typically pass these sections to an LLM.")
        print("In a production system, you could use OpenAI, Claude, or another LLM API here.")


if __name__ == "__main__":
    main()
