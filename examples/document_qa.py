#!/usr/bin/env python3
"""
Document question answering with MemoryTitan.

This script demonstrates how to use MemoryTitan for document question answering.
It loads a document, breaks it into chunks, adds them to the memory system,
and allows the user to ask questions about the document content.
"""

import os
import re
import sys
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add the necessary paths (required due to the current structure)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(parent_dir, "core"))
sys.path.insert(0, os.path.join(parent_dir, "memory"))
sys.path.insert(0, os.path.join(parent_dir, "embedding"))
sys.path.insert(0, parent_dir)

# Import directly from modules
from titans_manager import TitansManager
from embedders import get_default_embedder
from llm_integration import venice_api


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
    """Main entry point for the script."""
    # Load environment variables
    load_dotenv()
    
    # Verify API configuration
    if not os.getenv("VENICE_API_KEY"):
        print("Error: VENICE_API_KEY not found in environment variables.")
        print("Please create a .env file with your Venice.ai API key.")
        print("Example:")
        print("VENICE_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Document question answering with MemoryTitan"
    )
    parser.add_argument("--doc", type=str, required=True,
                       help="Path to the document file")
    parser.add_argument("--architecture", type=str, default="mac",
                       choices=["mac", "mag", "mal", "mem"],
                       help="Memory architecture to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for the LLM response (0.0-1.0)")
    parser.add_argument("--chunk-size", type=int, default=200,
                       help="Approximate chunk size in words")
    parser.add_argument("--overlap", type=int, default=50,
                       help="Word overlap between chunks")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of chunks to retrieve for each query")
    
    args = parser.parse_args()
    
    # Check if document file exists
    if not os.path.isfile(args.doc):
        print(f"Error: Document file '{args.doc}' not found.")
        sys.exit(1)
    
    # Load document
    print(f"Loading document from {args.doc}...")
    with open(args.doc, 'r', encoding='utf-8') as f:
        document = f.read()
    
    # Print document stats
    word_count = len(re.findall(r'\S+', document))
    print(f"Document loaded: {len(document)} characters, {word_count} words.")
    
    # Chunk document
    print("Chunking document...")
    chunks = chunk_document(document, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Created {len(chunks)} chunks.")
    
    # Initialize embedder
    print("Initializing embedder...")
    embedder = get_default_embedder()
    
    # Initialize TitansManager
    print(f"Initializing TitansManager with {args.architecture} architecture...")
    vector_dim = embedder.embed(["test"])[0].shape[0]
    titans = TitansManager(
        embedder=embedder,
        architecture=args.architecture,
        vector_dim=vector_dim
    )
    
    # Add document chunks to memory
    print("Adding document chunks to memory...")
    titans.add_documents(chunks)
    
    # Print memory stats
    stats = titans.get_stats()
    print(f"Memory stats:")
    print(f"  Architecture: {stats['architecture']}")
    if 'short_term' in stats:
        print(f"  Short-term memory size: {stats['short_term']['size']}/{stats['short_term']['capacity']}")
    print(f"  Long-term memory size: {stats['long_term']['size']}")
    
    # Set up system prompt for the LLM
    system_prompt = """You are a helpful assistant that answers questions based on the provided document context.
    Stick to the information in the provided context and avoid making up information.
    If the answer cannot be found in the context, say so clearly."""
    
    # Interactive Q&A loop
    print("\nDocument loaded into memory. Ask questions or type 'exit' to quit.")
    while True:
        # Get question from user
        question = input("\nQuestion: ")
        
        if question.lower() in ('exit', 'quit', 'q'):
            break
            
        # Query the memory system
        results = titans.query(question, top_k=args.top_k)
        
        if not results:
            print("No relevant context found in the document.")
            continue
            
        # Extract the content from results
        context_chunks = [result['content'] for result in results]
        
        # Display context for transparency
        print("\nRetrieved relevant document sections:")
        for i, (result, chunk) in enumerate(zip(results, context_chunks), 1):
            print(f"\n{i}. [{result['source']}] (score: {result['score']:.4f})")
            print("-" * 40)
            # Limit display length for readability
            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            print(preview)
            print("-" * 40)
        
        print("\nGenerating answer...")
        
        try:
            # Generate answer using the LLM with the retrieved context
            answer = venice_api.generate_with_context(
                query=question,
                context_chunks=context_chunks,
                system_prompt=system_prompt,
                temperature=args.temperature
            )
            
            print("\nAnswer:")
            print("=" * 80)
            print(answer)
            print("=" * 80)
            
        except Exception as e:
            print(f"\nError generating answer: {str(e)}")


if __name__ == "__main__":
    main()