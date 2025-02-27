#!/usr/bin/env python3
"""
MemoryTitan CLI - Command-line interface for MemoryTitan.

This script provides a simple command-line interface for using MemoryTitan
with the Venice.ai API for document question answering and chat applications.
"""

import argparse
import os
import sys
import re
from typing import List, Optional
from dotenv import load_dotenv

# Add the necessary paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memory"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "embedding"))
sys.path.insert(0, os.path.dirname(__file__))

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


def document_qa(args: argparse.Namespace) -> None:
    """
    Run the document QA mode.
    
    Args:
        args: Command-line arguments
    """
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
    chunks = chunk_document(document, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Created {len(chunks)} chunks.")
    
    # Initialize embedder
    print("Initializing embedder...")
    embedder = get_default_embedder()
    
    # Initialize TitansManager
    print(f"Initializing TitansManager with {args.architecture} architecture...")
    titans = TitansManager(
        embedder=embedder,
        architecture=args.architecture,
        short_term_size=args.short_term_size,
        long_term_size=args.long_term_size,
        persistent_size=args.persistent_size,
        vector_dim=embedder.embed(["test"])[0].shape[0]
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
        if not args.quiet:
            print("\nRetrieved relevant document sections:")
            for i, (result, chunk) in enumerate(zip(results, context_chunks), 1):
                print(f"\n{i}. [{result['source']}] (score: {result['score']:.4f})")
                print("-" * 40)
                # Limit display length for readability
                preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                print(preview)
                print("-" * 40)
        
        print("\nGenerating answer using deepseek-r1-671b...")
        
        try:
            # Generate answer using the LLM with the retrieved context
            answer = venice_api.generate_with_context(
                query=question,
                context_chunks=context_chunks,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            print("\nAnswer:")
            print("=" * 80)
            print(answer)
            print("=" * 80)
            
        except Exception as e:
            print(f"\nError generating answer: {str(e)}")


def chat_mode(args: argparse.Namespace) -> None:
    """
    Run the chat mode with memory retention.
    
    Args:
        args: Command-line arguments
    """
    print("Starting MemoryTitan Chat with deepseek-r1-671b...")
    
    # Initialize embedder
    print("Initializing embedder...")
    embedder = get_default_embedder()
    
    # Initialize TitansManager
    print(f"Initializing TitansManager with {args.architecture} architecture...")
    titans = TitansManager(
        embedder=embedder,
        architecture=args.architecture,
        short_term_size=args.short_term_size,
        long_term_size=args.long_term_size,
        persistent_size=args.persistent_size,
        vector_dim=embedder.embed(["test"])[0].shape[0]
    )
    
    # Set up system prompt
    system_prompt = """You are a helpful assistant engaging in a conversation.
    You have access to your conversation history through memory and can refer to previous topics.
    Answer the user's questions thoroughly and accurately."""
    
    # Store chat history
    chat_history = []
    
    print("\nMemory-assisted chat started. Type 'exit' to quit.")
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() in ('exit', 'quit', 'q'):
            break
            
        # Add to chat history
        chat_history.append(f"User: {user_input}")
        
        # Add to memory
        titans.add_documents([user_input])
        
        # Retrieve relevant context from memory
        memory_results = titans.query(user_input, top_k=args.top_k)
        memory_context = [result['content'] for result in memory_results]
        
        try:
            # Create a prompt with memory context
            context_prompt = user_input
            if memory_context and len(memory_context) > 1:  # Don't just use the current input
                context_prompt = f"""Current question: {user_input}
                
Relevant chat history:
"""
                for i, ctx in enumerate(memory_context[1:], 1):  # Skip the first one as it's likely the current input
                    context_prompt += f"\n- {ctx}\n"
                
                context_prompt += "\nPlease answer the current question, taking into account the relevant chat history if needed."
            
            # Generate response
            response_data = venice_api.generate_response(
                prompt=context_prompt,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            # Extract the response
            assistant_response = response_data["choices"][0]["message"]["content"]
            
            # Display response
            print("\nAssistant:", assistant_response)
            
            # Add to chat history and memory
            chat_entry = f"Assistant: {assistant_response}"
            chat_history.append(chat_entry)
            titans.add_documents([chat_entry])
            
            # Consolidate memories periodically
            if len(chat_history) % 10 == 0:
                titans.consolidate_memories()
                
        except Exception as e:
            print(f"\nError generating response: {str(e)}")


def main():
    """Main entry point for the CLI."""
    # Load environment variables
    load_dotenv()
    
    # Verify API configuration
    if not os.getenv("VENICE_API_KEY"):
        print("Error: VENICE_API_KEY not found in environment variables.")
        print("Please create a .env file based on .env_example and add your API key.")
        sys.exit(1)
    
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="MemoryTitan CLI for document QA and chat with memory"
    )
    
    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")
    
    # Document QA mode
    doc_parser = subparsers.add_parser("doc", help="Document QA mode")
    doc_parser.add_argument("--doc", type=str, required=True, help="Path to document file")
    doc_parser.add_argument("--architecture", type=str, default="mac", 
                        choices=["mac", "mag", "mal", "mem"],
                        help="Memory architecture to use")
    doc_parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for the LLM response (0.0-1.0)")
    doc_parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens to generate in the response")
    doc_parser.add_argument("--top-k", type=int, default=5,
                        help="Number of context chunks to retrieve")
    doc_parser.add_argument("--chunk-size", type=int, default=200,
                        help="Approximate chunk size in words")
    doc_parser.add_argument("--overlap", type=int, default=50,
                        help="Word overlap between chunks")
    doc_parser.add_argument("--short-term-size", type=int, default=20,
                        help="Size of short-term memory")
    doc_parser.add_argument("--long-term-size", type=int, default=1000,
                        help="Size of long-term memory")
    doc_parser.add_argument("--persistent-size", type=int, default=10,
                        help="Size of persistent memory")
    doc_parser.add_argument("--quiet", action="store_true",
                        help="Don't display retrieved context")
    
    # Chat mode
    chat_parser = subparsers.add_parser("chat", help="Chat mode with memory")
    chat_parser.add_argument("--architecture", type=str, default="mac", 
                         choices=["mac", "mag", "mal", "mem"],
                         help="Memory architecture to use")
    chat_parser.add_argument("--temperature", type=float, default=0.7,
                         help="Temperature for the LLM response (0.0-1.0)")
    chat_parser.add_argument("--max-tokens", type=int, default=1024,
                         help="Maximum tokens to generate in the response")
    chat_parser.add_argument("--top-k", type=int, default=5,
                         help="Number of memory items to retrieve")
    chat_parser.add_argument("--short-term-size", type=int, default=50,
                         help="Size of short-term memory")
    chat_parser.add_argument("--long-term-size", type=int, default=1000,
                         help="Size of long-term memory")
    chat_parser.add_argument("--persistent-size", type=int, default=10,
                         help="Size of persistent memory")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Route to appropriate mode
    if args.mode == "doc":
        document_qa(args)
    elif args.mode == "chat":
        chat_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
