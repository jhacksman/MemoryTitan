#!/usr/bin/env python3
"""
Memory-enhanced chat with MemoryTitan.

This script demonstrates how to use MemoryTitan for a persistent chat that
remembers conversation history using the hierarchical memory system.
"""

import os
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
        description="Memory-enhanced chat with MemoryTitan"
    )
    parser.add_argument("--architecture", type=str, default="mac",
                      choices=["mac", "mag", "mal", "mem"],
                      help="Memory architecture to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Temperature for the LLM response (0.0-1.0)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                      help="Maximum tokens to generate in the response")
    parser.add_argument("--short-term-size", type=int, default=50,
                      help="Size of short-term memory")
    parser.add_argument("--long-term-size", type=int, default=1000,
                      help="Size of long-term memory")
    parser.add_argument("--persistent-size", type=int, default=10,
                      help="Size of persistent memory")
    parser.add_argument("--top-k", type=int, default=5,
                      help="Number of memory items to retrieve")
    parser.add_argument("--surprise-threshold", type=float, default=0.6,
                      help="Threshold for storing items in long-term memory (0.0-1.0)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode to show memory details")
    
    args = parser.parse_args()
    
    print("Starting MemoryTitan Chat with deepseek-r1-671b...")
    
    # Initialize embedder
    print("Initializing embedder...")
    embedder = get_default_embedder()
    
    # Create long-term memory config with custom surprise threshold
    from memory_titan.memory.long_term_memory import LongTermMemoryConfig
    ltm_config = LongTermMemoryConfig(
        max_capacity=args.long_term_size,
        surprise_threshold=args.surprise_threshold,
        vector_dim=embedder.embed(["test"])[0].shape[0]
    )
    
    # Initialize TitansManager
    print(f"Initializing TitansManager with {args.architecture} architecture...")
    titans = TitansManager(
        embedder=embedder,
        architecture=args.architecture,
        short_term_size=args.short_term_size,
        long_term_size=args.long_term_size,
        persistent_size=args.persistent_size,
        long_term_config=ltm_config,
        vector_dim=embedder.embed(["test"])[0].shape[0]
    )
    
    # Set up system prompt
    system_prompt = """You are a helpful assistant engaging in a conversation.
    You have access to your conversation history through memory and can refer to previous topics.
    Answer the user's questions thoroughly and accurately.
    If asked about your memory capabilities, explain how you use short-term and long-term memory systems
    to retain important information from the conversation."""
    
    # Store chat history for reference
    chat_history = []
    
    print("\nMemory-assisted chat started. Type 'exit' to quit or 'stats' to see memory statistics.")
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() in ('exit', 'quit', 'q'):
            break
            
        # Handle special commands
        if user_input.lower() == 'stats':
            # Display memory statistics
            stats = titans.get_stats()
            print("\nMemory Statistics:")
            print(f"Architecture: {stats['architecture']}")
            
            if 'short_term' in stats:
                print(f"\nShort-term Memory:")
                print(f"  Size: {stats['short_term']['size']}/{stats['short_term']['capacity']}")
                
            print(f"\nLong-term Memory:")
            print(f"  Size: {stats['long_term']['size']}")
            print(f"  Average Surprise: {stats['long_term'].get('avg_surprise', 0):.4f}")
            print(f"  Maximum Surprise: {stats['long_term'].get('max_surprise', 0):.4f}")
            print(f"  Forget Rate: {stats['long_term'].get('forget_rate', 0):.4f}")
            print(f"  Momentum: {stats['long_term'].get('momentum', 0):.4f}")
            
            print(f"\nPersistent Memory:")
            print(f"  Vectors: {stats['persistent']['num_vectors']}")
            continue
        
        # Add to chat history and memory
        chat_entry = f"User: {user_input}"
        chat_history.append(chat_entry)
        titans.add_documents([chat_entry], memory_type="short_term", consolidate=False)
        
        # Retrieve relevant context from memory
        memory_results = titans.query(user_input, top_k=args.top_k)
        
        # Debug mode: display retrieved memory
        if args.debug:
            print("\nRetrieved from memory:")
            for i, result in enumerate(memory_results, 1):
                print(f"{i}. [{result['source']}] {result['score']:.4f}: {result['content'][:100]}...")
        
        memory_context = [result['content'] for result in memory_results]
        
        try:
            # Create a prompt with memory context
            context_prompt = user_input
            
            if memory_context and len(memory_context) > 1:  # Don't just use the current input
                context_prompt = f"""Current question: {user_input}
                
Relevant chat history:
"""
                # Skip the first one as it's likely the current input
                for i, ctx in enumerate(memory_context[1:], 1):  
                    # Avoid duplicating the current input in the context
                    if ctx != chat_entry:
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
            
            # Periodically consolidate memories to move important info to long-term memory
            if len(chat_history) % 5 == 0:
                print("\n(Consolidating memories...)")
                titans.consolidate_memories()
                
        except Exception as e:
            print(f"\nError generating response: {str(e)}")


if __name__ == "__main__":
    main()