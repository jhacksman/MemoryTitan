#!/usr/bin/env python3
"""
Test script to verify that the MemoryTitan package is set up correctly.

This script imports the key components of MemoryTitan and runs a simple
test to ensure everything is working properly.
"""

import sys
import os
import numpy as np
from dotenv import load_dotenv

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Try importing components
success = True
try:
    print("Testing imports...")
    # Import using direct paths to the files
    import sys
    import importlib.util
    
    # Load TitansManager module
    spec = importlib.util.spec_from_file_location("titans_manager", "./core/titans_manager.py")
    titans_manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(titans_manager)
    TitansManager = titans_manager.TitansManager
    ArchitectureType = titans_manager.ArchitectureType
    
    # Load memory modules
    spec = importlib.util.spec_from_file_location("long_term_memory", "./memory/long_term_memory.py")
    long_term_memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(long_term_memory)
    LongTermMemory = long_term_memory.LongTermMemory
    LongTermMemoryConfig = long_term_memory.LongTermMemoryConfig
    
    spec = importlib.util.spec_from_file_location("short_term_memory", "./memory/short_term_memory.py")
    short_term_memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(short_term_memory)
    ShortTermMemory = short_term_memory.ShortTermMemory
    
    spec = importlib.util.spec_from_file_location("persistent_memory", "./memory/persistent_memory.py")
    persistent_memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(persistent_memory)
    PersistentMemory = persistent_memory.PersistentMemory
    
    # Load embedders
    spec = importlib.util.spec_from_file_location("embedders", "./embedding/embedders.py")
    embedders = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(embedders)
    BaseEmbedder = embedders.BaseEmbedder
    SimpleAverageEmbedder = embedders.SimpleAverageEmbedder
    get_default_embedder = embedders.get_default_embedder
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    success = False

# Test the embedding functionality
if success:
    try:
        print("\nTesting embedding functionality...")
        # Use the simple embedder to avoid dependencies
        embedder = SimpleAverageEmbedder()
        texts = ["This is a test", "Another test sentence"]
        embeddings = embedder.embed(texts)
        
        if isinstance(embeddings, np.ndarray) and embeddings.shape[0] == 2:
            print(f"✓ Embedding successful: shape {embeddings.shape}")
        else:
            print(f"✗ Embedding output has unexpected shape: {embeddings.shape}")
            success = False
    except Exception as e:
        print(f"✗ Embedding error: {e}")
        success = False

# Test the memory system
if success:
    try:
        print("\nTesting memory components...")
        # Create a simple TitansManager instance
        titans = TitansManager(
            embedder=embedder,
            architecture="mac",
            vector_dim=50,  # Dimension from SimpleAverageEmbedder
            short_term_size=5,
            long_term_size=10
        )
        
        # Add some documents
        documents = ["Document one", "Document two", "Document three"]
        titans.add_documents(documents)
        
        # Query the memory
        results = titans.query("test query", top_k=2)
        
        if isinstance(results, list):
            print(f"✓ Memory query successful: {len(results)} results")
            
            # Test memory stats
            stats = titans.get_stats()
            print(f"✓ Memory stats: {stats['architecture']} architecture")
            print(f"  - Short-term size: {stats['short_term']['size']}/{stats['short_term']['capacity']}")
            print(f"  - Long-term size: {stats['long_term']['size']}")
        else:
            print(f"✗ Memory query returned unexpected type: {type(results)}")
            success = False
    except Exception as e:
        print(f"✗ Memory system error: {e}")
        success = False

# Test the LLM integration (lightly, without making API calls)
load_dotenv()
if success:
    try:
        print("\nTesting LLM integration (no API calls)...")
        # Load llm_integration
        spec = importlib.util.spec_from_file_location("llm_integration", "./llm_integration.py")
        llm_integration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llm_integration)
        venice_api = llm_integration.venice_api
        
        # Just test the tag cleaning functionality
        test_text = "Normal text <think>This is thinking text that should be removed</think> more normal text."
        cleaned = venice_api._clean_response(test_text)
        expected = "Normal text  more normal text."
        
        if cleaned == expected:
            print("✓ LLM tag cleaning works correctly")
        else:
            print(f"✗ LLM tag cleaning output unexpected: '{cleaned}'")
            success = False
    except Exception as e:
        print(f"✗ LLM integration error: {e}")
        success = False

# Print overall status
print("\n" + "=" * 50)
if success:
    print("SUCCESS: All tests passed! MemoryTitan is set up correctly.")
else:
    print("FAILURE: Some tests failed. See the errors above for details.")
    sys.exit(1)

print("\nNext steps:")
print("1. Make sure you have a Venice.ai API key in your .env file")
print("2. Try running the examples in the examples/ directory")
print("3. Read the implementation_guide.md for more details on the project")