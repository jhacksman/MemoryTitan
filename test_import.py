#!/usr/bin/env python3
"""
Simple test script to check if imports are working.
"""

import os
import sys

# First, try to import a module directly
print("Attempting to import core/titans_manager.py:")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
    from titans_manager import TitansManager, ArchitectureType
    print("✓ Successfully imported TitansManager")
except Exception as e:
    print(f"✗ Import error: {e}")

# Try with memory modules
print("\nAttempting to import memory modules:")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memory"))
    from long_term_memory import LongTermMemory
    print("✓ Successfully imported LongTermMemory")
except Exception as e:
    print(f"✗ Import error: {e}")

# Try with embedding modules
print("\nAttempting to import embedding modules:")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "embedding"))
    from embedders import SimpleAverageEmbedder
    print("✓ Successfully imported SimpleAverageEmbedder")
except Exception as e:
    print(f"✗ Import error: {e}")

# Try with LLM integration
print("\nAttempting to import llm_integration.py:")
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from llm_integration import venice_api
    print("✓ Successfully imported venice_api")
except Exception as e:
    print(f"✗ Import error: {e}")

# Print pythonpath for debugging
print("\nCurrent sys.path:")
for p in sys.path:
    print(f"- {p}")

# Print the contents of directories
print("\nContents of core directory:")
print(os.listdir("core") if os.path.exists("core") else "Directory not found")

print("\nContents of memory directory:")  
print(os.listdir("memory") if os.path.exists("memory") else "Directory not found")

print("\nContents of embedding directory:")
print(os.listdir("embedding") if os.path.exists("embedding") else "Directory not found")