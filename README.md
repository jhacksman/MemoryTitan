# MemoryTitan

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A vector database implementation inspired by the "Titans: Learning to Memorize at Test Time" paper that manages hierarchical memory systems for handling extremely long contexts, integrated with the deepseek-r1-671b model via Venice.ai.

## Overview

MemoryTitan implements the core concepts from the "Titans: Learning to Memorize at Test Time" paper, focusing on its memory architecture for practical applications. It provides a way to manage large context windows through three types of memory:

1. **Short-term Memory**: For immediate context (attention-based)
2. **Long-term Memory**: A neural memory module that learns to memorize important information at runtime
3. **Persistent Memory**: Fixed, learnable parameters that encode task knowledge

MemoryTitan provides these memory systems in three architectural variants:
- Memory as Context (MAC)
- Memory as Gate (MAG)
- Memory as Layer (MAL)

## Installation on macOS with uv

The fastest way to get started with MemoryTitan on macOS is to use [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

### 1. Install uv

If you don't have uv installed already, you can install it using:

```bash
curl -sSf https://astral.sh/uv/install.sh | bash
```

Or using Homebrew:

```bash
brew install uv
```

### 2. Clone the repository

```bash
git clone https://github.com/yourusername/memory-titan.git
cd memory-titan
```

### 3. Create and activate a virtual environment with uv

```bash
uv venv
source .venv/bin/activate
```

### 4. Install MemoryTitan with dependencies

For a basic installation:

```bash
uv pip install -e .
```

For a full installation with all optional dependencies:

```bash
uv pip install -e ".[full]"
```

### 5. Set up your Venice.ai API credentials

Copy the example environment file and add your API key:

```bash
cp .env_example .env
```

Then edit the `.env` file to add your Venice.ai API key:

```
VENICE_API_KEY=your_api_key_here
```

### 6. Run the test script

To verify that your installation is working correctly, run the simple test script:

```bash
python run_example.py
```

This will run a basic memory system example without requiring external APIs.

### 7. Try the CLI tool

MemoryTitan includes a command-line interface for document question answering and memory-enhanced chat. First, test with a local document:

```bash
# Test with the document QA example (requires your API key in the .env file)
python examples/document_qa.py --doc implementation_guide.md --architecture mac

# Test with the memory chat example with debug information
python examples/memory_chat.py --architecture mac --debug

# Try the main CLI tool (comprehensive interface)
python memory_titan_cli.py doc --doc implementation_guide.md --architecture mac

# Or try the CLI chat mode
python memory_titan_cli.py chat --architecture mac

# Test all three memory architectures to compare their behavior
python memory_titan_cli.py doc --doc implementation_guide.md --architecture mac  # Memory as Context
python memory_titan_cli.py doc --doc implementation_guide.md --architecture mag  # Memory as Gate
python memory_titan_cli.py doc --doc implementation_guide.md --architecture mal  # Memory as Layer
```

## Quick Start

### Running the Example

The simplest way to try MemoryTitan is using the provided example script:

```bash
python run_example.py
```

This script demonstrates the core functionality without requiring external dependencies such as sentence-transformers.

### Basic Usage

For more advanced usage, see the examples directory or use the code below:

```python
import os
import sys

# Add the necessary paths (required due to the current structure)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memory"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "embedding"))
sys.path.insert(0, os.path.dirname(__file__))

# Import the necessary modules
from titans_manager import TitansManager
from embedders import SimpleAverageEmbedder
from llm_integration import venice_api  # Requires your API key in .env

# Initialize the embedder
embedder = SimpleAverageEmbedder()

# Get the embedding dimension
embedding_dim = embedder.embed(["test"])[0].shape[0]

# Initialize the TitansManager with Memory as Context (MAC) architecture
titans = TitansManager(
    embedder=embedder,
    architecture="mac",  # Options: "mac", "mag", "mal"
    short_term_size=512,  # Size of the short-term memory window
    long_term_size=2048,  # Capacity of the long-term memory
    persistent_size=64,   # Size of the persistent memory
    vector_dim=embedding_dim,
)

# Add documents to the memory system
titans.add_documents([
    "Transformers are neural network architectures.",
    "The attention mechanism allows models to focus on specific parts of the input.",
    "Long-term memory helps models remember information over extended contexts."
])

# Query the memory system
results = titans.query("How do transformers work?", top_k=3)

# Generate a response using deepseek-r1-671b via Venice.ai (requires API key)
context_chunks = [result['content'] for result in results]
response = venice_api.generate_with_context(
    query="How do transformers work?",
    context_chunks=context_chunks
)

print(response)
```

## Using the Document QA CLI Tool

MemoryTitan includes a command-line tool for document question answering:

```bash
# Make sure your Venice.ai API key is in the .env file first
python document_question_answering.py --doc path/to/your/document.txt --architecture mac
```

Options:
- `--doc`: Path to the document file (required)
- `--architecture`: Memory architecture to use (choices: mac, mag, mal, mem; default: mac)
- `--temperature`: Temperature for the LLM response (default: 0.7)

## Core Components

### Memory Systems

MemoryTitan implements three types of memory as discussed in the Titans paper:

1. **Short-term Memory (`ShortTermMemory`)**: 
   - Attention-based memory for immediate context
   - Fixed context window size
   - Provides precise recall for recent information

2. **Long-term Memory (`LongTermMemory`)**:
   - Neural memory that adaptively stores information based on surprise and importance
   - Uses gradient-based updates with momentum and forgetting mechanisms
   - Allows storage of information beyond the immediate context window

3. **Persistent Memory (`PersistentMemory`)**:
   - Fixed, task-specific knowledge representation
   - Helps ground and guide the memory access process

### Architecture Variants

MemoryTitan offers three ways to combine these memory systems:

1. **Memory as Context (MAC)**:
   - Uses long-term memory and persistent memory as context for processing
   - Concatenates memory representations with input for deeper understanding
   - Best for tasks requiring precise recall with contextual understanding

2. **Memory as Gate (MAG)**:
   - Uses a gating mechanism to combine memory outputs
   - Balances short-term precision with long-term knowledge
   - Efficient for streaming data and real-time applications

3. **Memory as Layer (MAL)**:
   - Processes input sequentially through memory layers
   - Straightforward architecture with good performance
   - Simplest implementation of the three variants

### LLM Integration

MemoryTitan integrates with Venice.ai's API to use the deepseek-r1-671b model:

- Automatically removes the model's `<think></think>` tags
- Provides a clean interface for generating responses based on retrieved context
- Handles API authentication and error handling

## Advanced Usage

### Custom Embedders

You can create custom embedders by implementing the `BaseEmbedder` interface:

```python
from memory_titan.embedders import BaseEmbedder

class MyCustomEmbedder(BaseEmbedder):
    def __init__(self, model_name):
        # Initialize your embedding model
        self.model = load_my_model(model_name)
        
    def embed(self, texts):
        # Implement your embedding logic
        return self.model.encode(texts)
```

### Memory Configuration

Fine-tune how the long-term memory decides what to store:

```python
from memory_titan import TitansManager, LongTermMemoryConfig

# Configure the memory behavior
ltm_config = LongTermMemoryConfig(
    surprise_threshold=0.7,  # Higher values make the memory more selective
    forget_rate=0.05,        # Rate at which old memories fade
    momentum_factor=0.8,     # How much past surprise influences current storage
    memory_depth=2,          # Number of neural network layers in memory (1-4)
)

# Initialize with custom configuration
titans = TitansManager(
    embedder=embedder,
    architecture="mac",
    long_term_config=ltm_config,
    # ... other parameters
)
```

### Custom LLM Settings

Adjust the behavior of the deepseek-r1-671b model:

```python
from llm_integration import venice_api

# Generate with custom settings
response = venice_api.generate_response(
    prompt="Explain quantum computing",
    system_prompt="You are a quantum physics expert explaining complex concepts simply.",
    temperature=0.3,  # Lower for more deterministic responses
    max_tokens=2048,  # Longer responses
    clean_thinking=True  # Remove <think></think> tags
)
```

## References

The implementation is based on the paper:
- Behrouz, A., Zhong, P., & Mirrokni, V. (2024). Titans: Learning to Memorize at Test Time. arXiv:2501.00663v1.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
