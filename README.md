# MemoryTitan

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A vector database implementation inspired by the "Titans: Learning to Memorize at Test Time" paper that manages hierarchical memory systems for handling extremely long contexts.

## Overview

MemoryTitan implements the core concepts from the "Titans: Learning to Memorize at Test Time" paper, focusing on its memory architecture rather than the model training aspects. It provides a practical way to manage large context windows through three types of memory:

1. **Short-term Memory**: For immediate context (attention-based)
2. **Long-term Memory**: A neural memory module that learns to memorize important information at runtime
3. **Persistent Memory**: Fixed, learnable parameters that encode task knowledge

MemoryTitan provides these memory systems in three architectural variants:
- Memory as Context (MAC)
- Memory as Gate (MAG)
- Memory as Layer (MAL)

## Installation

```bash
pip install memory-titan
```

Or install from source:

```bash
git clone https://github.com/yourusername/memory-titan.git
cd memory-titan
pip install -e .
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy
- sentence-transformers
- FAISS or Hnswlib (for vector similarity search)

## Quick Start

```python
from memory_titan import TitansManager
from memory_titan.embedders import SentenceTransformerEmbedder

# Initialize the embedder
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

# Initialize the TitansManager with Memory as Context (MAC) architecture
titans = TitansManager(
    embedder=embedder,
    architecture="mac",  # Options: "mac", "mag", "mal"
    short_term_size=512,  # Size of the short-term memory window
    long_term_size=2048,  # Capacity of the long-term memory
    persistent_size=64,   # Size of the persistent memory
    vector_dim=384,       # Embedding dimension
)

# Add documents to the memory system
titans.add_documents([
    "Transformers are neural network architectures.",
    "The attention mechanism allows models to focus on specific parts of the input.",
    "Long-term memory helps models remember information over extended contexts."
])

# Query the memory system
results = titans.query("How do transformers work?", top_k=2)
for result in results:
    print(f"Content: {result['content']}")
    print(f"Score: {result['score']}")
    print(f"Memory source: {result['source']}")  # 'short_term', 'long_term', or 'persistent'
    print("---")
```

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

### Integration Examples

#### Document QA System

```python
# Load a long document
with open("long_document.txt", "r") as f:
    document = f.read()

# Split into chunks and add to memory
chunks = [document[i:i+1000] for i in range(0, len(document), 1000)]
titans.add_documents(chunks)

# Query the document
answer = titans.query("What are the main findings in the document?", top_k=3)
```

#### Chat System with Long-term Memory

```python
# Initialize chat history
chat_history = []

def chat(user_input):
    # Add user input to history
    chat_history.append(f"User: {user_input}")
    
    # Add to short-term memory
    titans.add_documents([user_input], memory_type="short_term")
    
    # Get response by combining short and long-term memories
    response_components = titans.query(user_input, top_k=5)
    
    # Generate response (in a real system, you'd use an LLM here)
    response = f"Response based on {len(response_components)} memory items"
    
    # Add response to history
    chat_history.append(f"Assistant: {response}")
    
    # Important information moves to long-term memory
    titans.consolidate_memories()
    
    return response
```

## Implementation Details

### Long-term Memory Module

The long-term memory module is implemented following the paper's description:

1. **Surprise Metric**: Measures how unexpected new information is compared to existing memories
2. **Momentum-based Update**: Considers both immediate and recent past surprise
3. **Forgetting Mechanism**: Adaptively erases old memories when they're no longer relevant
4. **Deep Memory**: Uses multiple layers to capture complex relationships in the data

```python
# Internal update mechanism (simplified)
def update_long_term_memory(self, input_embedding, input_text):
    # Compute surprise metric
    surprise = self._compute_surprise(input_embedding)
    
    # Update memory with momentum
    self.momentum = self.momentum_factor * self.momentum + surprise
    
    # Apply forgetting mechanism
    self.memory = (1 - self.forget_rate) * self.memory + self.momentum
    
    # Store the mapping for later retrieval
    self.content_map.append((input_text, input_embedding))
```

## References

The implementation is based on the paper:
- Behrouz, A., Zhong, P., & Mirrokni, V. (2024). Titans: Learning to Memorize at Test Time. arXiv:2501.00663v1.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
