# MemoryTitan Examples

This directory contains example scripts that demonstrate how to use the MemoryTitan library.

## Setup

Before running the examples, make sure you have:

1. Created a `.env` file in the project root with your Venice.ai API key
2. Installed the required dependencies with `pip install -r requirements.txt`
3. Run the `test_setup.py` script to verify that MemoryTitan is working correctly

## Available Examples

### Document Question Answering (`document_qa.py`)

This example demonstrates how to use MemoryTitan for document question answering:

```bash
python examples/document_qa.py --doc path/to/your/document.txt --architecture mac
```

Options:
- `--doc`: Path to the document file (required)
- `--architecture`: Memory architecture to use (choices: mac, mag, mal, mem; default: mac)
- `--temperature`: Temperature for the LLM response (default: 0.7)
- `--chunk-size`: Size of document chunks in words (default: 200)
- `--overlap`: Overlap between chunks in words (default: 50)
- `--top-k`: Number of context chunks to retrieve (default: 5)

### Memory-Enhanced Chat (`memory_chat.py`)

This example demonstrates an interactive chat with MemoryTitan's memory systems:

```bash
python examples/memory_chat.py --architecture mac
```

Options:
- `--architecture`: Memory architecture to use (choices: mac, mag, mal, mem; default: mac)
- `--temperature`: Temperature for the LLM response (default: 0.7)
- `--max-tokens`: Maximum tokens to generate (default: 1024)
- `--short-term-size`: Size of short-term memory (default: 50)
- `--long-term-size`: Size of long-term memory (default: 1000)
- `--persistent-size`: Size of persistent memory (default: 10)
- `--top-k`: Number of memory items to retrieve (default: 5)
- `--surprise-threshold`: Threshold for storing items in long-term memory (default: 0.6)
- `--debug`: Enable debug mode to show memory details

### Basic Usage (`basic_usage.py`)

A simple example that demonstrates the core functionality of MemoryTitan:

```bash
python examples/basic_usage.py
```

This example shows how to:
- Initialize the TitansManager with different memory architectures
- Add documents to the memory system
- Query the memory system
- Get memory statistics
- Use the Venice.ai API to generate responses based on retrieved context