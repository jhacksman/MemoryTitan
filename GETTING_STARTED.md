# Getting Started with MemoryTitan

This guide will help you get started with the MemoryTitan project.

## Prerequisites

- Python 3.9 or higher
- pip or uv for package installation

## Setup

1. Clone or download the repository:
```
git clone <repository-url>
cd MemoryTitan
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. Set up your Venice.ai API key:
   - Copy the `.env_example` file to `.env`
   - Edit the `.env` file and add your API key
   ```
   VENICE_API_KEY=your_api_key_here
   ```

## Running a Simple Example

The easiest way to confirm everything is working is to run the provided example script:

```
python run_example.py
```

This script:
- Creates a simple embedding model (no external dependencies)
- Initializes the TitansManager with the Memory as Context (MAC) architecture
- Adds some test documents to the memory system
- Demonstrates the surprise-based memorization mechanism
- Runs a query to retrieve relevant information

If this script runs successfully, the basic components of MemoryTitan are working correctly.

## Next Steps

### Try the Document Q&A Example

For more advanced examples, check the `examples` directory:

```
python examples/document_qa.py --doc implementation_guide.md
```

This script loads a document file, chunks it, stores it in memory, and allows you to ask questions about its content.

### Experiment with Different Memory Architectures

Try using different memory architectures by changing the `architecture` parameter:

- `mac` (Memory as Context): Concatenates memory with input (default)
- `mag` (Memory as Gate): Uses a gating mechanism to combine memory with processing
- `mal` (Memory as Layer): Processes input sequentially through memory layers
- `mem` (Memory Only): Uses only the long-term memory module

Example:
```
python examples/memory_chat.py --architecture mag
```

### Adjust Memory Parameters

Experiment with different memory sizes and surprise thresholds:

```
python examples/memory_chat.py --short-term-size 20 --long-term-size 50 --surprise-threshold 0.5
```

## Troubleshooting

If you encounter import errors, ensure you are running the scripts from the project root directory and that your virtual environment is activated.

If the venice_api fails with authentication errors, check your `.env` file and ensure your API key is correctly set.

## Learning More

Refer to the `README.md` for more details about the project and its components.

Read the `implementation_guide.md` for in-depth information about the theoretical background and the detailed implementation of the memory systems.