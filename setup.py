#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="memory-titan",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "requests>=2.28.0",
        "python-dotenv>=0.20.0",
        "sentence-transformers>=2.2.0"
    ],
    author="MemoryTitan Team",
    author_email="info@example.com",
    description="A vector database implementation inspired by the Titans paper",
    keywords="memory, llm, vector, database, context",
    python_requires=">=3.9"
)