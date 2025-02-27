"""
Integration with Venice.ai API using the deepseek-r1-671b model.

This module provides functionality to call the Venice.ai API with an OpenAI-compatible
interface, specifically configured for the deepseek-r1-671b model. It also handles
removing the <think></think> tags that are part of the model's response format.
"""

import os
import re
import json
import requests
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VeniceAPI:
    """
    Client for Venice.ai's OpenAI-compatible API using the deepseek-r1-671b model.
    """
    
    def __init__(self):
        """Initialize the Venice API client."""
        self.api_key = os.getenv("VENICE_API_KEY")
        self.api_base = os.getenv("VENICE_API_BASE_URL", "https://api.venice.ai/api/v1")
        self.model = os.getenv("VENICE_MODEL", "deepseek-r1-671b")
        
        if not self.api_key:
            raise ValueError(
                "Venice API key not found. Please set VENICE_API_KEY in your .env file."
            )
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _clean_response(self, text: str) -> str:
        """
        Remove <think></think> tags and their content from the response.
        
        Args:
            text: The raw response text from the model
            
        Returns:
            Cleaned response text without the thinking part
        """
        # Remove <think>...</think> blocks
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Clean up any extra whitespace resulting from removals
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def generate_response(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        clean_thinking: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response from the deepseek-r1-671b model.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            clean_thinking: Whether to remove <think></think> tags from the response
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Construct the messages in OpenAI-compatible format
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Make the API request
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        # Check for errors
        if response.status_code != 200:
            error_msg = f"API request failed with status {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += f": {error_data['error']['message']}"
            except:
                error_msg += f": {response.text}"
            
            raise Exception(error_msg)
        
        # Parse the response
        result = response.json()
        
        # Extract the assistant's message
        assistant_message = result["choices"][0]["message"]["content"]
        
        # Clean the response if requested
        if clean_thinking:
            assistant_message = self._clean_response(assistant_message)
            # Update the result object
            result["choices"][0]["message"]["content"] = assistant_message
            
        return result
    
    def generate_with_context(
        self,
        query: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
        max_chunks: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate a response using relevant context chunks.
        
        Args:
            query: The user query
            context_chunks: List of relevant context chunks
            system_prompt: Optional system prompt
            max_chunks: Maximum number of context chunks to include
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Limit the number of context chunks to avoid exceeding token limits
        context_chunks = context_chunks[:max_chunks]
        
        # Build a prompt that includes the context
        prompt = f"Question: {query}\n\nRelevant context:\n"
        
        for i, chunk in enumerate(context_chunks, 1):
            prompt += f"\n--- Context Chunk {i} ---\n{chunk}\n"
            
        prompt += "\nPlease answer the question based on the provided context."
        
        # Generate the response
        response = self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt or "You are a helpful assistant that answers questions based on the provided context.",
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response["choices"][0]["message"]["content"]


# Singleton instance for easy import
venice_api = VeniceAPI()
