"""
Low-level Ollama API client.
"""
import os
import ollama
from typing import Dict, Any

class OllamaClient:
    """Raw client for interacting with local Ollama API."""
    
    def __init__(self, config: Dict):
        self.model_name = os.getenv('OLLAMA_MODEL', 'llama3')
        self.default_params = config.get('generation_params', {})

    def call(self, prompt: str, **kwargs) -> str:
        """Make a raw local completion call."""
        params = {**self.default_params, **kwargs}
        
        response = ollama.chat(
            model=kwargs.get('model', self.model_name),
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': params.get('temperature', 1),
                'top_p': params.get('top_p', 0.95),
                'num_predict': params.get('max_tokens', 200),
            }
        )
        return response['message']['content'].strip()

    def get_name(self) -> str:
        return f"ollama_{self.model_name}"