"""
Low-level Groq API client.
"""
import os
from typing import Dict, List, Optional, Any
from groq import Groq

class GroqClient:
    """Raw client for interacting with Groq API."""
    
    def __init__(self, config: Dict):
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.model_name = os.getenv('GROQ_MODEL', 'llama3-70b-8192')
        self.client = Groq(api_key=self.api_key)
        self.default_params = config.get('generation_params', {})

    def call(self, prompt: str, **kwargs) -> str:
        """Make a raw completion call with retry logic for rate limits."""
        import time
        max_retries = 3
        retry_delay = 5
        
        params = {**self.default_params, **kwargs}
        
        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=kwargs.get('model', self.model_name),
                    temperature=params.get('temperature', 0.1),
                    top_p=params.get('top_p', 0.95),
                    max_tokens=params.get('max_tokens', 200),
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"⚠️  Rate limit reached. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e

    def get_name(self) -> str:
        return f"groq_{self.model_name}"