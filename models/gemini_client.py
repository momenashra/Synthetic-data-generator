import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

class GeminiClient:
    """Client for Google Vertex AI Gemini models."""
    
    def __init__(self, config: dict = None, model_name: str = "gemini-1.5-flash-001"):
        self.config = config or {}
        self.project_id = os.getenv('GOOGLE_PROJECT_ID')
        self.location = os.getenv('GOOGLE_LOCATION', 'us-central1')
        self.model_name = os.getenv('MODEL_NAME', model_name)
        
        if not self.project_id:
            raise ValueError("GOOGLE_PROJECT_ID environment variable is required for GeminiClient")
            
        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(self.model_name)
        
    def call(self, prompt: str, **kwargs) -> str:
        """Generate content using Gemini model."""
        generation_config = {
            "max_output_tokens": kwargs.get('max_tokens', 8192),
            "temperature": kwargs.get('temperature', 1.0),
            "top_p": kwargs.get('top_p', 0.95),
        }
        
        try:
            responses = self.model.generate_content(
                [prompt],
                generation_config=generation_config,
                stream=False,
            )
            return responses.text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise e

    def get_name(self) -> str:
        return f"google/{self.model_name}"
