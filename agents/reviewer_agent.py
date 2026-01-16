"""
Reviewer Agent for guardrail and quality checks.
"""
from typing import Dict, List, Optional, Any
import json
from models import GroqClient, OllamaClient, GeminiClient

class ReviewerAgent:
    """Agent responsible for ensuring quality and realism."""
    
    def __init__(self, config: Dict, provider: str = 'groq'):
        self.config = config
        self.provider = provider
        
        if provider == 'groq':
            self.client = GroqClient(config) # Pass config to GroqClient
        elif provider == 'ollama':
            self.client = OllamaClient(config) # Assuming OllamaClient also takes config
        elif provider == 'google':
            self.client = GeminiClient(config) # Assuming GeminiClient also takes config
        else:
            # Fallback or default
            self.client = GroqClient(config) # Fallback should also take config
        self.product_context = config.get('product_context', {})

    def review_review(self, review_text: str, rating: int, persona: Dict) -> Dict[str, Any]:
        """Evaluate the generated review."""
        prompt = self._build_review_prompt(review_text, rating, persona)
        response = self.client.call(prompt, temperature=0.2, max_tokens=500) # Lower temperature for consistency
        
        try:
            # Strip thought chains if present
            if '<think>' in response:
                response = response.split('</think>')[-1].strip()

            # Extract JSON if the model added padding
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            return json.loads(response.strip())
        except Exception as e:
            return {
                "pass": False,
                "overall_score": 0,
                "issues": [f"Reviewer output not JSON: {str(e)}"],
                "raw_response": response
            }

    def _build_review_prompt(self, text: str, rating: int, persona: Dict) -> str:
        aspects = self.product_context.get('aspects', [])
        return f"""
Evaluate this synthetic product review:
Rating: {rating}
Persona: {persona.get('name')}
Review Text: "{text}"

Criteria:
1. Consistency: Does the text match the {rating}-star rating?
2. Persona: Does the tone match the persona traits?
3. Realism: Does it sound like a real user or a standard AI response?
4. Relevance: Does it mention relevant aspects: {aspects}?

Respond ONLY with a JSON object:
{{
  "overall_score": (int 1-10),
  "pass": (bool),
  "issues": [list of strings],
  "sentiment_score": (float -1 to 1)
}}
"""
