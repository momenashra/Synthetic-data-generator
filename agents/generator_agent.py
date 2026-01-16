"""
Generator Agent that uses LLM clients to produce review text.
"""
from typing import Dict, List, Any
import json
from models import GroqClient, OllamaClient, GeminiClient

class GeneratorAgent:
    """Agent that generates reviews based on detailed plans."""
    
    def __init__(self, config: Dict, provider: str = 'groq'):
        self.config = config
        self.provider = provider
        self.product_context = config.get('product_context', {})
        
        if provider == 'groq':
            self.client = GroqClient()
        elif provider == 'ollama':
            self.client = OllamaClient()
        elif provider == 'google':
            self.client = GeminiClient()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def execute_plan(self, plan: Dict, real_reviews: List[str] = None) -> str:
        """
        Execute the detailed plan provided by the Planner.
        """
        prompt = self._build_execution_prompt(plan, real_reviews)
        result = self.client.call(prompt)
        
        # Basic cleanup
        import re
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
            
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        result = result.replace('**', '').replace('*', '')
        return result

    def _build_execution_prompt(self, plan: Dict, real_reviews: List[str] = None) -> str:
        """Build the prompt from the detailed plan."""
        prompt = f"INSTRUCTION: Write a SHORT, IMPACTFUL product review based EXACTLY on this plan:\n"
        prompt += json.dumps(plan, indent=2)
        
        prompt += "\n\nCRITICAL CONSTRAINTS:"
        prompt += "\n1. LENGTH: Review MUST be between 5 and 50 words. Be concise. NO fluff."
        prompt += "\n2. ASPECT COVERAGE: You MUST explicitly mention the 'focus_aspects' listed in the plan."
        prompt += "\n3. SENTIMENT: Content must align 100% with the star rating."
        prompt += "\n4. ACURRACY: Do NOT Hallucinate. Stick to the persona traits."
        prompt += "\n5. DIVERSITY: Avoid generic phrases like 'good value'. Be specific."
        
        if real_reviews:
            context = "\n".join(real_reviews[:3])
            prompt += f"\n\nContext (Real Review Style):\n{context}\n"
            
        prompt += "\nOutput ONLY the review text. No formatting, no extra words."
        return prompt

    def get_name(self) -> str:
        return self.client.get_name()
