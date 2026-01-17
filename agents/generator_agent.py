"""
Generator Agent responsible for converting structured review plans
into natural language text using multiple LLM providers.
"""

from typing import Dict, List, Any
import json
from models import GroqClient, OllamaClient, GeminiClient
import random
import re

BANNED_PHRASES = [
    "great value for money", "highly recommend", "must have", "game changer",
    "solid choice", "budget friendly", "impressed with the quality",
    "overall a great product", "pleasantly surprised", "exceeded my expectations",
    "for the price", "look no further", "can't go wrong"
]

VARIETY_HINTS = [
    "Start your first sentence with a verb or an adverb.",
    "Avoid starting any sentence with 'The' or 'I'.",
    "Use at least one slang term or informal contraction.",
    "Write in a fragmented, 'bullet-point' style text.",
    "Focus on a very specific, tiny detail rather than the whole product."
]

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
        params = self.config.get('generation_params', {})
        prompt = self._build_execution_prompt(plan, real_reviews)
        result = self.client.call(prompt, **params)
        
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
            
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        result = result.replace('**', '').replace('*', '')
        return result

    def _build_execution_prompt(self, plan: Dict, real_reviews: List[str] = None) -> str:
        """Build the prompt from the detailed plan."""
        instruction_variants = [
            "Write a SHORT, IMPACTFUL product review",
            "Craft a brief, raw, and authentic review",
            "Give me a quick, punchy take on this product",
            "Write a short review that sounds like a real social media post"
        ]
        
        prompt = f"INSTRUCTION: {random.choice(instruction_variants)} based on this plan:\n"
        prompt += json.dumps(plan, indent=2)
        
        prompt += "\n\nCRITICAL CONSTRAINTS:"
        prompt += "\n1. LENGTH: Review MUST be between 5 and 50 words. Be concise. NO fluff."
        prompt += "\n2. ASPECT COVERAGE: You MUST explicitly mention the 'focus_aspects' listed in the plan."
        prompt += "\n3. STYLE & CONTEXT: Adopt the 'writing_style' and mention the 'usage_location' naturally."
        prompt += "\n4. SYNTACTIC REQUIREMENT: You MUST follow this rule: " + plan.get('syntactic_requirement', 'None')
        
        anchors = plan.get('semantic_anchors', [])
        if anchors:
            prompt += f"\n5. SEMANTIC ANCHORS: You MUST use at least some of these terms naturally: {', '.join(anchors)}"
        else:
            prompt += "\n5. SEMANTIC ANCHORS: N/A"
        prompt += "\n6. SENTIMENT: Content must align 100% with the star rating."
        prompt += "\n7. NEGATIVE CONSTRAINTS: DO NOT use these clichés: " + ", ".join(BANNED_PHRASES)
        prompt += f"\n8. VARIETY BOOSTER: {random.choice(VARIETY_HINTS)}"
        prompt += "\n9. DIVERSITY: Avoid generic phrases. Be specific to the persona and usage context."
        
        if real_reviews:
            context = "\n".join(real_reviews[:3])
            prompt += f"\n\nContext (Real Review Style):\n{context}\n"
            
        prompt += "\nOutput ONLY the review text. No formatting, no extra words."
        return prompt

    def get_name(self) -> str:
        return self.client.get_name()
