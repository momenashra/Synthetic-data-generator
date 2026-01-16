"""
Planner Agent for orchestrating review generation batches.
"""
from typing import Dict, List, Any
import random

class PlannerAgent:
    """Agent responsible for planning the review generation batch."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rating_distribution = config.get('rating_distribution', {})
        self.personas = config.get('personas', [])
        
    def plan_batch(self, num_reviews: int) -> List[Dict[str, Any]]:
        plan = []
        dist_map = {int(k): v for k, v in self.rating_distribution.items()}
        
        all_ratings = random.choices(
            list(dist_map.keys()),
            weights=list(dist_map.values()),
            k=num_reviews
        )
        
        for rating in all_ratings:
            persona = random.choice(self.personas)
            plan.append({
                "rating": rating,
                "persona": persona
            })
        return plan

    def create_detailed_plan(self, rating: int, persona: Dict, feedback: str = None) -> Dict[str, Any]:
        """
        Create a detailed execution plan for a single review.
        """
        # Select specific aspects to focus on to ensure coverage
        aspects = ["Sound Quality", "Bass", "Battery Life", "Comfort/Fit", "Connectivity/Bluetooth", "Microphone Quality", "Build Quality", "Price/Value"]
        selected_aspects = random.sample(aspects, k=random.randint(1, 2))
        
        # Define sentiment guidance based on rating
        if rating <= 2:
            sentiment_guide = "Critical, disappointed, frustrated. Focus on flaws."
        elif rating == 3:
            sentiment_guide = "Mixed, average, neutral. Mention both pros and cons."
        else:
            sentiment_guide = "Positive, satisfied, impressed. Focus on strengths."

        base_plan = {
            "target_rating": rating,
            "persona_name": persona['name'],
            "persona_description": persona['description'],
            "key_traits": persona['traits'],
            "tone": "Natural, imperfect, authentic",
            "focus_aspects": selected_aspects,
            "sentiment_guidance": sentiment_guide,
            "length_guideline": "3-5 sentences",
            "structure": "Opening thought -> specific detail -> closing sentiment"
        }
        
        # Add refinement instructions if feedback exists
        if feedback:
            base_plan["correction_instruction"] = f"PREVIOUS ATTEMPT FAILED. FIX: {feedback}"
            base_plan["strict_constraint"] = "Do NOT repeat the previous mistake."
            
        return base_plan

    def get_summary(self, plan: List[Dict]) -> str:
        from collections import Counter
        ratings = [p['rating'] for p in plan]
        counts = Counter(ratings)
        summary = f"Planned {len(plan)} reviews: "
        summary += ", ".join([f"{r}⭐: {c}" for r, c in sorted(counts.items())])
        return summary
