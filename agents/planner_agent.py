"""
Planner Agent for orchestrating review generation batches.
"""
from typing import Dict, List, Any
import random

class PlannerAgent:
    """
    Planner agent that produces structured, model-agnostic
    generation instructions for synthetic reviews.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.rating_distribution = config.get('rating_distribution', {})
        self.personas = config.get('personas', [])
        self.attributes = config.get('attributes', {})
        self.dist_map = {int(k): v for k, v in self.rating_distribution.items()}

    def plan_batch(self, num_reviews: int) -> List[Dict[str, Any]]:
        plan = []
        
        all_ratings = random.choices(
            list(self.dist_map.keys()),
            weights=list(self.dist_map.values()),
            k=num_reviews
        )
        
        # Improved persona selection to ensure diversity and avoid repetition
        # Strategy: Shuffle personas and cycle through them, with occasional random selection
        # This ensures better distribution while maintaining some randomness
        persona_pool = self.personas.copy()
        random.shuffle(persona_pool)
        persona_index = 0
        last_persona_name = None
        consecutive_count = 0
        
        for i, rating in enumerate(all_ratings):
            # Avoid more than 2 consecutive identical personas
            if consecutive_count >= 2 and last_persona_name:
                # Force a different persona
                available_personas = [p for p in persona_pool if p['name'] != last_persona_name]
                if available_personas:
                    persona = random.choice(available_personas)
                else:
                    # If all personas are the same (shouldn't happen), just pick randomly
                    persona = random.choice(persona_pool)
            else:
                # Use round-robin with occasional random selection for variety
                if random.random() < 0.3:  # 30% chance of random selection
                    persona = random.choice(persona_pool)
                else:
                    # Round-robin through shuffled personas
                    persona = persona_pool[persona_index % len(persona_pool)]
                    persona_index += 1
            
            # Track consecutive personas
            if persona['name'] == last_persona_name:
                consecutive_count += 1
            else:
                consecutive_count = 1
                last_persona_name = persona['name']
            
            plan.append({
                "rating": rating,
                "persona": persona
            })
        return plan

    def create_detailed_plan(self, rating: int, persona: Dict, feedback: str = None) -> Dict[str, Any]:
        """
        Create a detailed execution plan for a single review.
        """
        # Select specific aspects to focus on to ensure coverage (Rotate more broadly)
        all_aspects = self.config.get('product_context', {}).get('aspects', [])
        
        # Weighted selection: Pick some specific ones and some random ones
        # This ensuring we don't just pick the first ones every time
        num_to_sample = min(len(all_aspects), random.randint(1, 2)) if all_aspects else 0
        selected_aspects = random.sample(all_aspects, k=num_to_sample) if num_to_sample > 0 else ["General Features"]
        
        # Select advanced attributes if available
        writing_style = random.choice(self.attributes.get('writing_styles', ["Natural"]))
        emotional_context = random.choice(self.attributes.get('emotional_contexts', ["Standard"]))
        usage_location = random.choice(self.attributes.get('usage_locations', ["General Use"]))
        
        # Syntactic requirements booster
        syntactic_styles = [
            "Use active voice",
            "Use passive voice for at least one sentence",
            "Start with a verb",
            "Start with an adverb",
            "Use a compound sentence (join two ideas with 'and' or 'but')",
            "Include a rhetorical question",
            "Keep it short and fragmented like a text message"
        ]
        chosen_syntax = random.choice(syntactic_styles)
        
        # Get semantic anchors if they exist
        anchors = persona.get('semantic_anchors', [])
        
        # Define sentiment guidance based on rating
        if rating <= 2:
            sentiment_guide = f"Critical, disappointed, frustrated. {emotional_context}. Focus on flaws."
        elif rating == 3:
            sentiment_guide = f"Mixed, average, neutral. {emotional_context}. Mention both pros and cons."
        else:
            sentiment_guide = f"Positive, satisfied, impressed. {emotional_context}. Focus on strengths."

        base_plan = {
            "target_rating": rating,
            "persona_name": persona.get('name', 'General User'),
            "persona_description": persona.get('description', 'A typical customer'),
            "semantic_anchors": anchors,
            "writing_style": writing_style,
            "usage_location": usage_location,
            "syntactic_requirement": chosen_syntax,
            "tone": f"{writing_style}, authentic",
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
