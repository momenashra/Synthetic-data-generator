"""
Main generation script for synthetic reviews.
"""
import os
import json
import yaml
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
from build_graph import run_agentic_workflow






def load_config(config_path: str = 'config/generation_config.yaml') -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_real_reviews(file_path: str = 'data/balanced_reviews_500.csv') -> List[str]:
    """Load real reviews from CSV or JSONL file."""
    reviews = []
    if not os.path.exists(file_path):
        # Try fallback to JSONL if default CSV not found
        if file_path == 'data/balanced_reviews_500.csv' and os.path.exists('data/real_reviews.jsonl'):
            file_path = 'data/real_reviews.jsonl'
        else:
            print(f"Warning: {file_path} not found. Running without real review context.")
            return []

    if file_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(file_path)
        # Assuming 'ReviewBody' is the column name based on analysis
        if 'ReviewBody' in df.columns:
            reviews = df['ReviewBody'].astype(str).tolist()
        elif 'text' in df.columns:
            reviews = df['text'].astype(str).tolist()
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    review_obj = json.loads(line)
                    review_text = review_obj.get('text') or review_obj.get('review_text') or review_obj.get('reviewText', '')
                    if review_text:
                        reviews.append(review_text)
                except:
                    continue
    return reviews



def generate_reviews(
    num_reviews: int,
    config_path: str = 'config/generation_config.yaml',
    real_reviews_path: str = 'data/balanced_reviews_500.csv',
    output_path: str = 'data/generated_reviews.jsonl'
) -> List[Dict]:
    """
    Generate synthetic reviews using the agentic workflow.
    """
    # Load real reviews for the Comparator
    real_reviews = load_real_reviews(real_reviews_path)
    
    # Run the full agentic workflow
    print(f"\n🚀 Starting Agentic Review Generation Strategy...")
    result = run_agentic_workflow(
        num_reviews=num_reviews,
        real_reviews=real_reviews
    )
    
    generated_reviews = result.get("all_synthetic_reviews", [])
    
    # Save results
    save_reviews(generated_reviews, output_path)
    
    print(f"\n✓ Workflow Complete. Generated {len(generated_reviews)} reviews.")
    print(f"✓ Saved to: {output_path}")
    
    return generated_reviews


def save_reviews(reviews: List[Dict], output_path: str):
    """Save reviews to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for review in reviews:
            f.write(json.dumps(review, ensure_ascii=False) + '\n')


def load_generated_reviews(input_path: str = 'data/generated_reviews.jsonl') -> List[Dict]:
    """Load generated reviews from JSONL file."""
    reviews = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            reviews.append(json.loads(line))
    return reviews


if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    
    # Load config to get num_reviews
    config = load_config()
    num_reviews = config.get('num_reviews', 10)
    
    # Generate reviews
    generate_reviews(
        num_reviews=num_reviews
    )