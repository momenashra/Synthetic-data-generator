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



def get_model_slug():
    """Extract a clean model slug including provider and model name."""
    provider = os.getenv('LLM_PROVIDER1', 'google').lower()
    
    if provider == 'groq':
        model_name = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
    elif provider == 'ollama':
        model_name = os.getenv('OLLAMA_MODEL', 'mistral')
    else:
        model_name = os.getenv('MODEL_NAME', 'unknown')

    # Extract base model name (e.g., 'gemini-2.0-flash-001' -> 'gemini')
    model_base = model_name.split('-')[0].split('/')[0].lower()
    # Combine provider and model (e.g., 'google_gemini', 'groq_llama')
    return f"{provider}_{model_base}"


def generate_reviews(
    num_reviews: int,
    config_path: str = 'config/generation_config.yaml',
    real_reviews_path: str = 'data/balanced_reviews_500.csv'
) -> List[Dict]:
    """
    Generate synthetic reviews using the agentic workflow.
    Saves to default path, then creates a copy with model name.
    """
    model_slug = get_model_slug()
    
    # Define model-specific paths
    model_reviews_path = f'data/generated_reviews_{model_slug}.jsonl'
    model_embeddings_path = f'data/review_embeddings_{model_slug}.npy'
    
    # Enable provider-specific logging
    provider = os.getenv('LLM_PROVIDER1', 'google').lower()
    if provider == 'groq':
        active_model = os.getenv('GROQ_MODEL', 'unknown')
    elif provider == 'ollama':
        active_model = os.getenv('OLLAMA_MODEL', 'unknown')
    else:
        active_model = os.getenv('MODEL_NAME', 'unknown')

    # Initialize storage with specific paths BEFORE starting workflow
    from models.review_storage import get_review_storage
    get_review_storage(
        reviews_path=model_reviews_path,
        embeddings_path=model_embeddings_path,
        force_restart=True
    )

    # Load real reviews for the Comparator
    real_reviews = load_real_reviews(real_reviews_path)
    
    # Run the full agentic workflow
    print(f"\n Starting Agentic Review Generation Strategy...")
    print(f"   Provider: {provider}")
    print(f"   Model:    {active_model}")
    print(f"   Storage:  {model_reviews_path}")
    
    result = run_agentic_workflow(
        num_reviews=num_reviews,
        real_reviews=real_reviews
    )
    
    generated_reviews = result.get("all_synthetic_reviews", [])
    model_performance = result.get("model_performance", {})
    
    print(f"\n✓ Workflow Complete. Generated {len(generated_reviews)} reviews.")
    print(f"✓ Saved to: {model_reviews_path}")
    if model_performance:
        print(f"⏱  Total time: {model_performance.get('total_time_seconds', 0):.1f}s")
        print(f"⏱  Avg per review: {model_performance.get('avg_time_per_review', 0):.2f}s")
    
    # Auto-generate quality report
    print(f"\n📊 Generating quality report...")
    from quality.quality_report import QualityReporter
    
    config = load_config()
    reporter = QualityReporter(config)
    report_data = reporter.generate_report(generated_reviews, real_reviews[:50] if real_reviews else None)
    
    # Add model performance to report
    report_data['model_performance'] = model_performance
    
    # Save report with model name only
    report_base = f'data/quality_report_{model_slug}'
    reporter.save_report(report_data, f"{report_base}.json")
    
    print(f"✅ Report saved: {report_base}.json")
    reporter.print_summary(report_data)
    
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