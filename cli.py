#!/usr/bin/env python3
import click
import os
import sys
import json
import glob
from datetime import datetime
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate import generate_reviews
from models.review_storage import ReviewStorage
from quality.quality_report import QualityReporter

@click.group()
def cli():
    """Synthetic Data Generator CLI"""
    load_dotenv()


def _get_model_from_path(path):
    """Extract model name from file path."""
    # e.g., 'generated_reviews_gemini.jsonl' -> 'gemini'
    basename = os.path.basename(path)
    parts = basename.replace('.jsonl', '').replace('.json', '').split('_')
    if len(parts) > 2:
        return parts[-1]
    return 'unknown'


@click.command()
@click.option('--count', '-n', default=10, help='Number of reviews to generate')
def generate(count):
    """Generate synthetic reviews."""
    # Note: Model slug is handled internally by generate_reviews
    click.echo(f"� Generating {count} reviews...")
    generate_reviews(num_reviews=count)


@click.command()
def compare():
    """Compare quality reports across different models."""
    click.echo("📊 Comparing models...")
    
    # Find all quality report JSON files
    report_files = glob.glob('data/quality_report_*.json')
    
    if len(report_files) < 2:
        click.echo("❌ Need at least 2 model reports to compare.")
        click.echo("   Run generation with different models first:")
        click.echo("   MODEL_NAME=gemini-2.0-flash python cli.py generate --count 50")
        click.echo("   MODEL_NAME=llama-3.3-70b python cli.py generate --count 50")
        return
    
    # Load all reports
    reports = {}
    for path in report_files:
        model = _get_model_from_path(path)
        with open(path, 'r') as f:
            reports[model] = json.load(f)
    
    # Create comparison report
    click.echo(f"\n{'='*60}")
    click.echo(f"MODEL COMPARISON REPORT")
    click.echo(f"{'='*60}\n")
    
    # Header
    models = list(reports.keys())
    header = f"{'Metric':<25}" + "".join([f"{m:>15}" for m in models])
    click.echo(header)
    click.echo("-" * len(header))
    
    # Scores
    for metric in ['overall', 'diversity', 'bias', 'realism']:
        row = f"{metric.capitalize():<25}"
        for model in models:
            score = reports[model]['scores'].get(metric, 'N/A')
            row += f"{score:>15}"
        click.echo(row)
    
    # Performance
    click.echo("\n" + "-" * len(header))
    click.echo("Performance:")
    for pmetric in ['total_time_seconds', 'avg_time_per_review', 'reviews_generated']:
        row = f"{pmetric:<25}"
        for model in models:
            perf = reports[model].get('model_performance', {})
            val = perf.get(pmetric, 'N/A')
            if isinstance(val, (int, float)):
                row += f" {val:>15.2f}"
            else:
                row += f" {val:>15}"
        click.echo(row)
    
    click.echo(f"\n{'='*60}")
    
    # Save comparison report
    comparison = {
        'models': models,
        'scores': {m: reports[m]['scores'] for m in models},
        'performance': {m: reports[m].get('model_performance', {}) for m in models}
    }
    
    with open('data/model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    click.echo("✅ Comparison saved to data/model_comparison.json")
    
    # Generate Markdown Comparison
    md_output = []
    md_output.append("# Model Comparison Report\n")
    md_output.append(f"**Generated:** {datetime.now().isoformat()}\n")
    md_output.append(f"**Models Compared:** {', '.join(models)}\n")
    
    # Markdown Table
    md_output.append("\n## Metrics Summary\n")
    header_row = "| Metric | " + " | ".join(models) + " |"
    sep_row = "|---|" + "---|" * len(models)
    md_output.append(header_row)
    md_output.append(sep_row)
    
    for metric in ['overall', 'diversity', 'bias', 'realism']:
        row = f"| **{metric.capitalize()}** |"
        for model in models:
            score = reports[model]['scores'].get(metric, 'N/A')
            row += f" {score} |"
        md_output.append(row)
        
    md_output.append("\n## Performance Metrics\n")
    md_output.append(header_row)
    md_output.append(sep_row)
    
    for pmetric in ['total_time_seconds', 'avg_time_per_review', 'reviews_generated']:
        row = f"| {pmetric} |"
        for model in models:
            perf = reports[model].get('model_performance', {})
            val = perf.get(pmetric, 'N/A')
            if isinstance(val, (int, float)):
                row += f" {val:.2f} |"
            else:
                row += f" {val} |"
        md_output.append(row)
        
    md_path = 'data/model_comparison.md'
    with open(md_path, 'w') as f:
        f.write("\n".join(md_output))
    click.echo(f"✅ Comparison report saved to {md_path}")


@click.command()
@click.option('--file', '-f', required=False, help='Path to reviews file to migrate (optional)')
def migrate(file):
    """Migrate existing reviews to new embedding format."""
    click.echo("📦 Migrating reviews...")
    
    reviews_path = file if file else 'data/generated_reviews.jsonl'
    
    # Auto-deduce embedding path
    dir_name = os.path.dirname(reviews_path)
    base_name = os.path.basename(reviews_path)
    
    if base_name.startswith('generated_reviews_'):
        # e.g. generated_reviews_groq_llama.jsonl -> review_embeddings_groq_llama.npy
        slug = base_name.replace('generated_reviews_', '').replace('.jsonl', '')
        embedding_file = f'review_embeddings_{slug}.npy'
    else:
        # Fallback: reviews.jsonl -> review_embeddings.npy (or simply reviews_embeddings.npy)
        embedding_file = base_name.replace('.jsonl', '_embeddings.npy').replace('.json', '_embeddings.npy')
        
    embeddings_path = os.path.join(dir_name, embedding_file)
    
    click.echo(f"   Input:  {reviews_path}")
    click.echo(f"   Output: {embeddings_path}")
    
    from models.review_storage import get_review_storage
    # Force initialize storage with these specific paths
    storage = get_review_storage(
        reviews_path=reviews_path, 
        embeddings_path=embeddings_path, 
        force_restart=True
    )
    
    count = storage.migrate_existing_reviews()
    click.echo(f"✅ Migrated {count} reviews.")

@click.command()
@click.option('--file', '-f', required=False, help='Path to reviews file')
def report(file):
    """Generate quality report for a specific reviews file."""
    click.echo("📊 Generating quality report...")
    
    reviews_path = file if file else 'data/generated_reviews.jsonl'
    
    if not os.path.exists(reviews_path):
        click.echo(f"❌ File not found: {reviews_path}")
        return

    # Extract slug for report naming
    basename = os.path.basename(reviews_path)
    slug = basename.replace('generated_reviews_', '').replace('.jsonl', '').replace('.json', '')
    if slug == basename: 
         slug = 'custom'
    
    report_path = f"data/quality_report_{slug}.json"
    
    # Load reviews
    reviews = []
    try:
        with open(reviews_path, 'r') as f:
            for line in f:
                if line.strip():
                    reviews.append(json.loads(line))
    except Exception as e:
        click.echo(f"❌ Error reading reviews: {e}")
        return
                
    if not reviews:
        click.echo("❌ No reviews found in file.")
        return

    click.echo(f"   Input: {reviews_path}")
    click.echo(f"   Count: {len(reviews)}")
    
    # Initialize reporter 
    config = {
        'product_context': {}, 
        'rating_distribution': {}, 
    }
    
    try:
        reporter = QualityReporter(config)
        report_data = reporter.generate_report(reviews)
        
        # Save
        reporter.save_report(report_data, report_path)
        click.echo(f"✅ Report saved to {report_path}")
        
        # Print summary
        reporter.print_summary(report_data)
    except Exception as e:
        click.echo(f"❌ Error generating report: {e}")

# Add commands to group
cli.add_command(generate)
cli.add_command(compare)
cli.add_command(migrate)
cli.add_command(report)

if __name__ == '__main__':
    cli()

