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
def migrate():
    """Migrate existing reviews to new embedding format."""
    click.echo("📦 Migrating reviews...")
    storage = ReviewStorage()
    count = storage.migrate_existing_reviews()
    click.echo(f"✅ Migrated {count} reviews.")

# Add commands to group
cli.add_command(generate)
cli.add_command(compare)
cli.add_command(migrate)

if __name__ == '__main__':
    cli()

