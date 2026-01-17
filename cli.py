#!/usr/bin/env python3
import click
import os
import sys
import json
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate import generate_reviews, load_config
from models.review_storage import ReviewStorage
from quality.quality_report import QualityReporter

@click.group()
def cli():
    """Synthetic Data Generator CLI"""
    load_dotenv()

@cli.command()
@click.option('--count', '-n', default=10, help='Number of reviews to generate')
def generate(count):
    """Generate synthetic reviews."""
    click.echo(f"🚀 Generating {count} reviews...")
    generate_reviews(num_reviews=count)

@cli.command()
@click.option('--input', '-i', 'input_path', default='data/generated_reviews.jsonl', help='Input reviews file')
@click.option('--output', '-o', default='data/quality_report.json', help='Output report file')
def report(input_path, output):
    """Generate quality report for existing reviews."""
    click.echo("📊 Generating quality report...")
    config = load_config()
    reporter = QualityReporter(config)
    
    reviews = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                reviews.append(json.loads(line))
    except FileNotFoundError:
        click.echo(f"❌ Input file not found: {input_path}")
        return

    report_data = reporter.generate_report(reviews)
    reporter.save_report(report_data, output)
    reporter.print_summary(report_data)
    click.echo(f"Report saved to {output}")

@cli.command()
def migrate():
    """Migrate existing reviews to new embedding format."""
    click.echo("📦 Migrating reviews...")
    storage = ReviewStorage()
    count = storage.migrate_existing_reviews()
    click.echo(f"✅ Migrated {count} reviews.")

if __name__ == '__main__':
    cli()
