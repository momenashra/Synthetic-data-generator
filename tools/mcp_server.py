import os
import sys

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import asyncio
import argparse
import yaml
import logging
from mcp.server.fastmcp import FastMCP, Context
from tools.evaluator_tools import evaluate_review_batch, compare_to_real_data, save_quality_report

mcp = FastMCP("evaluator")

def load_config(config_path: str = 'config/generation_config.yaml') -> dict:
    from generate import load_config as load
    return load(os.path.join(project_root, config_path))

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

@mcp.tool()
def evaluate_reviews(reviews: list) -> str:
    """Evaluate a batch of synthetic reviews."""
    config = load_config()
    report = evaluate_review_batch(reviews, config)
    return str(report)

@mcp.tool()
def compare_dataset(synthetic_reviews: list, real_reviews: list) -> str:
    """Compare synthetic reviews to real reviews."""
    config = load_config()
    comparison = compare_to_real_data(synthetic_reviews, real_reviews, config)
    return str(comparison)

@mcp.tool()
def save_report(report_data: dict, output_path: str = 'data/quality_report.json') -> str:
    """Save the quality report to disk."""
    return save_quality_report(report_data, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--transport", type=str, choices=["stdio", "sse"], default="stdio", help="Transport mode")
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")
