"""
MCP Evaluation Tools for review analysis.
"""
from typing import List, Dict, Any
import json
import os
from quality.quality_report import QualityReporter

def evaluate_review_batch(reviews: List[Dict], config: Dict) -> Dict[str, Any]:
    """
    Tool to perform a full quality analysis on a batch of reviews.
    
    Args:
        reviews: List of review dictionaries
        config: System configuration
    """
    reporter = QualityReporter(config)
    report = reporter.generate_report(reviews)
    return report

def compare_to_real_data(synthetic_reviews: List[Dict], real_reviews: List[str], config: Dict) -> Dict[str, Any]:
    """
    Tool to compare synthetic reviews against real baseline data.
    """
    reporter = QualityReporter(config)
    report = reporter.generate_report(synthetic_reviews, real_reviews)
    return report.get('comparison_with_real', {})

def save_quality_report(report: Dict, path: str = 'data/quality_report.json'):
    """
    Tool to persist a quality report to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return f"Report saved to {path}"
