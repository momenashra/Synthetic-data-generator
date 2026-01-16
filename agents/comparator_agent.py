"""
Comparator Agent for dataset comparison.
"""
from typing import Dict, List, Any
from quality.quality_report import QualityReporter

class ComparatorAgent:
    """Agent responsible for comparing datasets."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.reporter = QualityReporter(config)
        
    def compare(self, synthetic_reviews: List[Dict], real_reviews: List[str]) -> Dict[str, Any]:
        report = self.reporter.generate_report(synthetic_reviews, real_reviews)
        comparison = report.get('comparison_with_real', {})
        
        sim = comparison.get('similarity_to_real', {}).get('avg_max_similarity', 0)
        verdict = "Realistic" if 0.4 <= sim <= 0.8 else "Potentially Unrealistic"
        comparison['verdict'] = verdict
        
        return comparison
