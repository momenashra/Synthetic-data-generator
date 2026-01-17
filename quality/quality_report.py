"""
Quality report generation for synthetic reviews.
Combines all quality metrics into a comprehensive report.
"""
from typing import Dict, List, Optional
import json
from datetime import datetime
import numpy as np

from quality.diversity import DiversityAnalyzer
from quality.bias import BiasAnalyzer
from quality.realism import RealismAnalyzer
from models import EmbeddingClient


class QualityReporter:
    """Generates comprehensive quality reports for synthetic reviews."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.diversity_analyzer = DiversityAnalyzer()
        self.bias_analyzer = BiasAnalyzer()
        self.realism_analyzer = RealismAnalyzer(config.get('product_context', {}))
        self.embedding_client = EmbeddingClient()
    
    def generate_report(self, synthetic_reviews: List[Dict], real_reviews: List[str] = None) -> Dict:
        """Generate comprehensive quality report."""
        synthetic_texts = [r['text'] for r in synthetic_reviews]
        
        # Run analyses
        diversity_results = self.diversity_analyzer.analyze(synthetic_texts)
        bias_results = self.bias_analyzer.analyze(
            synthetic_reviews,
            self.config.get('rating_distribution', {})
        )
        realism_results = self.realism_analyzer.analyze(synthetic_texts)
        
        # Calculate scores
        scores = self._calculate_scores(diversity_results, bias_results, realism_results)
        
        # Compare with real (if provided)
        comparison = self._compare_with_real(synthetic_texts, real_reviews) if real_reviews else None
        
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'count': len(synthetic_reviews),
                'provider': synthetic_reviews[0].get('provider', 'unknown') if synthetic_reviews else 'unknown'
            },
            'scores': scores,
            'metrics': {
                'diversity': diversity_results,
                'bias': bias_results,
                'realism': realism_results
            },
            'comparison': comparison,
            'recommendations': self._generate_recommendations(diversity_results, bias_results, realism_results)
        }
    
    def _calculate_scores(self, diversity: Dict, bias: Dict, realism: Dict) -> Dict:
        """Calculate high-level quality scores."""
        # Diversity Score: Avg of TTR and Embedding Diversity
        div_score = (
            diversity['vocabulary']['type_token_ratio'] * 100 + 
            diversity['semantic_similarity']['embedding_diversity'] * 100
        ) / 2
        
        # Bias Score: penalized by failure
        bias_score = 100 if not bias['overall_bias_detected'] else 40
        
        # Realism Score: rewarded by pass
        realism_score = 100 if realism['overall_realistic'] else 40
        
        overall = (div_score * 0.3 + bias_score * 0.35 + realism_score * 0.35)
        
        return {
            'overall': round(overall, 1),
            'diversity': round(div_score, 1),
            'bias': bias_score,
            'realism': realism_score,
            'grade': self._get_grade(overall)
        }
    
    def _compare_with_real(self, synthetic_texts: List[str], real_texts: List[str]) -> Dict:
        """Compare synthetic vs real distribution using shared embeddings."""
        # Semantic Similarity between datasets
        syn_emb = self.embedding_client.encode(synthetic_texts)
        real_emb = self.embedding_client.encode(real_texts)
        
        # Compute max similarity of each synthetic review to ANY real review
        # High max value = potential copying/plagiarism
        # Low max value = novelty
        similarity_matrix = np.dot(syn_emb, real_emb.T) # (N_syn, N_real) assuming normalized
        max_similarities = np.max(similarity_matrix, axis=1)
        
        return {
            'semantic_overlap': round(float(np.mean(max_similarities)), 4),
            'interpretation': "Higher overlap (>0.85) indicates copying; lower (<0.6) indicates novelty."
        }

    def _generate_recommendations(self, diversity: Dict, bias: Dict, realism: Dict) -> List[str]:
        """Simple rule-based recommendations."""
        recs = []
        
        if diversity['semantic_similarity']['avg_similarity'] > 0.75:
            recs.append("⚠️ High semantic repetition. Increase temperature or variety in prompts.")
            
        if bias['overall_bias_detected']:
            recs.append("⚠️ Bias detected (rating skew or sentiment inconsistency). Check checks.")
            
        if not realism['overall_realistic']:
            recs.append("⚠️ Reviews flagged as unrealistic (AI patterns or readability).")
            
        if not recs:
            recs.append("🎉 Quality control passed. Data looks good.")
            
        return recs

    def _get_grade(self, score: float) -> str:
        return 'A' if score >= 90 else 'B' if score >= 80 else 'C' if score >= 70 else 'D' if score >= 60 else 'F'
    
    def save_report(self, report: Dict, path: str):
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
    def print_summary(self, report: Dict):
        s = report['scores']
        print(f"\n Quality Report: {s['overall']}/100 ({s['grade']})")
        print(f"   Diversity: {s['diversity']} | Bias: {s['bias']} | Realism: {s['realism']}")
        print(f"   Recs: {report['recommendations'][0]}")
