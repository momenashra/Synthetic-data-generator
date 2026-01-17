"""
Markdown report generator for quality metrics.
Converts JSON quality report to human-readable markdown.
"""
from typing import Dict, List, Optional
from datetime import datetime
import os


def generate_markdown_report(report: Dict, output_path: str = 'data/quality_report.md') -> str:
    """Generate a markdown quality report from JSON data."""
    
    md = []
    
    # Header
    md.append("# Synthetic Review Quality Report\n")
    md.append(f"**Generated:** {report['metadata']['generated_at']}\n")
    md.append(f"**Reviews Analyzed:** {report['metadata']['count']}\n")
    md.append(f"**Model:** {report['metadata'].get('provider', 'unknown')}\n")
    
    # Overall Score
    scores = report['scores']
    md.append("\n## Overall Quality\n")
    md.append(f"| Metric | Score | Grade |")
    md.append(f"|--------|-------|-------|")
    md.append(f"| **Overall** | {scores['overall']}/100 | **{scores['grade']}** |")
    md.append(f"| Diversity | {scores['diversity']}/100 | |")
    md.append(f"| Bias | {scores['bias']}/100 | |")
    md.append(f"| Realism | {scores['realism']}/100 | |")
    md.append("")
    
    # Diversity Metrics
    md.append("\n## Diversity Metrics\n")
    div = report['metrics']['diversity']
    
    md.append("### Vocabulary")
    vocab = div.get('vocabulary', {})
    md.append(f"- **Type-Token Ratio:** {vocab.get('type_token_ratio', 'N/A')}")
    md.append(f"- **Unique Tokens:** {vocab.get('unique_tokens', 'N/A')}")
    md.append(f"- **Total Tokens:** {vocab.get('total_tokens', 'N/A')}")
    md.append("")
    
    md.append("### Semantic Similarity")
    sem = div.get('semantic_similarity', {})
    md.append(f"- **Avg Similarity:** {sem.get('avg_similarity', 'N/A')}")
    md.append(f"- **Embedding Diversity:** {sem.get('embedding_diversity', 'N/A')}")
    md.append("")
    
    md.append("### Lexical Diversity")
    lex = div.get('lexical_diversity', {})
    md.append(f"- **Distinct-1:** {lex.get('distinct_1', 'N/A')}")
    md.append(f"- **Distinct-2:** {lex.get('distinct_2', 'N/A')}")
    md.append(f"- **Distinct-3:** {lex.get('distinct_3', 'N/A')}")
    md.append("")
    
    # Bias Metrics
    md.append("\n## Bias Analysis\n")
    bias = report['metrics']['bias']
    md.append(f"**Overall Bias Detected:** {'⚠️ Yes' if bias.get('overall_bias_detected') else '✅ No'}")
    md.append("")
    
    md.append("### Rating Distribution")
    rating = bias.get('rating_distribution', {})
    actual = rating.get('actual_distribution', {})
    if actual:
        md.append("| Rating | Actual | Expected |")
        md.append("|--------|--------|----------|")
        for r in ['1', '2', '3', '4', '5']:
            act = actual.get(r, 0)
            exp = rating.get('expected_distribution', {}).get(r, 0.2)
            md.append(f"| {r}⭐ | {act:.1%} | {exp:.1%} |")
    md.append("")
    
    md.append("### Sentiment Consistency")
    sent = bias.get('sentiment_consistency', {})
    md.append(f"- **Inconsistency Rate:** {sent.get('inconsistency_rate', 0):.1%}")
    md.append(f"- **Status:** {'✅ Consistent' if sent.get('is_consistent') else '⚠️ Inconsistent'}")
    md.append("")
    
    # Realism Metrics
    md.append("\n## Realism Analysis\n")
    real = report['metrics']['realism']
    md.append(f"**Overall Realistic:** {'✅ Yes' if real.get('overall_realistic') else '⚠️ No'}")
    md.append("")
    
    md.append("### Readability")
    read = real.get('readability', {})
    md.append(f"- **Flesch Score:** {read.get('avg_flesch_score', 'N/A')}")
    md.append(f"- **Interpretation:** {read.get('interpretation', 'N/A')}")
    md.append("")
    
    md.append("### AI Patterns")
    ai = real.get('ai_patterns', {})
    md.append(f"- **AI Pattern Rate:** {ai.get('ai_pattern_rate', 0):.1%}")
    md.append(f"- **Status:** {'✅ Natural' if ai.get('is_realistic') else '⚠️ AI-like'}")
    md.append("")
    
    # Comparison with Real Reviews
    md.append("\n## Synthetic vs Real Comparison\n")
    comp = report.get('comparison')
    if comp:
        md.append(f"**Semantic Overlap:** {comp.get('semantic_overlap', 'N/A')}")
        md.append(f"\n_{comp.get('interpretation', '')}_")
    else:
        md.append("_No real reviews provided for comparison._")
    md.append("")
    
    # Model Performance (if present)
    perf = report.get('model_performance')
    if perf:
        md.append("\n## Model Performance\n")
        md.append(f"- **Total Generation Time:** {perf.get('total_time_seconds', 0):.2f}s")
        md.append(f"- **Avg Time per Review:** {perf.get('avg_time_per_review', 0):.2f}s")
        md.append(f"- **Model:** {perf.get('model_name', 'unknown')}")
        md.append("")
    
    # Recommendations
    md.append("\n## Recommendations\n")
    for rec in report.get('recommendations', []):
        md.append(f"- {rec}")
    md.append("")
    
    # Join and save
    content = "\n".join(md)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return content
