"""
Bias detection for synthetic reviews.
Identifies sentiment skew and unrealistic patterns.
"""
from typing import List, Dict
import numpy as np
from collections import Counter
import re
from models import EmbeddingClient
import numpy as np


class BiasAnalyzer:
    """Analyzes bias in synthetic reviews."""
    
    def __init__(self):
        """Initialize the bias analyzer."""
        pass
    
    def analyze_rating_distribution(self, ratings: List[int], expected_dist: Dict[int, float]) -> Dict[str, any]:
        """
        Compare actual rating distribution to expected distribution.
        
        Args:
            ratings: List of ratings from generated reviews
            expected_dist: Expected distribution from config
            
        Returns:
            Dictionary with distribution analysis
        """
        # Calculate actual distribution
        rating_counts = Counter(ratings)
        total = len(ratings)
        actual_dist = {r: rating_counts.get(r, 0) / total for r in range(1, 6)}
        
        # Calculate deviation from expected
        deviations = {}
        for rating in range(1, 6):
            expected = expected_dist.get(rating, 0)
            actual = actual_dist.get(rating, 0)
            deviation = abs(actual - expected)
            deviations[rating] = round(deviation, 4)
        
        # Calculate chi-square-like statistic
        total_deviation = sum(deviations.values())
        
        return {
            'actual_distribution': {k: round(v, 4) for k, v in actual_dist.items()},
            'expected_distribution': {k: round(v, 4) for k, v in expected_dist.items()},
            'deviations': deviations,
            'total_deviation': round(total_deviation, 4),
            'is_biased': bool(total_deviation > 0.15)  # Threshold for significant bias
        }
    
    def analyze_sentiment_consistency(self, reviews: List[Dict]) -> Dict[str, any]:
        """
        Check if sentiment matches ratings using semantic embeddings.
        
        Uses cosine similarity to sentiment anchor embeddings for more
        accurate sentiment detection than simple word counting.
        
        Args:
            reviews: List of review dictionaries with 'rating' and 'text'
            
        Returns:
            Dictionary with sentiment consistency metrics
        """
        if not reviews:
            return {
                'total_reviews': 0,
                'inconsistencies_found': 0,
                'inconsistency_rate': 0,
                'is_consistent': True,
                'examples': []
            }
        
        embedding_client = EmbeddingClient()
        
        # Define sentiment anchor phrases
        positive_anchors = [
            "excellent product I love it",
            "highly recommend great quality",
            "fantastic amazing experience",
            "best purchase very satisfied"
        ]
        negative_anchors = [
            "terrible product I hate it",
            "do not buy poor quality",
            "awful horrible experience",
            "worst purchase very disappointed"
        ]
        
        # Compute sentiment centroids
        pos_embeddings = embedding_client.encode(positive_anchors)
        neg_embeddings = embedding_client.encode(negative_anchors)
        pos_centroid = pos_embeddings.mean(axis=0)
        neg_centroid = neg_embeddings.mean(axis=0)
        
        # Normalize centroids
        pos_centroid = pos_centroid / np.linalg.norm(pos_centroid)
        neg_centroid = neg_centroid / np.linalg.norm(neg_centroid)
        
        # Encode all review texts
        review_texts = [r['text'] for r in reviews]
        review_embeddings = embedding_client.encode(review_texts)
        
        inconsistencies = []
        
        for i, review in enumerate(reviews):
            rating = review['rating']
            embedding = review_embeddings[i]
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            embedding = embedding / norm
            
            # Compute similarities
            pos_sim = float(np.dot(embedding, pos_centroid))
            neg_sim = float(np.dot(embedding, neg_centroid))
            
            # Determine sentiment direction
            is_positive = pos_sim > neg_sim
            sim_diff = abs(pos_sim - neg_sim)
            
            # Check for inconsistencies (only if clear sentiment signal)
            if sim_diff > 0.05:  # Threshold for meaningful difference
                if rating >= 4 and not is_positive:
                    inconsistencies.append({
                        'rating': rating,
                        'issue': 'High rating but negative sentiment',
                        'pos_similarity': round(pos_sim, 4),
                        'neg_similarity': round(neg_sim, 4)
                    })
                elif rating <= 2 and is_positive:
                    inconsistencies.append({
                        'rating': rating,
                        'issue': 'Low rating but positive sentiment',
                        'pos_similarity': round(pos_sim, 4),
                        'neg_similarity': round(neg_sim, 4)
                    })
        
        total = len(reviews)
        inconsistent_count = len(inconsistencies)
        
        return {
            'total_reviews': total,
            'inconsistencies_found': inconsistent_count,
            'inconsistency_rate': round(inconsistent_count / total, 4) if total else 0,
            'is_consistent': bool(inconsistent_count / total < 0.1) if total else True,
            'examples': inconsistencies[:5]  # Show first 5 examples
        }
    
    def detect_repetitive_patterns(self, reviews: List[str]) -> Dict[str, any]:
        """
        Detect repetitive phrases or patterns.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with pattern detection results
        """
        # Extract common phrases (3-5 words)
        phrase_counter = Counter()
        
        for review in reviews:
            words = review.lower().split()
            # Extract 3-grams and 4-grams
            for n in [3, 4]:
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    # Filter out very common phrases
                    if len(phrase) > 10:  # Minimum phrase length
                        phrase_counter[phrase] += 1
        
        # Find phrases that appear too frequently
        total_reviews = len(reviews)
        repetitive_phrases = []
        
        for phrase, count in phrase_counter.most_common(20):
            frequency = count / total_reviews
            if frequency > 0.15:  # Appears in >15% of reviews
                repetitive_phrases.append({
                    'phrase': phrase,
                    'count': count,
                    'frequency': round(frequency, 4)
                })
        
        # Check for identical reviews
        unique_reviews = len(set(reviews))
        duplicate_rate = 1 - (unique_reviews / total_reviews) if total_reviews > 0 else 0
        
        return {
            'total_reviews': total_reviews,
            'unique_reviews': unique_reviews,
            'duplicate_rate': round(duplicate_rate, 4),
            'repetitive_phrases': repetitive_phrases,
            'has_repetition_issues': bool(len(repetitive_phrases) > 5 or duplicate_rate > 0.05)
        }
    
    def analyze_length_distribution(self, reviews: List[str]) -> Dict[str, any]:
        """
        Analyze review length distribution for unrealistic uniformity.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with length distribution metrics
        """
        lengths = [len(review.split()) for review in reviews]
        
        return {
            'avg_length': round(np.mean(lengths), 2),
            'std_length': round(np.std(lengths), 2),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'is_too_uniform': bool(np.std(lengths) < 10)  # Very low variance is suspicious
        }
    
    def analyze(self, reviews: List[Dict], expected_rating_dist: Dict[int, float]) -> Dict[str, any]:
        """
        Perform complete bias analysis.
        
        Args:
            reviews: List of review dictionaries with 'rating' and 'text'
            expected_rating_dist: Expected rating distribution from config
            
        Returns:
            Dictionary with all bias metrics
        """
        ratings = [r['rating'] for r in reviews]
        texts = [r['text'] for r in reviews]
        
        rating_analysis = self.analyze_rating_distribution(ratings, expected_rating_dist)
        sentiment_analysis = self.analyze_sentiment_consistency(reviews)
        pattern_analysis = self.detect_repetitive_patterns(texts)
        length_analysis = self.analyze_length_distribution(texts)
        # Calculate weighted bias score
        bias_severity_score = 0.0
        
        # heavy weights for critical issues
        if rating_analysis['is_biased']:
            bias_severity_score += 1.0
            
        if not sentiment_analysis['is_consistent']:
            bias_severity_score += 1.0
            
        # lighter weights for minor issues
        if pattern_analysis['has_repetition_issues']:
            bias_severity_score += 0.5
            
        if length_analysis['is_too_uniform']:
            bias_severity_score += 0.5

        # Overall bias assessment (Threshold > 1.5 means at least 2 major or 1 major + 2 minor issues)
        return {
            'rating_distribution': rating_analysis,
            'sentiment_consistency': sentiment_analysis,
            'repetitive_patterns': pattern_analysis,
            'length_distribution': length_analysis,
            'overall_bias_detected': bool(bias_severity_score > 1.5)
        }
