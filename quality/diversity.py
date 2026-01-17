"""
Diversity metrics for synthetic reviews.
Measures vocabulary overlap and semantic similarity.
"""
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# SentenceTransformer now accessed via shared EmbeddingClient
from transformers import AutoTokenizer
from collections import Counter
import math
import zlib
from scipy.special import softmax
from sklearn.cluster import KMeans
from diversity import ngram_diversity_score, compression_ratio
from models import EmbeddingClient
import os
class DiversityAnalyzer:
    """Analyzes diversity of synthetic reviews."""
    
    def __init__(self):
        """Initialize the diversity analyzer."""
        self._embedding_client = None
        self.tokenizer_name = os.environ.get("TOKENIZER_NAME", "intfloat/e5-small-v2")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
    def _get_embedding_client(self):
        """Get the shared embedding client."""
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient()
        return self._embedding_client
    
    def calculate_vocabulary_overlap(self, reviews: List[str]) -> Dict[str, float]:
        """
        Calculate vocabulary diversity metrics.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with vocabulary metrics
            Returns:
                {
                    type_token_ratio,
                    unique_tokens,
                    total_tokens,
                    avg_unique_per_review,
                    top_10_token_ratio
                }
        """
        # Tokenize all reviews using the embedding model's tokenizer
        # This provides a more accurate view of "vocabulary" as seen by the model

        all_tokens = []
        review_tokens = []
        
        for review in reviews:
            # Tokenize and filter special tokens if needed
            tokens = self.tokenizer.tokenize(review.lower())
            # Filter out pure punctuation if desired, but subwords like '##ing' are valuable
            tokens = [t for t in tokens if t not in self.tokenizer.all_special_tokens]
            review_tokens.append(tokens)
            all_tokens.extend(tokens)
        
        # Calculate metrics
        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        
        # Type-Token Ratio (TTR)
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        # Average unique tokens per review
        avg_unique_per_review = np.mean([len(set(tokens)) for tokens in review_tokens])
        
        # Token frequency distribution
        token_freq = Counter(all_tokens)
        top_10_freq = sum([count for _, count in token_freq.most_common(10)])
        top_10_ratio = top_10_freq / total_tokens if total_tokens > 0 else 0
        
        return {
            'type_token_ratio': round(ttr, 4),
            'unique_tokens': unique_tokens,
            'total_tokens': total_tokens,
            'avg_unique_per_review': round(float(avg_unique_per_review), 2),
            'top_10_token_ratio': round(top_10_ratio, 4)
        }
    
    def calculate_semantic_similarity(self, reviews: List[str], sample_size: int = 100) -> Dict[str, float]:
        """
        Calculate semantic similarity between reviews using embeddings.
        
        Args:
            reviews: List of review texts
            sample_size: Number of review pairs to sample for efficiency
            
        Returns:
            Dictionary with similarity metrics
        """
        if len(reviews) < 2:
            return {
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'std_similarity': 0.0,
                'embedding_diversity': 1.0
            }

        
        # Load model and generate embeddings
        embedding_client = self._get_embedding_client()
        embeddings = embedding_client.encode(reviews)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal)
        n = len(reviews)
        upper_triangle_indices = np.triu_indices(n, k=1)
        similarity_values = similarities[upper_triangle_indices]
        
        # Sample if too many pairs
        if len(similarity_values) > sample_size:
            similarity_values = np.random.choice(similarity_values, sample_size, replace=False)
        
        return {
            'avg_similarity': round(float(np.mean(similarity_values)), 4),
            'max_similarity': round(float(np.max(similarity_values)), 4),
            'min_similarity': round(float(np.min(similarity_values)), 4),
            'std_similarity': round(float(np.std(similarity_values)), 4), #Low std + high mean = uniform collapse.
            'embedding_diversity': round(1 - float(np.mean(similarity_values)), 4)
        }

    def calculate_lexical_diversity(self, reviews: List[str]) -> Dict[str, float]:
        """
        Calculate advanced lexical diversity metrics.
        Measures surface-form variation (words, phrases, redundancy).
        """
        if not reviews:
            return {}

        # --- Normalize text once ---
        tokenized_reviews = self.tokenizer.tokenize(review.lower())

        # Reconstruct normalized text for library calls
        normalized_reviews = [" ".join(tokens) for tokens in tokenized_reviews]

        # --- Distinct n-grams ---
        distinct_1 = ngram_diversity_score(normalized_reviews, 1)
        distinct_2 = ngram_diversity_score(normalized_reviews, 2)
        distinct_3 = ngram_diversity_score(normalized_reviews, 3)

        # --- N-gram Entropy ---
        def get_entropy(token_lists, n=2):
            ngrams = []
            for tokens in token_lists:
                if len(tokens) >= n:
                    ngrams.extend(
                        [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
                    )

            if not ngrams:
                return 0.0, 0.0

            counts = Counter(ngrams)
            total = sum(counts.values())
            probs = [c / total for c in counts.values()]

            entropy = -sum(p * math.log(p) for p in probs)

            # Correct ideal entropy
            ideal_entropy = math.log(len(counts))
            normalized_entropy = entropy / ideal_entropy if ideal_entropy > 0 else 0.0

            return round(entropy, 4), round(normalized_entropy, 4)

        ent_2, norm_ent_2 = get_entropy(tokenized_reviews, 2)

        # --- Compression diversity ---
        compression = compression_ratio(normalized_reviews)
        compression_diversity = 1 / compression if compression > 0 else 0.0

        return {
            "distinct_1": round(distinct_1, 4),
            "distinct_2": round(distinct_2, 4),
            "distinct_3": round(distinct_3, 4),
            "ngram_entropy_2": ent_2,
            "normalized_ngram_entropy_2": norm_ent_2,
            "compression_ratio_diversity": round(compression_diversity, 4),
        }


    def calculate_dcscore(self, reviews: List[str]) -> float:
        """
        Calculate DCScore for semantic diversity.
        """
        if len(reviews) < 2:
            return 0.0
            
        embedding_client = self._get_embedding_client()
        embeddings = embedding_client.encode(reviews, normalize=True)
        
        # Step 3: Cosine similarity
        pairwise_matrix = embeddings @ embeddings.T
        
        # Step 4: Row-level softmax
        softmax_matrix = softmax(pairwise_matrix, axis=1)
        
        # Step 5: Mean of diagonal
        return round(float(np.mean(np.diag(softmax_matrix))), 4)

    def calculate_cluster_inertia(self, reviews: List[str], n_clusters: int = 10) -> float:
        """
        Calculate KMeans inertia of embeddings.
        """
        if len(reviews) < n_clusters:
            return 0.0
            
        embedding_client = self._get_embedding_client()
        embeddings = embedding_client.encode(reviews)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(embeddings)
        return round(float(kmeans.inertia_), 2)

    def calculate_syntactic_diversity(self, reviews: List[str]) -> Dict[str, float]:
        """
        Calculate syntactic diversity using CR-POS.
        """
        if not reviews:
            return {'cr_pos': 0.0}
            
        pos_sequences = []
        for review in reviews:
            tokens = nltk.word_tokenize(review)
            pos_tags = nltk.pos_tag(tokens)
            pos_str = " ".join([tag for _, tag in pos_tags])
            pos_sequences.append(pos_str)
            
        # Concatenate for compression
        combined_pos = "\n".join(pos_sequences).encode('utf-8')
        compressed = zlib.compress(combined_pos)
        
        # Compression Ratio = Original / Compressed
        ratio = len(combined_pos) / len(compressed) if len(compressed) > 0 else 1
        return {'cr_pos_diversity': round(1/ratio if ratio > 0 else 0, 4)}
    
    def calculate_tfidf_diversity(self, reviews: List[str]) -> Dict[str, float]:
        """
        Calculate diversity using TF-IDF vectors.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with TF-IDF diversity metrics
        """
        if len(reviews) < 2:
            return {'avg_tfidf_similarity': 0.0}
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(reviews)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # Get upper triangle (excluding diagonal)
        n = len(reviews)
        upper_triangle_indices = np.triu_indices(n, k=1)
        similarity_values = similarities[upper_triangle_indices]
        
        return {
            'avg_tfidf_similarity': round(float(np.mean(similarity_values)), 4),
            'std_tfidf_similarity': round(float(np.std(similarity_values)), 4)
        }
    
    def semantic_deduplication(self, reviews: List[str], threshold: float = 0.9) -> List[int]:
        """
        Identify indices of reviews to keep after deduplication.
        Filters out reviews with cosine similarity > threshold.
        """
        if len(reviews) < 2:
            return [0] if reviews else []
            
        embedding_client = self._get_embedding_client()
        embeddings = embedding_client.encode(reviews)
        similarities = cosine_similarity(embeddings)
        
        keep_indices = []
        discarded_indices = set()
        
        for i in range(len(reviews)):
            if i in discarded_indices:
                continue
            keep_indices.append(i)
            # Mark all subsequent similar reviews as discarded
            for j in range(i + 1, len(reviews)):
                if similarities[i, j] > threshold:
                    discarded_indices.add(j)
                    
        return keep_indices

    def analyze(self, reviews: List[str]) -> Dict[str, any]:
        """
        Perform complete diversity analysis.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with all diversity metrics
        """
        vocab_metrics = self.calculate_vocabulary_overlap(reviews)
        semantic_metrics = self.calculate_semantic_similarity(reviews)
        tfidf_metrics = self.calculate_tfidf_diversity(reviews)
        lexical_metrics = self.calculate_lexical_diversity(reviews)
        dc_score = self.calculate_dcscore(reviews)
        inertia = self.calculate_cluster_inertia(reviews)
        syntactic_metrics = self.calculate_syntactic_diversity(reviews)
        
        return {
            'vocabulary': vocab_metrics,
            'semantic_similarity': semantic_metrics,
            'tfidf_similarity': tfidf_metrics,
            'lexical_diversity': lexical_metrics,
            'dcscore': dc_score,
            'cluster_inertia': inertia,
            'syntactic_diversity': syntactic_metrics
        }
