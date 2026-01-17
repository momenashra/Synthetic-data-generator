"""
Incremental review storage with separate embedding file support.
Appends validated reviews immediately to JSONL and stores embeddings in .npy file.
"""
import json
import os
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
from pathlib import Path


class ReviewStorage:
    """
    Manages incremental review storage with separate embedding files.
    
    Reviews are stored in JSONL format (one JSON object per line).
    Embeddings are stored in a separate .npy file for efficiency.
    
    File structure:
        data/generated_reviews.jsonl  - Review text, metadata
        data/review_embeddings.npy    - (N, 384) embedding matrix
    """
    
    def __init__(
        self, 
        reviews_path: str = 'data/generated_reviews.jsonl',
        embeddings_path: str = 'data/review_embeddings.npy'
    ):
        self.reviews_path = reviews_path
        self.embeddings_path = embeddings_path
        self._embedding_client = None
        self._cached_embeddings: Optional[np.ndarray] = None
        
        # Ensure data directory exists
        Path(reviews_path).parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def embedding_client(self):
        """Lazy load embedding client."""
        if self._embedding_client is None:
            from models import EmbeddingClient
            self._embedding_client = EmbeddingClient()
        return self._embedding_client
    
    def save_review(
        self, 
        review_data: Dict, 
        compute_embedding: bool = True
    ) -> None:
        """
        Immediately append a validated review to storage.
        
        Args:
            review_data: Review dictionary with 'text', 'rating', etc.
            compute_embedding: Whether to compute and store embedding
        """
        # Remove embedding from review_data if present (stored separately)
        review_copy = {k: v for k, v in review_data.items() if k != 'embedding'}
        
        # Compute embedding if requested
        if compute_embedding:
            embedding = self.embedding_client.encode_single(review_data['text'])
            self._append_embedding(embedding)
        
        # Append to JSONL
        with open(self.reviews_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(review_copy, ensure_ascii=False) + '\n')
        
        # Invalidate cache
        self._cached_embeddings = None
    
    def _append_embedding(self, embedding: np.ndarray) -> None:
        """Append a single embedding to the .npy file."""
        embedding = embedding.reshape(1, -1)  # Ensure 2D
        
        if os.path.exists(self.embeddings_path):
            existing = np.load(self.embeddings_path)
            updated = np.vstack([existing, embedding])
        else:
            updated = embedding
        
        np.save(self.embeddings_path, updated)
    
    def load_all_reviews(self) -> List[Dict]:
        """Load all reviews from JSONL."""
        if not os.path.exists(self.reviews_path):
            return []
        
        reviews = []
        with open(self.reviews_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    reviews.append(json.loads(line))
        return reviews
    
    def load_all_embeddings(self) -> np.ndarray:
        """
        Load all embeddings from .npy file.
        
        Returns:
            numpy array of shape (N, 384) or empty (0, 384) if no embeddings
        """
        if self._cached_embeddings is not None:
            return self._cached_embeddings
        
        if os.path.exists(self.embeddings_path):
            self._cached_embeddings = np.load(self.embeddings_path)
        else:
            self._cached_embeddings = np.empty((0, self.embedding_client.get_embedding_dim()))
        
        return self._cached_embeddings
    
    def get_review_count(self) -> int:
        """Get the number of stored reviews."""
        if not os.path.exists(self.reviews_path):
            return 0
        
        count = 0
        with open(self.reviews_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
    
    def get_embedding_count(self) -> int:
        """Get the number of stored embeddings."""
        if not os.path.exists(self.embeddings_path):
            return 0
        return len(np.load(self.embeddings_path))
    
    def clear(self) -> None:
        """Clear all stored reviews and embeddings (use with caution!)."""
        if os.path.exists(self.reviews_path):
            os.remove(self.reviews_path)
        if os.path.exists(self.embeddings_path):
            os.remove(self.embeddings_path)
        self._cached_embeddings = None
    
    def migrate_existing_reviews(self) -> int:
        """
        Generate embeddings for existing reviews that don't have them.
        
        Returns:
            Number of embeddings generated
        """
        reviews = self.load_all_reviews()
        if not reviews:
            return 0
        
        # Check if embeddings already exist and match count
        existing_count = self.get_embedding_count()
        if existing_count >= len(reviews):
            print(f"✅ Embeddings already exist for all {len(reviews)} reviews")
            return 0
        
        # Generate embeddings for reviews without them
        texts = [r['text'] for r in reviews]
        
        print(f"🔧 Generating embeddings for {len(texts)} reviews...")
        embeddings = self.embedding_client.encode(texts)
        
        # Save all embeddings
        np.save(self.embeddings_path, embeddings)
        self._cached_embeddings = embeddings
        
        print(f"✅ Generated and saved {len(embeddings)} embeddings to {self.embeddings_path}")
        return len(embeddings)
    
    def get_review_with_embedding(self, index: int) -> Dict:
        """
        Get a review with its corresponding embedding.
        
        Args:
            index: Index of the review
            
        Returns:
            Review dict with 'embedding' key added
        """
        reviews = self.load_all_reviews()
        embeddings = self.load_all_embeddings()
        
        if index >= len(reviews):
            raise IndexError(f"Review index {index} out of range")
        
        review = reviews[index].copy()
        if index < len(embeddings):
            review['embedding'] = embeddings[index]
        
        return review


# Convenience function for global access
_storage_instance: Optional[ReviewStorage] = None

def get_review_storage() -> ReviewStorage:
    """Get the global ReviewStorage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = ReviewStorage()
    return _storage_instance
