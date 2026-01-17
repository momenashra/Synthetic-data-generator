"""
Shared embedding model for semantic analysis.
Provides a singleton instance used across all quality metrics and review storage.
"""
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union
import numpy as np
import os
import sys

class EmbeddingClient:
    """
    Singleton embedding client for semantic analysis.
    
    Uses sentence-transformers 'all-MiniLM-L6-v2' model (384 dimensions).
    Lazily loads the model on first use to avoid startup overhead.
    """
    
    _instance: Optional['EmbeddingClient'] = None
    _model: Optional[SentenceTransformer] = None
    MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "intfloat/e5-small-v2")
    EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 384))
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            print(f"🔧 Loading embedding model: {self.MODEL_NAME}...", file=sys.stderr)
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        normalize: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (n_texts, 384)
        """
        model = self._load_model()
        if isinstance(texts, str):
            texts = [texts]
        return model.encode(
            texts, 
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        )
    
    def encode_single(self, text: str, normalize: bool = False) -> np.ndarray:
        """
        Encode a single text into embedding.
        
        Args:
            text: Text to encode
            normalize: Whether to L2-normalize embedding
            
        Returns:
            numpy array of shape (384,)
        """
        return self.encode([text], normalize=normalize)[0]
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension (384)."""
        return self.EMBEDDING_DIM
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
