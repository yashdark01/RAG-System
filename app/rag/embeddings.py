"""
Embedding service for generating text embeddings.
Supports multiple embedding models with caching and batching.
"""

import asyncio
import hashlib
from typing import List, Optional, Union
import numpy as np
from functools import lru_cache
import time

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import torch

from app.config import settings
from app.logging.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating and managing text embeddings.
    Supports HuggingFace models with optional GPU acceleration.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        cache_size: int = 10000
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model
            dimension: Expected embedding dimension
            device: Device to run model on ('cpu', 'cuda', 'mps')
            normalize_embeddings: Whether to L2-normalize embeddings
            cache_size: Size of embedding cache
        """
        self.model_name = model_name
        self.dimension = dimension
        self.normalize = normalize_embeddings
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Initialize models
        self._init_models()
        
        # Embedding cache
        self._cache = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance metrics
        self._total_embeddings = 0
        self._total_time = 0.0
        
        logger.info(
            f"EmbeddingService initialized with model={model_name}, "
            f"device={self.device}, dimension={dimension}"
        )

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Get the LangChain compatible embeddings instance."""
        return self.langchain_model
    
    def _init_models(self):
        """Initialize both LangChain and SentenceTransformer models."""
        try:
            # LangChain compatible model
            self.langchain_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={
                    'normalize_embeddings': self.normalize,
                    'batch_size': 32
                }
            )
            
            # Direct SentenceTransformer for batch processing
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Verify dimension
            test_embedding = self.model.encode(["test"])
            if test_embedding.shape[1] != self.dimension:
                logger.warning(
                    f"Model dimension mismatch. Expected {self.dimension}, "
                    f"got {test_embedding.shape[1]}"
                )
                self.dimension = test_embedding.shape[1]
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {e}")
            raise
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _cache_get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        key = self._cache_key(text)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        self._cache_misses += 1
        return None
    
    def _cache_set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache with LRU eviction."""
        key = self._cache_key(text)
        
        # Evict oldest items if cache is full
        if len(self._cache) >= self._cache_size:
            # Remove 10% of oldest entries
            num_to_remove = self._cache_size // 10
            for old_key in list(self._cache.keys())[:num_to_remove]:
                del self._cache[old_key]
        
        self._cache[key] = embedding
    
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embedding
        """
        # Check cache
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        
        try:
            start_time = time.time()
            
            # Generate embedding
            embedding = await asyncio.to_thread(
                self.model.encode,
                [text],
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )
            
            embedding = embedding[0]
            
            # Update metrics
            self._total_embeddings += 1
            self._total_time += time.time() - start_time
            
            # Cache result
            self._cache_set(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (n_texts x dimension)
        """
        if not texts:
            return np.array([])
        
        # Check cache for each text
        uncached_texts = []
        uncached_indices = []
        embeddings = [None] * len(texts)
        
        for i, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                embeddings[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if not uncached_texts:
            return np.array(embeddings)
        
        try:
            start_time = time.time()
            
            # Generate embeddings in batches
            new_embeddings = await asyncio.to_thread(
                self.model.encode,
                uncached_texts,
                batch_size=batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress
            )
            
            # Cache new embeddings and update list
            for idx, text in zip(uncached_indices, uncached_texts):
                embedding = new_embeddings[len([e for e in embeddings[:idx] if e is None])]
                embeddings[idx] = embedding
                self._cache_set(text, embedding)
            
            # Update metrics
            self._total_embeddings += len(uncached_texts)
            self._total_time += time.time() - start_time
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding specifically for queries.
        May use different encoding parameters than document embeddings.
        
        Args:
            query: Query text to embed
            
        Returns:
            numpy array of query embedding
        """
        # Queries might benefit from different encoding parameters
        return await self.embed_text(query)
    
    async def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Generate embeddings for documents.
        May use different encoding parameters than queries.
        
        Args:
            documents: List of document texts
            
        Returns:
            numpy array of document embeddings
        """
        return await self.embed_texts(documents, batch_size=64)
    
    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get LangChain-compatible embeddings instance.
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        return self.langchain_model
    
    def get_cache_stats(self) -> dict:
        """Get embedding cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self._cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_embeddings": self._total_embeddings,
            "avg_embedding_time": self._total_time / self._total_embeddings if self._total_embeddings > 0 else 0
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")
    
    async def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        if self.normalize:
            # For normalized vectors, dot product equals cosine similarity
            return float(np.dot(embedding1, embedding2))
        else:
            # Calculate cosine similarity manually
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
    
    async def batch_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate similarities between query and multiple documents.
        
        Args:
            query_embedding: Query embedding
            document_embeddings: Array of document embeddings
            
        Returns:
            Array of similarity scores
        """
        if self.normalize:
            return np.dot(document_embeddings, query_embedding)
        else:
            query_norm = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(document_embeddings, axis=1)
            
            dot_products = np.dot(document_embeddings, query_embedding)
            norms = doc_norms * query_norm
            
            # Avoid division by zero
            norms = np.where(norms == 0, 1e-8, norms)
            
            return dot_products / norms