"""
Reranking service for improving retrieval quality.
Uses cross-encoders and other models to rerank search results.
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import numpy as np
from dataclasses import dataclass

from app.config import settings
from app.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RankedDocument:
    """Data class for ranked documents."""
    text: str
    score: float
    original_rank: int
    new_rank: int


class RerankerService:
    """
    Service for reranking search results to improve relevance.
    Supports multiple reranking strategies.
    """
    
    def __init__(
        self,
        model_type: str = "cross_encoder",
        model_name: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32
    ):
        """
        Initialize reranker service.
        
        Args:
            model_type: Type of reranking model
            model_name: Model identifier
            device: Device for model inference
            batch_size: Batch size for processing
        """
        self.model_type = model_type
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.device = device
        self.batch_size = batch_size
        
        # Initialize model
        self._model = None
        self._init_model()
        
        # Metrics
        self._rerank_count = 0
        self._total_time = 0.0
        
        logger.info(f"RerankerService initialized with {model_type} model: {self.model_name}")
    
    def _init_model(self):
        """Initialize the reranking model."""
        try:
            if self.model_type == "cross_encoder":
                from sentence_transformers import CrossEncoder
                
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=512
                )
                logger.info(f"Cross-encoder model loaded: {self.model_name}")
            
            elif self.model_type == "bi_encoder":
                from sentence_transformers import SentenceTransformer
                
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device
                )
                logger.info(f"Bi-encoder model loaded: {self.model_name}")
            
            else:
                logger.warning(f"Unknown model type: {self.model_type}, using score-based reranking")
                self._model = None
                
        except ImportError as e:
            logger.warning(f"Failed to load reranking model: {e}. Using score-based reranking.")
            self._model = None
        except Exception as e:
            logger.error(f"Error initializing reranker model: {e}")
            self._model = None
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return
            return_scores: Whether to include scores in results
            
        Returns:
            List of reranked documents with scores
        """
        if not documents:
            return []
        
        start_time = asyncio.get_event_loop().time()
        self._rerank_count += 1
        
        try:
            if self._model and self.model_type == "cross_encoder":
                scores = await self._cross_encoder_rerank(query, documents)
            elif self._model and self.model_type == "bi_encoder":
                scores = await self._bi_encoder_rerank(query, documents)
            else:
                scores = await self._score_based_rerank(query, documents, **kwargs)
            
            # Create ranked list
            ranked = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                ranked.append({
                    "text": doc,
                    "score": float(score),
                    "original_rank": i + 1,
                    "new_rank": 0  # Will be set after sorting
                })
            
            # Sort by score (descending)
            ranked.sort(key=lambda x: x["score"], reverse=True)
            
            # Update new ranks
            for i, item in enumerate(ranked, 1):
                item["new_rank"] = i
            
            # Limit to top_k
            if top_k:
                ranked = ranked[:top_k]
            
            # Update metrics
            self._total_time += asyncio.get_event_loop().time() - start_time
            
            logger.debug(f"Reranked {len(documents)} documents, returning top {len(ranked)}")
            
            return ranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order
            return [
                {
                    "text": doc,
                    "score": 1.0 - (i * 0.01),
                    "original_rank": i + 1,
                    "new_rank": i + 1
                }
                for i, doc in enumerate(documents[:top_k or len(documents)])
            ]
    
    async def _cross_encoder_rerank(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Rerank using cross-encoder model.
        
        Args:
            query: Search query
            documents: List of documents
            
        Returns:
            List of relevance scores
        """
        try:
            # Prepare pairs
            pairs = [[query, doc] for doc in documents]
            
            # Run inference in thread pool
            def predict():
                return self._model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False
                )
            
            scores = await asyncio.to_thread(predict)
            
            # Normalize scores to 0-1 range
            if len(scores) > 1:
                min_score = np.min(scores)
                max_score = np.max(scores)
                if max_score > min_score:
                    scores = (scores - min_score) / (max_score - min_score)
            
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return [1.0 - (i * 0.01) for i in range(len(documents))]
    
    async def _bi_encoder_rerank(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Rerank using bi-encoder model (cosine similarity).
        
        Args:
            query: Search query
            documents: List of documents
            
        Returns:
            List of similarity scores
        """
        try:
            def encode():
                query_embedding = self._model.encode(
                    [query],
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                doc_embeddings = self._model.encode(
                    documents,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size
                )
                return query_embedding, doc_embeddings
            
            query_emb, doc_embs = await asyncio.to_thread(encode)
            
            # Calculate cosine similarity
            scores = np.dot(doc_embs, query_emb.T).flatten()
            
            return scores.tolist()
            
        except Exception as e:
            logger.error(f"Bi-encoder reranking failed: {e}")
            return [1.0 - (i * 0.01) for i in range(len(documents))]
    
    async def _score_based_rerank(
        self,
        query: str,
        documents: List[str],
        **kwargs
    ) -> List[float]:
        """
        Score-based reranking using various heuristics.
        
        Args:
            query: Search query
            documents: List of documents
            **kwargs: Additional scoring parameters
            
        Returns:
            List of relevance scores
        """
        query_terms = set(query.lower().split())
        
        scores = []
        for doc in documents:
            score = 0.0
            doc_lower = doc.lower()
            doc_terms = set(doc_lower.split())
            
            # 1. Exact match bonus
            if query.lower() in doc_lower:
                score += 2.0
            
            # 2. Term frequency score
            term_matches = sum(1 for term in query_terms if term in doc_lower)
            score += term_matches / len(query_terms) * 3.0
            
            # 3. Jaccard similarity
            intersection = len(query_terms & doc_terms)
            union = len(query_terms | doc_terms)
            jaccard = intersection / union if union > 0 else 0
            score += jaccard * 2.0
            
            # 4. Position bias - earlier terms more important
            for i, term in enumerate(query_terms):
                if term in doc_lower:
                    position = doc_lower.index(term)
                    position_score = 1.0 / (1.0 + position / 100.0)
                    score += position_score * (1.0 - i * 0.1)
            
            # 5. Length normalization - prefer concise documents
            length_penalty = min(1.0, 200 / max(len(doc_lower), 1))
            score *= length_penalty
            
            scores.append(score)
        
        # Normalize scores
        if scores:
            max_score = max(scores)
            if max_score > 0:
                scores = [s / max_score for s in scores]
        
        return scores
    
    async def rerank_with_context(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        context_field: str = "text",
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents that have associated metadata/context.
        
        Args:
            query: Search query
            documents: List of document objects
            context_field: Field name for text content
            top_k: Number of top results
            
        Returns:
            Reranked documents with original metadata preserved
        """
        # Extract texts
        texts = [doc.get(context_field, "") for doc in documents]
        
        # Rerank
        reranked = await self.rerank(query, texts, top_k=top_k)
        
        # Map back to original documents
        text_to_doc = {doc.get(context_field, ""): doc for doc in documents}
        
        results = []
        for item in reranked:
            original_doc = text_to_doc.get(item["text"], {}).copy()
            original_doc.update({
                "rerank_score": item["score"],
                "rerank_rank": item["new_rank"],
                "original_rank": item["original_rank"]
            })
            results.append(original_doc)
        
        return results
    
    async def multi_stage_rerank(
        self,
        query: str,
        documents: List[str],
        stages: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Multi-stage reranking with progressive filtering.
        
        Args:
            query: Search query
            documents: List of documents
            stages: List of reranking stages ('score', 'bi_encoder', 'cross_encoder')
            top_k: Final number of results
            
        Returns:
            Reranked documents
        """
        stages = stages or ["score", "cross_encoder"]
        current_docs = documents[:]
        
        for stage_idx, stage in enumerate(stages):
            # Progressively reduce candidates
            current_k = top_k * (2 ** (len(stages) - stage_idx)) if top_k else None
            
            if stage == "score":
                scores = await self._score_based_rerank(query, current_docs)
            elif stage == "cross_encoder" and self._model:
                scores = await self._cross_encoder_rerank(query, current_docs)
            elif stage == "bi_encoder" and self._model:
                scores = await self._bi_encoder_rerank(query, current_docs)
            else:
                continue
            
            # Sort and filter
            scored = list(zip(current_docs, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            
            if current_k and current_k < len(scored):
                scored = scored[:current_k]
            
            current_docs = [doc for doc, _ in scored]
        
        # Final reranking with best available method
        return await self.rerank(query, current_docs, top_k=top_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "rerank_count": self._rerank_count,
            "avg_rerank_time": (
                self._total_time / self._rerank_count
                if self._rerank_count > 0 else 0
            ),
            "device": self.device,
            "batch_size": self.batch_size
        }