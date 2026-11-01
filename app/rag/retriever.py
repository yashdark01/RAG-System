"""
Retrieval service for finding relevant documents.
Implements multi-stage retrieval with reranking and fusion strategies.
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import numpy as np
from dataclasses import dataclass, field

from app.config import settings
from app.logging.logger import get_logger
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStoreService
from app.rag.reranker import RerankerService

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Data class for retrieval results."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = 1
    retrieval_method: str = "semantic"
    chunk_info: Optional[Dict[str, Any]] = None

    @property
    def meta(self) -> Dict[str, Any]:
        """Backward compatibility for meta property."""
        return self.metadata


class RetrieverService:
    """
    Advanced retrieval service with multi-stage retrieval pipeline.
    Supports semantic search, keyword search, and hybrid strategies.
    """
    
    def __init__(
        self,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService,
        reranker: Optional[RerankerService] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        reranking_enabled: bool = True,
        retrieval_strategy: str = "hybrid"
    ):
        """
        Initialize the retriever service.
        
        Args:
            vector_store: Vector store for semantic search
            embedding_service: Embedding service
            reranker: Reranking service
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            reranking_enabled: Whether to use reranking
            retrieval_strategy: Retrieval strategy (semantic, keyword, hybrid)
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.reranker = reranker
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.reranking_enabled = reranking_enabled
        self.retrieval_strategy = retrieval_strategy
        
        # Performance metrics
        self._query_count = 0
        self._total_time = 0.0
        self._cache_hits = 0
        
        # Simple query cache
        self._cache: Dict[str, List[RetrievalResult]] = {}
        
        logger.info(
            f"RetrieverService initialized with strategy={retrieval_strategy}, "
            f"top_k={top_k}, reranking={reranking_enabled}"
        )
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        retrieval_strategy: Optional[str] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Main retrieval method with multi-stage pipeline.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            use_cache: Whether to use result cache
            retrieval_strategy: Override retrieval strategy
            **kwargs: Additional parameters
            
        Returns:
            List of retrieval results
        """
        start_time = asyncio.get_event_loop().time()
        self._query_count += 1
        
        k = top_k or self.top_k
        strategy = retrieval_strategy or self.retrieval_strategy
        
        # Check cache
        cache_key = f"{query}:{k}:{strategy}:{str(filters)}"
        if use_cache and cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        try:
            # Stage 1: Initial retrieval based on strategy
            if strategy == "semantic":
                results = await self._semantic_retrieval(query, k * 2, filters)
            elif strategy == "keyword":
                results = await self._keyword_retrieval(query, k * 2, filters)
            elif strategy == "hybrid":
                results = await self._hybrid_retrieval(query, k * 2, filters)
            else:
                results = await self._semantic_retrieval(query, k * 2, filters)
            
            # Stage 2: Reranking (optional)
            if self.reranking_enabled and self.reranker and len(results) > k:
                results = await self._rerank_results(query, results, k)
            
            # Stage 3: Post-processing
            results = await self._post_process_results(results, k)
            
            # Update metrics
            self._total_time += asyncio.get_event_loop().time() - start_time
            
            # Cache results
            if use_cache and len(results) > 0:
                self._cache[cache_key] = results
                # Limit cache size
                if len(self._cache) > 1000:
                    # Remove oldest entries
                    keys = list(self._cache.keys())
                    for old_key in keys[:100]:
                        del self._cache[old_key]
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query[:100]}...': {e}")
            return []
    
    async def _semantic_retrieval(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Semantic retrieval using embeddings.
        
        Args:
            query: Search query
            k: Number of results
            filters: Metadata filters
            
        Returns:
            List of retrieval results
        """
        try:
            # Search vector store
            results = await self.vector_store.search(
                query=query,
                top_k=k,
                filter_metadata=filters,
                min_similarity=self.similarity_threshold
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in results:
                retrieval_results.append(
                    RetrievalResult(
                        id=result["id"],
                        text=result["text"],
                        score=result["score"],
                        metadata=result.get("metadata", {}),
                        rank=result.get("rank", len(retrieval_results) + 1),
                        retrieval_method="semantic"
                    )
                )
            
            logger.debug(f"Semantic retrieval returned {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            return []
    
    async def _keyword_retrieval(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Keyword-based retrieval using text matching.
        
        Args:
            query: Search query
            k: Number of results
            filters: Metadata filters
            
        Returns:
            List of retrieval results
        """
        try:
            # Tokenize query
            query_tokens = [w for w in query.lower().split() if len(w) > 1]
            if not query_tokens:
                return []

            # Retrieve candidate documents
            doc_ids = list(self.vector_store._metadata_index.keys())[:1000]
            all_docs = await self.vector_store.get_documents(doc_ids)
            valid_docs = [d for d in all_docs if d and d.get("text")]

            if not valid_docs:
                return []

            scored_docs = []
            try:
                from rank_bm25 import BM25Okapi
                corpus = [d["text"].lower().split() for d in valid_docs]
                bm25 = BM25Okapi(corpus)
                scores = bm25.get_scores(query_tokens)
                max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
                
                for doc, raw_score in zip(valid_docs, scores):
                    norm_score = float(raw_score / max_score)
                    if norm_score > 0.05:
                        scored_docs.append((doc, norm_score))
            except ImportError:
                query_terms = set(query_tokens)
                for doc in valid_docs:
                    doc_terms = set(doc["text"].lower().split())
                    intersection = len(query_terms & doc_terms)
                    union = len(query_terms | doc_terms)
                    score = intersection / union if union > 0 else 0.0
                    if score > 0.05:
                        scored_docs.append((doc, score))

            scored_docs.sort(key=lambda x: x[1], reverse=True)

            results = []
            for i, (doc, score) in enumerate(scored_docs[:k]):
                if filters and not self._matches_filters(doc.get("metadata", {}), filters):
                    continue
                results.append(
                    RetrievalResult(
                        id=doc["id"],
                        text=doc["text"],
                        score=score,
                        metadata=doc.get("metadata", {}),
                        rank=i + 1,
                        retrieval_method="keyword"
                    )
                )
            logger.debug(f"BM25 keyword retrieval returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Keyword retrieval failed: {e}")
            return []
    
    async def _hybrid_retrieval(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining semantic and keyword search.
        Uses reciprocal rank fusion for combining results.
        
        Args:
            query: Search query
            k: Number of results
            filters: Metadata filters
            
        Returns:
            List of retrieval results
        """
        try:
            # Run both retrieval methods in parallel
            semantic_task = asyncio.create_task(
                self._semantic_retrieval(query, k, filters)
            )
            keyword_task = asyncio.create_task(
                self._keyword_retrieval(query, k, filters)
            )
            
            semantic_results, keyword_results = await asyncio.gather(
                semantic_task, keyword_task
            )
            
            # Combine results using Reciprocal Rank Fusion (RRF)
            fused_scores = {}
            all_results = {}
            
            # RRF constant
            k_rrf = 60
            
            # Score from semantic results
            for rank, result in enumerate(semantic_results, 1):
                rrf_score = 1.0 / (k_rrf + rank)
                fused_scores[result.id] = fused_scores.get(result.id, 0) + rrf_score
                all_results[result.id] = result
            
            # Score from keyword results
            for rank, result in enumerate(keyword_results, 1):
                rrf_score = 1.0 / (k_rrf + rank)
                fused_scores[result.id] = fused_scores.get(result.id, 0) + rrf_score
                
                # Update with keyword result if semantic result exists
                if result.id in all_results:
                    existing = all_results[result.id]
                    existing.score = max(existing.score, result.score)
                    existing.retrieval_method = "hybrid"
                else:
                    all_results[result.id] = result
                    result.retrieval_method = "hybrid"
            
            # Sort by fused scores
            sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
            
            # Build final results
            hybrid_results = []
            for rank, doc_id in enumerate(sorted_ids[:k], 1):
                result = all_results[doc_id]
                result.score = fused_scores[doc_id]
                result.rank = rank
                hybrid_results.append(result)
            
            logger.debug(
                f"Hybrid retrieval: semantic={len(semantic_results)}, "
                f"keyword={len(keyword_results)}, fused={len(hybrid_results)}"
            )
            
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            # Fallback to semantic retrieval
            return await self._semantic_retrieval(query, k, filters)
    
    async def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results using cross-encoder or other reranking model.
        
        Args:
            query: Search query
            results: Initial retrieval results
            top_k: Number of top results to return
            
        Returns:
            Reranked results
        """
        if not self.reranker:
            return results[:top_k]
        
        try:
            # Prepare documents for reranking
            documents = [result.text for result in results]
            
            # Rerank
            reranked = await self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k
            )
            
            # Update scores and ranks
            for i, reranked_doc in enumerate(reranked):
                if i < len(results):
                    results[i].score = reranked_doc.get("score", results[i].score)
                    results[i].rank = i + 1
                    results[i].retrieval_method += "+reranked"
            
            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug(f"Reranked {len(results)} results, returning top {top_k}")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k]
    
    async def _post_process_results(
        self,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Post-process retrieval results.
        - Remove duplicates
        - Ensure diversity
        - Format outputs
        
        Args:
            results: Retrieved results
            top_k: Number of results to keep
            
        Returns:
            Post-processed results
        """
        if not results:
            return []
        
        # Remove duplicates based on text similarity
        unique_results = []
        seen_texts = set()
        
        for result in results:
            # Simple deduplication using text hash
            text_hash = hash(result.text[:100])  # First 100 chars as signature
            
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_results.append(result)
            
            if len(unique_results) >= top_k:
                break
        
        # Ensure diversity - penalize very similar results
        if len(unique_results) > 1:
            diverse_results = [unique_results[0]]
            
            for result in unique_results[1:]:
                # Check similarity with already selected results
                is_diverse = True
                for selected in diverse_results:
                    # Simple overlap check
                    overlap = len(set(result.text.split()) & set(selected.text.split()))
                    if overlap / len(set(result.text.split())) > 0.8:
                        is_diverse = False
                        break
                
                if is_diverse:
                    diverse_results.append(result)
                
                if len(diverse_results) >= top_k:
                    break
            
            unique_results = diverse_results
        
        # Update ranks
        for i, result in enumerate(unique_results, 1):
            result.rank = i
        
        return unique_results[:top_k]
    
    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            meta_value = metadata[key]
            if isinstance(value, list):
                if meta_value not in value:
                    return False
            elif meta_value != value:
                return False
        return True
    
    async def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Retrieve documents with just text and scores.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (text, score) tuples
        """
        results = await self.retrieve(query, top_k=top_k, **kwargs)
        return [(result.text, result.score) for result in results]
    
    async def retrieve_context(
        self,
        query: str,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Retrieve and format context for LLM input.
        
        Args:
            query: Search query
            max_tokens: Maximum tokens for context
            **kwargs: Additional retrieval parameters
            
        Returns:
            Formatted context string
        """
        results = await self.retrieve(query, **kwargs)
        
        if not results:
            return ""
        
        # Build context string
        context_parts = []
        token_count = 0
        
        for i, result in enumerate(results, 1):
            chunk = f"[Document {i}]\n"
            if result.meta.get("title"):
                chunk += f"Title: {result.meta['title']}\n"
            chunk += f"{result.text}\n"
            if result.meta.get("source"):
                chunk += f"Source: {result.meta['source']}\n"
            chunk += f"Relevance: {result.score:.3f}\n\n"
            
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            estimated_tokens = len(chunk) // 4
            
            if token_count + estimated_tokens > max_tokens:
                break
            
            context_parts.append(chunk)
            token_count += estimated_tokens
        
        return "\n".join(context_parts)
    
    async def multi_query_retrieval(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        aggregation: str = "fusion"
    ) -> List[RetrievalResult]:
        """
        Retrieve documents for multiple queries and aggregate results.
        
        Args:
            queries: List of query strings
            top_k: Number of final results
            aggregation: Aggregation method (fusion, max, mean)
            
        Returns:
            Aggregated retrieval results
        """
        if not queries:
            return []
        
        # Retrieve for each query
        all_results = []
        for query in queries:
            results = await self.retrieve(query, top_k=top_k or self.top_k)
            all_results.append(results)
        
        # Aggregate results
        if aggregation == "fusion":
            # Reciprocal rank fusion
            fused_scores = {}
            result_map = {}
            
            for results in all_results:
                for rank, result in enumerate(results, 1):
                    rrf_score = 1.0 / (60 + rank)
                    fused_scores[result.id] = fused_scores.get(result.id, 0) + rrf_score
                    result_map[result.id] = result
            
            # Sort by fused score
            sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
            
            aggregated = []
            for i, doc_id in enumerate(sorted_ids[:top_k or self.top_k], 1):
                result = result_map[doc_id]
                result.score = fused_scores[doc_id]
                result.rank = i
                aggregated.append(result)
            
            return aggregated
        
        elif aggregation == "max":
            # Take max score for each document
            score_map = {}
            result_map = {}
            
            for results in all_results:
                for result in results:
                    if result.id not in score_map or result.score > score_map[result.id]:
                        score_map[result.id] = result.score
                        result_map[result.id] = result
            
            sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)
            
            aggregated = []
            for i, doc_id in enumerate(sorted_ids[:top_k or self.top_k], 1):
                result = result_map[doc_id]
                result.rank = i
                aggregated.append(result)
            
            return aggregated
        
        else:
            # Simple concatenation and deduplication
            seen = set()
            aggregated = []
            for results in all_results:
                for result in results:
                    if result.id not in seen:
                        seen.add(result.id)
                        aggregated.append(result)
            
            aggregated.sort(key=lambda x: x.score, reverse=True)
            
            for i, result in enumerate(aggregated[:top_k or self.top_k], 1):
                result.rank = i
            
            return aggregated[:top_k or self.top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever performance statistics."""
        cache_hit_rate = self._cache_hits / self._query_count if self._query_count > 0 else 0
        
        return {
            "total_queries": self._query_count,
            "avg_retrieval_time": (
                self._total_time / self._query_count if self._query_count > 0 else 0
            ),
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "strategy": self.retrieval_strategy,
            "reranking_enabled": self.reranking_enabled,
            "default_top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold
        }
    
    def clear_cache(self):
        """Clear the retrieval cache."""
        self._cache.clear()
        self._cache_hits = 0
        logger.info("Retrieval cache cleared")

    async def retrieve_langchain_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Retrieve results as native LangChain Document instances."""
        from langchain_core.documents import Document
        results = await self.retrieve(query=query, top_k=top_k, filters=filters)
        return [
            Document(page_content=r.text, metadata=r.metadata)
            for r in results
        ]