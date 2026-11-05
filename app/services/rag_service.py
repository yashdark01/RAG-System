"""
RAG (Retrieval-Augmented Generation) service.
Orchestrates retrieval and generation for question answering.
"""

from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
import time
import asyncio

from app.config import settings
from app.logging.logger import get_logger
from app.rag.retriever import RetrieverService
from app.llm.provider import LLMProvider, LLMResponse
from app.llm.prompt_templates import PromptManager
from app.services.cache_service import CacheService

logger = get_logger(__name__)


class RAGService:
    """
    Service for Retrieval-Augmented Generation.
    Combines document retrieval with LLM generation for accurate answers.
    """
    
    def __init__(
        self,
        retriever: RetrieverService,
        llm_provider: LLMProvider,
        cache_service: Optional[CacheService] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Initialize RAG service.
        
        Args:
            retriever: Document retrieval service
            llm_provider: LLM provider for generation
            cache_service: Optional cache service
            prompt_manager: Optional prompt manager
        """
        self.retriever = retriever
        self.llm = llm_provider
        self.cache = cache_service
        self.prompt_manager = prompt_manager or PromptManager()
        
        # Configuration
        self.default_top_k = settings.TOP_K_RETRIEVAL
        self.min_relevance_threshold = settings.SIMILARITY_THRESHOLD
        self.max_context_tokens = 3000
        
        # Metrics
        self._query_count = 0
        self._total_time = 0.0
        self._cache_hits = 0
        
        logger.info("RAGService initialized")
    
    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
        return_sources: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a RAG query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            filters: Metadata filters for retrieval
            temperature: LLM temperature
            use_cache: Whether to use cache
            return_sources: Whether to include source documents
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        self._query_count += 1
        
        k = top_k or self.default_top_k
        
        # Check cache
        cache_key = f"rag:{query}:{k}:{str(filters)}"
        if use_cache and self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                self._cache_hits += 1
                cached["from_cache"] = True
                return cached
        
        try:
            # Stage 1: Retrieve relevant documents
            retrieval_start = time.time()
            documents = await self.retriever.retrieve(
                query=query,
                top_k=k,
                filters=filters
            )
            retrieval_time = time.time() - retrieval_start
            
            # Filter by relevance threshold
            relevant_docs = [
                doc for doc in documents 
                if doc.score >= self.min_relevance_threshold
            ]
            
            if not relevant_docs:
                # Fallback: use top documents even if below threshold
                relevant_docs = documents[:k]
                logger.warning(f"No documents above threshold, using top {len(relevant_docs)}")
            
            # Stage 2: Build context
            context = self._build_context(relevant_docs)
            
            # Stage 3: Generate answer
            generation_start = time.time()
            answer = await self._generate_answer(
                query=query,
                context=context,
                temperature=temperature
            )
            generation_time = time.time() - generation_start
            
            # Stage 4: Post-process and validate
            validated_answer = await self._validate_answer(
                query=query,
                answer=answer,
                context=context
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                relevant_docs, validated_answer, query
            )
            
            # Build response
            total_time = time.time() - start_time
            
            response = {
                "query": query,
                "answer": validated_answer,
                "sources": [
                    {
                        "id": doc.id,
                        "text": doc.text[:500],  # Truncated for response
                        "score": doc.score,
                        "metadata": doc.meta
                    }
                    for doc in relevant_docs
                ] if return_sources else [],
                "retrieval_count": len(relevant_docs),
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "processing_time": total_time,
                "confidence": confidence,
                "from_cache": False
            }
            
            # Cache response
            if use_cache and self.cache:
                await self.cache.set(
                    cache_key,
                    response,
                    ttl=300  # Cache for 5 minutes
                )
            
            # Update metrics
            self._total_time += total_time
            
            logger.info(
                f"RAG query completed: {len(relevant_docs)} docs, "
                f"confidence={confidence:.2f}, time={total_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            
            # Fallback: generate without context
            try:
                fallback_answer = await self._generate_answer(
                    query=query,
                    context="",
                    temperature=temperature
                )
                
                return {
                    "query": query,
                    "answer": fallback_answer,
                    "sources": [],
                    "retrieval_count": 0,
                    "processing_time": time.time() - start_time,
                    "confidence": 0.3,
                    "error": str(e),
                    "fallback": True
                }
                
            except Exception as fallback_error:
                logger.error(f"Fallback generation failed: {fallback_error}")
                return {
                    "query": query,
                    "answer": "I apologize, but I encountered an error processing your query. Please try again.",
                    "sources": [],
                    "retrieval_count": 0,
                    "processing_time": time.time() - start_time,
                    "confidence": 0.0,
                    "error": str(e)
                }
    
    def _build_context(self, documents: List[Any]) -> str:
        """
        Build formatted context from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        token_estimate = 0
        
        for i, doc in enumerate(documents, 1):
            # Build document chunk
            chunk = f"[Document {i}]\n"
            
            if doc.meta.get("title"):
                chunk += f"Title: {doc.meta['title']}\n"
            
            chunk += f"{doc.text}\n"
            
            if doc.meta.get("source"):
                chunk += f"Source: {doc.meta['source']}\n"
            
            chunk += f"Relevance Score: {doc.score:.3f}\n\n"
            
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            estimated_tokens = len(chunk) // 4
            
            if token_estimate + estimated_tokens > self.max_context_tokens:
                break
            
            context_parts.append(chunk)
            token_estimate += estimated_tokens
        
        return "\n".join(context_parts)
    
    async def _generate_answer(
        self,
        query: str,
        context: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate answer using LLM with context.
        
        Args:
            query: User query
            context: Retrieved context
            temperature: LLM temperature
            
        Returns:
            Generated answer
        """
        # Get RAG prompt messages
        messages = self.prompt_manager.get_rag_prompt(
            query=query,
            context=context if context else "No relevant documents found."
        )
        
        # Generate response
        response = await self.llm.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )
        
        return response.content
    
    async def _validate_answer(
        self,
        query: str,
        answer: str,
        context: str
    ) -> str:
        """
        Validate and improve generated answer.
        
        Args:
            query: User query
            answer: Generated answer
            context: Source context
            
        Returns:
            Validated/improved answer
        """
        # Basic validation
        if not answer or len(answer.strip()) < 10:
            return "I couldn't generate a meaningful answer. Please try rephrasing your question."
        
        # Check for hallucination indicators
        hallucination_phrases = [
            "I don't know",
            "I'm not sure",
            "I cannot find"
        ]
        
        # If answer indicates no knowledge, but we have context, regenerate
        if any(phrase in answer.lower() for phrase in hallucination_phrases) and context:
            logger.warning("Answer indicates uncertainty despite having context, regenerating...")
            
            validation_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. The following context was retrieved "
                        "for the user's query. Please provide a complete answer based on "
                        "the context. If the context is insufficient, explain what additional "
                        "information would be needed.\n\n"
                        f"Context:\n{context}"
                    )
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nPlease provide a comprehensive answer based on the context above."
                }
            ]
            
            response = await self.llm.generate(messages=validation_prompt)
            return response.content
        
        return answer
    
    def _calculate_confidence(
        self,
        documents: List[Any],
        answer: str,
        query: str
    ) -> float:
        """
        Calculate confidence score for the answer.
        
        Args:
            documents: Retrieved documents
            answer: Generated answer
            query: User query
            
        Returns:
            Confidence score (0-1)
        """
        if not documents:
            return 0.1
        
        # Factors affecting confidence
        scores = []
        
        # 1. Average retrieval score
        avg_score = sum(doc.score for doc in documents) / len(documents)
        scores.append(avg_score)
        
        # 2. Number of documents found
        doc_count_score = min(len(documents) / 5, 1.0)  # Max at 5+ docs
        scores.append(doc_count_score)
        
        # 3. Answer length (too short might indicate uncertainty)
        answer_length = len(answer.split())
        if answer_length < 20:
            scores.append(0.3)
        elif answer_length < 50:
            scores.append(0.6)
        else:
            scores.append(0.9)
        
        # 4. Top document score
        top_score = documents[0].score if documents else 0
        scores.append(top_score)
        
        # Calculate weighted confidence
        weights = [0.4, 0.2, 0.2, 0.2]  # More weight on retrieval scores
        confidence = sum(s * w for s, w in zip(scores, weights))
        
        return min(max(confidence, 0.0), 1.0)
    
    async def batch_query(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple RAG queries concurrently.
        
        Args:
            queries: List of query strings
            top_k: Number of documents per query
            **kwargs: Additional parameters
            
        Returns:
            List of query results
        """
        tasks = [
            self.query(query, top_k=top_k, **kwargs)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch query {i} failed: {result}")
                processed_results.append({
                    "query": queries[i],
                    "answer": "Error processing query",
                    "error": str(result),
                    "sources": [],
                    "retrieval_count": 0,
                    "confidence": 0.0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def retrieve_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents without generating answer.
        
        Args:
            query: Search query
            top_k: Number of documents
            filters: Metadata filters
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved documents
        """
        documents = await self.retriever.retrieve(
            query=query,
            top_k=top_k or self.default_top_k,
            filters=filters
        )
        
        return [
            {
                "id": doc.id,
                "text": doc.text,
                "score": doc.score,
                "metadata": doc.meta,
                "rank": doc.rank
            }
            for doc in documents
        ]
    
    async def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Get formatted context for a query.
        
        Args:
            query: Search query
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            Formatted context string
        """
        return await self.retriever.retrieve_context(
            query=query,
            max_tokens=max_tokens
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics."""
        avg_time = self._total_time / self._query_count if self._query_count > 0 else 0
        cache_hit_rate = self._cache_hits / self._query_count if self._query_count > 0 else 0
        
        return {
            "total_queries": self._query_count,
            "avg_processing_time": avg_time,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "default_top_k": self.default_top_k,
            "min_relevance_threshold": self.min_relevance_threshold
        }
    
    def clear_cache(self):
        """Clear service cache."""
        if self.cache:
            # Clear RAG-specific cache entries
            self._cache_hits = 0
            logger.info("RAG service cache cleared")

    async def stream_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None
    ) -> AsyncIterator[str]:
        """Stream RAG response tokens in real-time."""
        documents = await self.retriever.retrieve(
            query=query,
            top_k=top_k or self.default_top_k,
            filters=filters
        )
        context = self._build_context(documents)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. Answer the user question accurately using only "
                    f"the provided context. Include source citations.\n\nContext:\n{context}"
                )
            },
            {"role": "user", "content": query}
        ]
        async for chunk in self.llm.stream(messages=messages, temperature=temperature):
            if chunk.content:
                yield chunk.content