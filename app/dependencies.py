"""
Dependency injection container for the application.
Provides centralized access to services and components.
"""

from functools import lru_cache
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.config import settings
from app.database.session import get_db_session, init_db, close_db
from app.cache.redis_cache import RedisCache, get_redis
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStoreService
from app.rag.retriever import RetrieverService
from app.rag.document_processor import DocumentProcessor
from app.rag.reranker import RerankerService
from app.llm.provider import LLMProvider
from app.services.document_service import DocumentService
from app.services.chat_service import ChatService
from app.services.agent_service import AgentService
from app.services.rag_service import RAGService
from app.services.cache_service import CacheService


class Dependencies:
    """Centralized dependency container."""
    
    def __init__(self):
        self._vector_store: Optional[VectorStoreService] = None
        self._embedding_service: Optional[EmbeddingService] = None
        self._llm_provider: Optional[LLMProvider] = None
        self._reranker: Optional[RerankerService] = None
    
    async def get_db(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session."""
        async for session in get_db_session():
            yield session
    
    async def get_redis_client(self) -> Redis:
        """Get Redis client."""
        return await get_redis()
    
    @lru_cache()
    def get_embedding_service(self) -> EmbeddingService:
        """Get or create embedding service singleton."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(
                model_name=settings.EMBEDDING_MODEL,
                dimension=settings.EMBEDDING_DIMENSION
            )
        return self._embedding_service
    
    @lru_cache()
    def get_vector_store(self) -> VectorStoreService:
        """Get or create vector store service singleton."""
        if self._vector_store is None:
            embedding_service = self.get_embedding_service()
            self._vector_store = VectorStoreService(
                embedding_service=embedding_service,
                store_path=settings.VECTOR_STORE_PATH
            )
        return self._vector_store
    
    @lru_cache()
    def get_llm_provider(self) -> LLMProvider:
        """Get or create LLM provider singleton."""
        if self._llm_provider is None:
            self._llm_provider = LLMProvider(
                provider=settings.LLM_PROVIDER,
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS
            )
        return self._llm_provider
    
    @lru_cache()
    def get_reranker(self) -> RerankerService:
        """Get or create reranker service singleton."""
        if self._reranker is None:
            self._reranker = RerankerService()
        return self._reranker
    
    def get_document_processor(self) -> DocumentProcessor:
        """Get document processor instance."""
        return DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    
    async def get_retriever(self) -> RetrieverService:
        """Get retriever service."""
        vector_store = self.get_vector_store()
        embedding_service = self.get_embedding_service()
        reranker = self.get_reranker()
        
        return RetrieverService(
            vector_store=vector_store,
            embedding_service=embedding_service,
            reranker=reranker,
            top_k=settings.TOP_K_RETRIEVAL,
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
            reranking_enabled=settings.RERANKING_ENABLED
        )
    
    async def get_rag_service(self) -> RAGService:
        """Get RAG service."""
        retriever = await self.get_retriever()
        llm_provider = self.get_llm_provider()
        
        return RAGService(
            retriever=retriever,
            llm_provider=llm_provider
        )
    
    async def get_document_service(self) -> DocumentService:
        """Get document service."""
        processor = self.get_document_processor()
        vector_store = self.get_vector_store()
        
        return DocumentService(
            processor=processor,
            vector_store=vector_store
        )
    
    async def get_chat_service(self) -> ChatService:
        """Get chat service."""
        rag_service = await self.get_rag_service()
        llm_provider = self.get_llm_provider()
        
        return ChatService(
            rag_service=rag_service,
            llm_provider=llm_provider
        )
    
    async def get_agent_service(self) -> AgentService:
        """Get agent service."""
        llm_provider = self.get_llm_provider()
        
        return AgentService(
            llm_provider=llm_provider,
            max_iterations=settings.MAX_AGENT_ITERATIONS,
            timeout=settings.AGENT_TIMEOUT_SECONDS
        )
    
    async def get_cache_service(self) -> CacheService:
        """Get cache service."""
        redis = await self.get_redis_client()
        return CacheService(
            redis_client=redis,
            default_ttl=settings.CACHE_TTL_SECONDS
        )


# Global dependency instance
dependencies = Dependencies()