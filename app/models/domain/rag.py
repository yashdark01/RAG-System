"""
Domain models for RAG-related entities.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class RetrievalStrategy(str, Enum):
    """Strategies for document retrieval."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class RetrievalResult(BaseModel):
    """Represents a single retrieval result."""
    id: str = Field(..., description="Unique result identifier")
    text: str = Field(..., description="Retrieved text content")
    score: float = Field(..., description="Similarity score", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")
    rank: int = Field(..., description="Rank in results", ge=1)
    retrieval_method: str = Field(..., description="Method used for retrieval")
    chunk_info: Optional[Dict[str, Any]] = Field(None, description="Chunk information")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "result_123",
                "text": "This is the retrieved text...",
                "score": 0.95,
                "rank": 1,
                "retrieval_method": "semantic"
            }
        }
    )


class RetrievalConfig(BaseModel):
    """Configuration for retrieval operations."""
    top_k: int = Field(default=5, ge=1, le=100, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
    strategy: RetrievalStrategy = Field(default=RetrievalStrategy.HYBRID, description="Retrieval strategy")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    reranking_enabled: bool = Field(default=True, description="Whether to use reranking")
    use_cache: bool = Field(default=True, description="Whether to use cache")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "top_k": 5,
                "similarity_threshold": 0.7,
                "strategy": "hybrid",
                "reranking_enabled": True
            }
        }
    )


class RAGQuery(BaseModel):
    """Represents a RAG query and its response."""
    id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="User query")
    context: str = Field(default="", description="Retrieved context")
    answer: str = Field(..., description="Generated answer")
    sources: List[RetrievalResult] = Field(default_factory=list, description="Source documents")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    retrieval_count: int = Field(default=0, description="Number of documents retrieved")
    retrieval_time: Optional[float] = Field(None, description="Retrieval time in seconds")
    generation_time: Optional[float] = Field(None, description="Generation time in seconds")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    from_cache: bool = Field(default=False, description="Whether result was from cache")
    error: Optional[str] = Field(None, description="Error message if query failed")
    fallback: bool = Field(default=False, description="Whether fallback was used")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "query_123",
                "query": "What is machine learning?",
                "answer": "Machine learning is a subset of AI...",
                "confidence": 0.85,
                "retrieval_count": 5,
                "processing_time": 2.5
            }
        }
    )


class RetrievalMetrics(BaseModel):
    """Metrics for retrieval operations."""
    total_queries: int = Field(default=0, description="Total number of queries")
    avg_retrieval_time: float = Field(default=0.0, description="Average retrieval time in seconds")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, description="Number of cache misses")
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Cache hit rate")
    avg_similarity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average similarity score")
    total_documents_retrieved: int = Field(default=0, description="Total documents retrieved")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_queries": 100,
                "avg_retrieval_time": 0.5,
                "cache_hits": 60,
                "cache_misses": 40,
                "cache_hit_rate": 0.6
            }
        }
    )


class RAGConfig(BaseModel):
    """Global RAG configuration."""
    max_context_tokens: int = Field(default=3000, ge=100, le=10000, description="Maximum context tokens")
    min_relevance_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum relevance threshold")
    default_top_k: int = Field(default=5, ge=1, le=100, description="Default top_k for retrieval")
    enable_validation: bool = Field(default=True, description="Enable answer validation")
    enable_reranking: bool = Field(default=True, description="Enable result reranking")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold for answers")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "max_context_tokens": 3000,
                "min_relevance_threshold": 0.5,
                "default_top_k": 5,
                "enable_validation": True,
                "enable_reranking": True
            }
        }
    )
