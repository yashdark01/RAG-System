"""
Health check endpoints.
"""

from typing import Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.dependencies import dependencies
from app.config import settings

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV
    }


@router.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check - verifies all services are ready."""
    checks = {}
    healthy = True
    
    # Check LLM provider
    try:
        llm = dependencies.get_llm_provider()
        llm_health = await llm.check_health()
        checks["llm"] = {
            "status": "healthy" if all(llm_health.values()) else "degraded",
            "providers": llm_health
        }
        healthy = healthy and all(llm_health.values())
    except Exception as e:
        checks["llm"] = {"status": "unhealthy", "error": str(e)}
        healthy = False
    
    # Check vector store
    try:
        vector_store = dependencies.get_vector_store()
        stats = vector_store.get_stats()
        checks["vector_store"] = {
            "status": "healthy",
            "documents": stats["active_documents"]
        }
    except Exception as e:
        checks["vector_store"] = {"status": "unhealthy", "error": str(e)}
        healthy = False
    
    # Check Redis
    try:
        redis = await dependencies.redis_client
        await redis.ping()
        checks["redis"] = {"status": "healthy"}
    except Exception as e:
        checks["redis"] = {"status": "unhealthy", "error": str(e)}
        healthy = False
    
    return {
        "status": "ready" if healthy else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks
    }


@router.get("/health/live")
async def liveness_check() -> Dict[str, str]:
    """Liveness check - minimal check to verify service is running."""
    return {"status": "alive"}


@router.get("/health/stats")
async def system_stats() -> Dict[str, Any]:
    """Get system statistics."""
    stats = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION
    }
    
    # Embedding service stats
    try:
        embedding_service = dependencies.get_embedding_service()
        stats["embeddings"] = embedding_service.get_cache_stats()
    except Exception as e:
        stats["embeddings"] = {"error": str(e)}
    
    # Vector store stats
    try:
        vector_store = dependencies.get_vector_store()
        stats["vector_store"] = vector_store.get_stats()
    except Exception as e:
        stats["vector_store"] = {"error": str(e)}
    
    # LLM stats
    try:
        llm = dependencies.get_llm_provider()
        stats["llm"] = llm.get_usage_stats()
    except Exception as e:
        stats["llm"] = {"error": str(e)}
    
    # Retriever stats
    try:
        retriever = await dependencies.get_retriever()
        stats["retriever"] = retriever.get_stats()
    except Exception as e:
        stats["retriever"] = {"error": str(e)}
    
    return stats