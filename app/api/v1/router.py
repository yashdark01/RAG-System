"""
Main API v1 router.
Aggregates all endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    health,
    rag,
    documents,
    chat,
    agents
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    health.router,
    tags=["Health"]
)

api_router.include_router(
    rag.router,
    prefix="/rag",
    tags=["RAG"]
)

api_router.include_router(
    documents.router,
    prefix="/documents",
    tags=["Documents"]
)

api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"]
)

api_router.include_router(
    agents.router,
    prefix="/agents",
    tags=["Agents"]
)