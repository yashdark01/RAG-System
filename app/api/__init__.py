"""
API module for the RAG system.
Provides REST API endpoints for all system functionality.
"""

from app.api.v1.router import api_router

__all__ = ["api_router"]