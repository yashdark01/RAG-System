"""
LangChain Search Tools with Tavily Search Integration.
Provides authoritative web search and document search tools for agents.
"""

from typing import Dict, Any, Optional, List
from langchain_core.tools import tool, BaseTool
from app.config import settings
from app.logging.logger import get_logger

logger = get_logger(__name__)


def get_tavily_tool(max_results: int = 5) -> BaseTool:
    """Get the official Tavily search tool if API key is present, else mock fallback."""
    if settings.TAVILY_API_KEY:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            return TavilySearchResults(
                max_results=max_results,
                tavily_api_key=settings.TAVILY_API_KEY
            )
        except Exception as e:
            logger.warning(f"Failed to initialize TavilySearchResults: {e}")

    # Fallback tool if Tavily is not configured
    @tool
    def fallback_web_search(query: str) -> str:
        """Search the web for up-to-date information."""
        return f"[Web Search Result for '{query}']: Please configure TAVILY_API_KEY in .env for live web search."

    return fallback_web_search


async def web_search(
    query: str,
    num_results: int = 5,
    search_engine: str = "tavily",
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform web search using Tavily or return formatted results.
    """
    api_key = api_key or settings.TAVILY_API_KEY
    if api_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, max_results=num_results)
            results = []
            for item in response.get("results", []):
                results.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "source": "tavily",
                    "score": item.get("score", 0.9)
                })
            return results
        except Exception as e:
            logger.warning(f"Tavily search API failed: {e}. Falling back.")

    return [
        {
            "url": "https://tavily.com",
            "title": "Tavily Search Required",
            "snippet": f"Web search for '{query}' requires TAVILY_API_KEY configured in your environment.",
            "source": "system",
            "score": 0.5
        }
    ]


async def search_documents(
    query: str,
    document_service: Optional[Any] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search indexed documents."""
    if not document_service:
        return []
    try:
        return await document_service.search_documents(
            query=query,
            filters=filters,
            limit=limit
        )
    except Exception as e:
        logger.error(f"Document search failed: {e}")
        return []


async def hybrid_search(
    query: str,
    retriever_service: Optional[Any] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Execute hybrid retrieval combining semantic and keyword matching."""
    if not retriever_service:
        return []
    try:
        return await retriever_service.retrieve(query=query, top_k=top_k)
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return []


async def filter_by_metadata(
    documents: List[Dict[str, Any]],
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Filter documents by metadata attributes."""
    if not filters:
        return documents
    filtered = []
    for doc in documents:
        meta = doc.get("metadata", {})
        matches = all(meta.get(k) == v for k, v in filters.items())
        if matches:
            filtered.append(doc)
    return filtered


async def deduplicate_results(
    documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Deduplicate document results by ID."""
    seen_ids = set()
    deduped = []
    for doc in documents:
        doc_id = doc.get("id")
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            deduped.append(doc)
    return deduped
