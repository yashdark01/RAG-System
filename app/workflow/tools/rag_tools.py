"""
LangChain RAG Tools.
Provides LangChain @tool definitions for vector and hybrid knowledge base retrieval.
"""

from typing import Dict, Any, Optional, List
import asyncio
from langchain_core.tools import tool, BaseTool
from app.logging.logger import get_logger

logger = get_logger(__name__)


def create_knowledge_base_tool(retriever_service: Any, top_k: int = 5) -> BaseTool:
    """Create a LangChain tool bound to the RAG retriever service."""

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the internal knowledge base for relevant facts, policies, and documents."""
        try:
            import asyncio
            # If inside an event loop, create task or run
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Return formatted sync representation or schedule
                    future = asyncio.run_coroutine_threadsafe(
                        retriever_service.retrieve(query=query, top_k=top_k),
                        loop
                    )
                    results = future.result(timeout=10)
                else:
                    results = loop.run_until_complete(
                        retriever_service.retrieve(query=query, top_k=top_k)
                    )
            except RuntimeError:
                results = asyncio.run(retriever_service.retrieve(query=query, top_k=top_k))

            if not results:
                return "No matching documents found in knowledge base."

            formatted = []
            for r in results:
                src = r.metadata.get("source", "doc") if hasattr(r, "metadata") else "doc"
                txt = r.text if hasattr(r, "text") else str(r)
                formatted.append(f"[Source: {src}]\n{txt}")
            return "\n\n---\n\n".join(formatted)

        except Exception as e:
            logger.error(f"Error in search_knowledge_base tool: {e}")
            return f"Error querying knowledge base: {e}"

    return search_knowledge_base


async def retrieve_documents(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    rag_service: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Retrieve documents using RAG service (backward compatible)."""
    if not rag_service:
        return []
    try:
        results = await rag_service.retrieve_only(query=query, top_k=top_k, filters=filters or {})
        return results
    except Exception as e:
        logger.error(f"Failed to retrieve documents: {e}")
        return []


async def search_vector_store(
    query_embedding: List[float],
    top_k: int = 5,
    vector_store_service: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Search vector store by embedding."""
    if not vector_store_service:
        return []
    try:
        return await vector_store_service.search_by_vector(query_embedding, top_k=top_k)
    except Exception as e:
        logger.error(f"Vector store search failed: {e}")
        return []


async def get_document_context(
    document_id: str,
    vector_store_service: Optional[Any] = None
) -> Dict[str, Any]:
    """Get context and chunks for a specific document ID."""
    if not vector_store_service:
        return {"context": "", "chunks": []}
    try:
        doc = vector_store_service.get_document(document_id)
        if asyncio.iscoroutine(doc):
            doc = await doc
        if not doc:
            return {"context": "", "chunks": []}
        chunks = doc.get("chunks", [])
        return {
            "document_id": document_id,
            "context": "\n\n".join(c.get("text", "") for c in chunks),
            "chunks": chunks
        }
    except Exception as e:
        logger.error(f"Failed to get document context: {e}")
        return {"context": "", "chunks": []}


async def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    reranker_service: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Rerank retrieved results using a reranker service."""
    if not reranker_service or not results:
        return results
    try:
        doc_texts = [r.get("text", "") for r in results]
        scores = reranker_service.rerank(query, doc_texts)
        if asyncio.iscoroutine(scores):
            scores = await scores
        for r, score in zip(results, scores):
            r["reranked_score"] = score
        results.sort(key=lambda x: x.get("reranked_score", 0), reverse=True)
        return results
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return results


async def get_document_metadata(
    document_id: str,
    vector_store_service: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """Retrieve metadata for a specific document ID."""
    if not vector_store_service:
        return None
    try:
        doc = vector_store_service.get_document(document_id)
        if asyncio.iscoroutine(doc):
            doc = await doc
        if not doc:
            return None
        return doc.get("metadata", {})
    except Exception as e:
        logger.error(f"Failed to get document metadata: {e}")
        return None
