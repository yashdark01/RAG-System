"""
RAG (Retrieval-Augmented Generation) endpoints.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import StreamingResponse

from app.dependencies import dependencies
from app.models.schemas.request import (
    RAGQueryRequest,
    DocumentUploadRequest,
    BatchRAGQueryRequest
)
from app.models.schemas.response import (
    RAGResponse,
    RetrievalResponse,
    ErrorResponse
)
from app.logging.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=RAGResponse,
    summary="Query the RAG system",
    description="Submit a query and get an AI-generated answer based on retrieved documents"
)
async def rag_query(
    request: RAGQueryRequest,
    rag_service = Depends(dependencies.get_rag_service)
) -> RAGResponse:
    """
    Execute a RAG query.
    
    - **query**: The question or prompt to process
    - **top_k**: Number of documents to retrieve (default: 5)
    - **filters**: Optional metadata filters for document retrieval
    - **temperature**: LLM temperature for generation
    - **stream**: Whether to stream the response
    """
    try:
        # Execute RAG query
        result = await rag_service.query(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            temperature=request.temperature
        )
        
        return RAGResponse(
            query=request.query,
            answer=result["answer"],
            sources=result.get("sources", []),
            retrieval_count=result.get("retrieval_count", 0),
            confidence=result.get("confidence"),
            processing_time=result.get("processing_time"),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/query/stream",
    summary="Stream RAG query response",
    description="Submit a query and stream real-time tokens via Server-Sent Events"
)
async def stream_rag_query(
    request: RAGQueryRequest,
    rag_service = Depends(dependencies.get_rag_service)
):
    """Stream RAG response tokens."""
    async def token_generator():
        try:
            async for token in rag_service.stream_query(
                query=request.query,
                top_k=request.top_k,
                filters=request.filters,
                temperature=request.temperature
            ):
                yield f"data: {token}\n\n"
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: [Error: {str(e)}]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    summary="Retrieve relevant documents",
    description="Search for relevant documents without generating an answer"
)
async def retrieve_documents(
    request: RAGQueryRequest,
    retriever = Depends(dependencies.get_retriever)
) -> RetrievalResponse:
    """
    Retrieve relevant documents for a query without LLM generation.
    
    - **query**: Search query
    - **top_k**: Number of results
    - **filters**: Metadata filters
    """
    try:
        # Retrieve documents
        results = await retriever.retrieve(
            query=request.query,
            top_k=request.top_k or 5,
            filters=request.filters
        )
        
        return RetrievalResponse(
            query=request.query,
            results=[
                {
                    "id": r.id,
                    "text": r.text,
                    "score": r.score,
                    "metadata": r.meta,
                    "rank": r.rank
                }
                for r in results
            ],
            total_found=len(results),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/batch-query",
    response_model=List[RAGResponse],
    summary="Batch RAG queries",
    description="Process multiple RAG queries in batch"
)
async def batch_rag_query(
    request: BatchRAGQueryRequest,
    rag_service = Depends(dependencies.get_rag_service)
) -> List[RAGResponse]:
    """
    Execute multiple RAG queries in batch.
    
    - **queries**: List of query configurations
    """
    try:
        responses = []
        
        for query_req in request.queries:
            result = await rag_service.query(
                query=query_req.query,
                top_k=query_req.top_k,
                filters=query_req.filters,
                temperature=query_req.temperature
            )
            
            responses.append(RAGResponse(
                query=query_req.query,
                answer=result["answer"],
                sources=result.get("sources", []),
                retrieval_count=result.get("retrieval_count", 0),
                confidence=result.get("confidence"),
                processing_time=result.get("processing_time"),
                timestamp=datetime.utcnow().isoformat()
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/context",
    summary="Get RAG context",
    description="Retrieve and format context for a query without generating answer"
)
async def get_context(
    query: str = Body(..., embed=True),
    max_tokens: int = Body(2000, embed=True),
    retriever = Depends(dependencies.get_retriever)
) -> Dict[str, Any]:
    """
    Get formatted context from retrieved documents.
    
    - **query**: Search query
    - **max_tokens**: Maximum tokens for context
    """
    try:
        context = await retriever.retrieve_context(
            query=query,
            max_tokens=max_tokens
        )
        
        return {
            "query": query,
            "context": context,
            "token_estimate": len(context) // 4,  # Rough token estimate
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/stats",
    summary="Get RAG statistics",
    description="Get performance statistics for the RAG system"
)
async def get_rag_stats(
    retriever = Depends(dependencies.get_retriever),
    embedding_service = Depends(dependencies.get_embedding_service),
    vector_store = Depends(dependencies.get_vector_store)
) -> Dict[str, Any]:
    """Get comprehensive RAG system statistics."""
    try:
        return {
            "retriever": retriever.get_stats(),
            "embeddings": embedding_service.get_cache_stats(),
            "vector_store": vector_store.get_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))