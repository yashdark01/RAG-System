"""
Tool implementations for the RAG system.
"""

from app.workflow.tools.rag_tools import (
    retrieve_documents,
    search_vector_store,
    get_document_context,
    rerank_results,
    get_document_metadata
)
from app.workflow.tools.search_tools import (
    web_search,
    search_documents,
    hybrid_search,
    filter_by_metadata,
    deduplicate_results
)
from app.workflow.tools.verification_tools import (
    verify_claim,
    validate_source,
    calculate_confidence,
    check_consistency,
    cross_reference
)

__all__ = [
    # RAG tools
    "retrieve_documents",
    "search_vector_store",
    "get_document_context",
    "rerank_results",
    "get_document_metadata",
    # Search tools
    "web_search",
    "search_documents",
    "hybrid_search",
    "filter_by_metadata",
    "deduplicate_results",
    # Verification tools
    "verify_claim",
    "validate_source",
    "calculate_confidence",
    "check_consistency",
    "cross_reference"
]
