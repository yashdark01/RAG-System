"""
Unit tests for tool implementations.
"""

import pytest
from unittest.mock import Mock, AsyncMock
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


class TestRAGTools:
    """Test RAG tool functions."""
    
    @pytest.mark.asyncio
    async def test_retrieve_documents_without_service(self):
        """Test retrieve_documents without RAG service."""
        result = await retrieve_documents("test query")
        assert result == []
    
    @pytest.mark.asyncio
    async def test_retrieve_documents_with_service(self):
        """Test retrieve_documents with RAG service."""
        mock_rag_service = Mock()
        mock_rag_service.retrieve_only = AsyncMock(return_value=[
            {"id": "doc_1", "text": "Document 1", "score": 0.9}
        ])
        
        result = await retrieve_documents("test query", rag_service=mock_rag_service)
        
        assert len(result) == 1
        assert result[0]["id"] == "doc_1"
    
    @pytest.mark.asyncio
    async def test_search_vector_store_without_service(self):
        """Test search_vector_store without vector store service."""
        result = await search_vector_store([0.1, 0.2, 0.3])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_search_vector_store_with_service(self):
        """Test search_vector_store with vector store service."""
        mock_vector_store = Mock()
        mock_vector_store.search = Mock(return_value=[
            {"id": "doc_1", "text": "Document 1", "score": 0.95}
        ])
        
        result = await search_vector_store([0.1, 0.2, 0.3], vector_store_service=mock_vector_store)
        
        assert len(result) == 1
    
    @pytest.mark.asyncio
    async def test_get_document_context_without_service(self):
        """Test get_document_context without vector store service."""
        result = await get_document_context("doc_123")
        assert result == {"context": "", "chunks": []}
    
    @pytest.mark.asyncio
    async def test_get_document_context_with_service(self):
        """Test get_document_context with vector store service."""
        mock_vector_store = Mock()
        mock_vector_store.get_document = Mock(return_value={
            "id": "doc_123",
            "chunks": [{"text": "Chunk 1"}, {"text": "Chunk 2"}]
        })
        
        result = await get_document_context("doc_123", vector_store_service=mock_vector_store)
        
        assert result["document_id"] == "doc_123"
        assert len(result["chunks"]) == 2
    
    @pytest.mark.asyncio
    async def test_rerank_results_without_service(self):
        """Test rerank_results without reranker service."""
        results = [{"id": "doc_1", "text": "Document 1", "score": 0.9}]
        result = await rerank_results("query", results)
        assert result == results
    
    @pytest.mark.asyncio
    async def test_rerank_results_with_service(self):
        """Test rerank_results with reranker service."""
        mock_reranker = Mock()
        mock_reranker.rerank = Mock(return_value=[0.95, 0.85])
        
        results = [
            {"id": "doc_1", "text": "Document 1"},
            {"id": "doc_2", "text": "Document 2"}
        ]
        
        result = await rerank_results("query", results, mock_reranker)
        
        assert len(result) == 2
        assert "reranked_score" in result[0]
    
    @pytest.mark.asyncio
    async def test_get_document_metadata_without_service(self):
        """Test get_document_metadata without vector store service."""
        result = await get_document_metadata("doc_123")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_document_metadata_with_service(self):
        """Test get_document_metadata with vector store service."""
        mock_vector_store = Mock()
        mock_vector_store.get_document = Mock(return_value={
            "id": "doc_123",
            "metadata": {"title": "Test Document", "author": "Test Author"}
        })
        
        result = await get_document_metadata("doc_123", vector_store_service=mock_vector_store)
        
        assert result["title"] == "Test Document"
        assert result["author"] == "Test Author"


class TestSearchTools:
    """Test search tool functions."""
    
    @pytest.mark.asyncio
    async def test_web_search(self):
        """Test web_search function."""
        result = await web_search("test query", num_results=3)
        
        assert isinstance(result, list)
        assert len(result) <= 3
        if result:
            assert "url" in result[0]
            assert "title" in result[0]
    
    @pytest.mark.asyncio
    async def test_search_documents_without_service(self):
        """Test search_documents without document service."""
        result = await search_documents("test query")
        assert result == []
    
    @pytest.mark.asyncio
    async def test_search_documents_with_service(self):
        """Test search_documents with document service."""
        mock_document_service = Mock()
        mock_document_service.search_documents = AsyncMock(return_value=[
            {"id": "doc_1", "text": "Document 1"}
        ])
        
        result = await search_documents("test query", document_service=mock_document_service)
        
        assert len(result) == 1
    
    @pytest.mark.asyncio
    async def test_hybrid_search_without_service(self):
        """Test hybrid_search without retriever service."""
        result = await hybrid_search("test query")
        assert result == []
    
    @pytest.mark.asyncio
    async def test_hybrid_search_with_service(self):
        """Test hybrid_search with retriever service."""
        mock_retriever = Mock()
        mock_retriever.retrieve = AsyncMock(return_value=[
            {"id": "doc_1", "text": "Document 1", "score": 0.9}
        ])
        
        result = await hybrid_search("test query", retriever_service=mock_retriever)
        
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_filter_by_metadata(self):
        """Test filter_by_metadata function."""
        results = [
            {"id": "doc_1", "metadata": {"category": "tech"}},
            {"id": "doc_2", "metadata": {"category": "science"}},
            {"id": "doc_3", "metadata": {"category": "tech"}}
        ]
        
        filtered = await filter_by_metadata(results, {"category": "tech"})
        
        assert len(filtered) == 2
        assert filtered[0]["id"] == "doc_1"
        assert filtered[1]["id"] == "doc_3"
    
    @pytest.mark.asyncio
    async def test_filter_by_metadata_no_filters(self):
        """Test filter_by_metadata with no filters."""
        results = [{"id": "doc_1"}, {"id": "doc_2"}]
        filtered = await filter_by_metadata(results, {})
        assert filtered == results
    
    @pytest.mark.asyncio
    async def test_deduplicate_results(self):
        """Test deduplicate_results function."""
        results = [
            {"id": "doc_1", "text": "Document 1"},
            {"id": "doc_2", "text": "Document 2"},
            {"id": "doc_1", "text": "Document 1 duplicate"}
        ]
        
        deduplicated = await deduplicate_results(results)
        
        assert len(deduplicated) == 2
        assert deduplicated[0]["id"] == "doc_1"
    
    @pytest.mark.asyncio
    async def test_deduplicate_results_empty(self):
        """Test deduplicate_results with empty list."""
        result = await deduplicate_results([])
        assert result == []


class TestVerificationTools:
    """Test verification tool functions."""
    
    @pytest.mark.asyncio
    async def test_verify_claim_without_sources(self):
        """Test verify_claim without sources."""
        result = await verify_claim("Test claim", [])
        
        assert result["status"] == "unverified"
        assert result["confidence"] == 0.0
    
    @pytest.mark.asyncio
    async def test_verify_claim_with_sources(self):
        """Test verify_claim with sources."""
        sources = [
            {"text": "This supports the claim", "snippet": "Supporting text"}
        ]
        
        result = await verify_claim("Test claim", sources)
        
        assert result["claim"] == "Test claim"
        assert result["status"] in ["supported", "unverified"]
        assert 0.0 <= result["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_verify_claim_with_llm(self):
        """Test verify_claim with LLM provider."""
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(return_value=Mock(content="Claim is supported"))
        
        sources = [{"text": "Supporting text"}]
        result = await verify_claim("Test claim", sources, mock_llm)
        
        assert result["status"] == "supported"
    
    @pytest.mark.asyncio
    async def test_validate_source_without_content(self):
        """Test validate_source without content."""
        source = {"id": "doc_1"}
        result = await validate_source(source)
        
        assert result["credible"] is False
        assert "no content" in result["reason"].lower()
    
    @pytest.mark.asyncio
    async def test_validate_source_knowledge_base(self):
        """Test validate_source with knowledge base source."""
        source = {
            "id": "doc_1",
            "text": "Document content",
            "source": "knowledge_base"
        }
        
        result = await validate_source(source)
        
        assert result["credible"] is True
        assert result["credibility_score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_validate_source_web(self):
        """Test validate_source with web source."""
        source = {
            "id": "doc_1",
            "text": "Document content",
            "source": "web",
            "url": "https://example.edu/article"
        }
        
        result = await validate_source(source)
        
        assert result["credibility_score"] == 0.8
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_without_results(self):
        """Test calculate_confidence without verification results."""
        result = await calculate_confidence("claim", [], [])
        assert result == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_with_results(self):
        """Test calculate_confidence with verification results."""
        verification_results = [
            {"confidence": 0.8},
            {"confidence": 0.9}
        ]
        sources = [
            {"credible": True},
            {"credible": True}
        ]
        
        result = await calculate_confidence("claim", sources, verification_results)
        
        assert 0.0 <= result <= 1.0
    
    @pytest.mark.asyncio
    async def test_check_consistency_single_claim(self):
        """Test check_consistency with single claim."""
        claims = [{"claim": "Claim 1", "status": "supported"}]
        result = await check_consistency(claims)
        
        assert result["consistent"] is True
        # Check that the result has the expected structure
        assert "consistent" in result
    
    @pytest.mark.asyncio
    async def test_check_consistency_multiple_claims(self):
        """Test check_consistency with multiple claims."""
        claims = [
            {"claim": "Claim 1", "status": "supported"},
            {"claim": "Claim 2", "status": "supported"}
        ]
        result = await check_consistency(claims)
        
        assert result["consistent"] is True
        assert result["supported"] == 2
    
    @pytest.mark.asyncio
    async def test_check_consistency_with_contradiction(self):
        """Test check_consistency with contradictory claims."""
        claims = [
            {"claim": "Claim 1", "status": "supported"},
            {"claim": "Claim 2", "status": "contradicted"}
        ]
        result = await check_consistency(claims)
        
        assert result["consistent"] is False
        assert result["contradicted"] == 1
    
    @pytest.mark.asyncio
    async def test_cross_reference_without_sources(self):
        """Test cross_reference without additional sources."""
        result = await cross_reference("Test claim", [])
        
        assert result["cross_referenced"] is False
        assert result["reason"] == "No additional sources provided"
    
    @pytest.mark.asyncio
    async def test_cross_reference_with_sources(self):
        """Test cross_reference with additional sources."""
        additional_sources = [
            {"text": "This mentions the claim", "snippet": "Claim text"}
        ]
        
        result = await cross_reference("Test claim", additional_sources)
        
        assert result["cross_referenced"] is True
        assert result["total_sources"] == 1
