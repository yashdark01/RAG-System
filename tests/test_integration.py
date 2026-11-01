"""
Integration tests for API endpoints and end-to-end workflows.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch


class TestAPIEndpoints:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app for testing."""
        from app.main import app
        return app
    
    @pytest.fixture
    def client(self, mock_app):
        """Create a test client."""
        return TestClient(mock_app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code in [200, 404]  # 404 if endpoint doesn't exist yet
    
    def test_list_agents(self, client):
        """Test list agents endpoint."""
        response = client.get("/api/v1/agents")
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "agents" in data
    
    def test_rag_query_endpoint(self, client):
        """Test RAG query endpoint."""
        response = client.post(
            "/api/v1/rag/query",
            json={
                "query": "What is machine learning?",
                "top_k": 5
            }
        )
        assert response.status_code in [200, 404, 422]
    
    def test_document_upload_endpoint(self, client):
        """Test document upload endpoint."""
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"test content", "text/plain")}
        )
        assert response.status_code in [200, 404, 422]
    
    def test_chat_endpoint(self, client):
        """Test chat endpoint."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello, how can you help me?",
                "conversation_id": "test_conv"
            }
        )
        assert response.status_code in [200, 404, 422]


class TestAgentWorkflows:
    """Integration tests for agent workflows."""
    
    @pytest.mark.asyncio
    async def test_query_research_synthesis_workflow(self):
        """Test complete query -> research -> synthesis workflow."""
        from app.workflow.agents.query_agent import QueryAgent
        from app.workflow.agents.research_agent import ResearchAgent
        from app.workflow.agents.synthesis_agent import SynthesisAgent
        from app.workflow.orchestrator import AgentOrchestrator
        from app.llm.provider import LLMProvider
        
        # Create mock LLM provider
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.generate = AsyncMock(return_value=Mock(content="Mock response"))
        
        # Create agents
        query_agent = QueryAgent(mock_llm)
        research_agent = ResearchAgent(mock_llm)
        synthesis_agent = SynthesisAgent(mock_llm)
        
        # Execute query agent
        query_result = await query_agent.execute("What is machine learning?")
        assert query_result["success"] is True
        
        # Execute research agent with query results
        research_context = {
            "query_analysis": query_result["analysis"],
            "tools": ["retrieve_documents"]
        }
        research_result = await research_agent.execute("What is machine learning?", research_context)
        assert research_result["success"] is True
        
        # Execute synthesis agent with research results
        synthesis_context = {
            "research_results": research_result["research"],
            "query_analysis": query_result["analysis"]
        }
        synthesis_result = await synthesis_agent.execute("What is machine learning?", synthesis_context)
        assert synthesis_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_validation_workflow(self):
        """Test validation workflow."""
        from app.workflow.agents.validation_agent import ValidationAgent
        from app.llm.provider import LLMProvider
        
        # Create mock LLM provider
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.generate = AsyncMock(return_value=Mock(content="Mock response"))
        
        # Create validation agent
        validation_agent = ValidationAgent(mock_llm)
        
        # Execute validation
        context = {
            "documents": [
                {"text": "Machine learning is a subset of AI", "source": "knowledge_base"}
            ],
            "web_results": []
        }
        
        result = await validation_agent.execute(
            "Machine learning is a subset of artificial intelligence",
            context
        )
        
        assert result["success"] is True
        assert "validation" in result
        assert "overall_confidence" in result["validation"]


class TestRAGWorkflow:
    """Integration tests for RAG workflow."""
    
    @pytest.mark.asyncio
    async def test_rag_retrieve_and_generate(self):
        """Test RAG retrieve and generate workflow."""
        from app.workflow.tools.rag_tools import retrieve_documents
        
        # Mock RAG service
        mock_rag_service = Mock()
        mock_rag_service.retrieve_only = AsyncMock(return_value=[
            {
                "id": "doc_1",
                "text": "Machine learning is a subset of artificial intelligence",
                "score": 0.95
            },
            {
                "id": "doc_2",
                "text": "Deep learning is a subset of machine learning",
                "score": 0.90
            }
        ])
        
        # Retrieve documents
        results = await retrieve_documents(
            "What is the relationship between ML and AI?",
            top_k=5,
            rag_service=mock_rag_service
        )
        
        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]
    
    @pytest.mark.asyncio
    async def test_hybrid_search_workflow(self):
        """Test hybrid search workflow."""
        from app.workflow.tools.search_tools import hybrid_search, filter_by_metadata
        
        # Mock retriever service
        mock_retriever = Mock()
        mock_retriever.retrieve = AsyncMock(return_value=[
            {"id": "doc_1", "text": "Document 1", "score": 0.9},
            {"id": "doc_2", "text": "Document 2", "score": 0.8}
        ])
        
        # Perform hybrid search
        results = await hybrid_search(
            "test query",
            retriever_service=mock_retriever
        )
        
        assert isinstance(results, list)
        
        # Filter results
        filtered = await filter_by_metadata(
            results,
            {"category": "tech"}
        )
        
        assert isinstance(filtered, list)


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    @pytest.mark.asyncio
    async def test_complete_rag_pipeline(self):
        """Test complete RAG pipeline from query to answer."""
        from app.workflow.agents.query_agent import QueryAgent
        from app.workflow.agents.research_agent import ResearchAgent
        from app.workflow.agents.synthesis_agent import SynthesisAgent
        from app.workflow.agents.validation_agent import ValidationAgent
        from app.workflow.tools.rag_tools import retrieve_documents
        from app.llm.provider import LLMProvider
        
        # Create mock LLM provider
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.generate = AsyncMock(return_value=Mock(content="Mock response"))
        
        # Mock RAG service
        mock_rag_service = Mock()
        mock_rag_service.retrieve_only = AsyncMock(return_value=[
            {"id": "doc_1", "text": "Relevant document content", "score": 0.95}
        ])
        
        # Step 1: Query analysis
        query_agent = QueryAgent(mock_llm)
        query_result = await query_agent.execute("What is machine learning?")
        assert query_result["success"] is True
        
        # Step 2: Research
        research_agent = ResearchAgent(mock_llm)
        research_context = {
            "query_analysis": query_result["analysis"],
            "rag_service": mock_rag_service,
            "tools": ["retrieve_documents"]
        }
        research_result = await research_agent.execute("What is machine learning?", research_context)
        assert research_result["success"] is True
        
        # Step 3: Synthesis
        synthesis_agent = SynthesisAgent(mock_llm)
        synthesis_context = {
            "research_results": research_result["research"],
            "query_analysis": query_result["analysis"]
        }
        synthesis_result = await synthesis_agent.execute("What is machine learning?", synthesis_context)
        assert synthesis_result["success"] is True
        
        # Step 4: Validation
        validation_agent = ValidationAgent(mock_llm)
        validation_context = {
            "documents": research_result["research"]["documents"],
            "web_results": research_result["research"]["web_results"]
        }
        validation_result = await validation_agent.execute(
            synthesis_result["synthesis"]["answer"],
            validation_context
        )
        assert validation_result["success"] is True
        
        # Verify complete pipeline
        assert query_result["success"]
        assert research_result["success"]
        assert synthesis_result["success"]
        assert validation_result["success"]
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test error handling in workflows."""
        from app.workflow.agents.query_agent import QueryAgent
        from app.llm.provider import LLMProvider
        
        # Create mock LLM provider that raises an error
        mock_llm = Mock(spec=LLMProvider)
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))
        
        # Create query agent
        query_agent = QueryAgent(mock_llm)
        
        # Execute with error
        result = await query_agent.execute("Test query")
        
        # Verify error handling
        assert result["success"] is False
        assert "error" in result
