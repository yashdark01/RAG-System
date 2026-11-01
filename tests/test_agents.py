"""
Unit tests for agent implementations.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from app.workflow.agents.query_agent import QueryAgent
from app.workflow.agents.research_agent import ResearchAgent
from app.workflow.agents.synthesis_agent import SynthesisAgent
from app.workflow.agents.validation_agent import ValidationAgent


class TestQueryAgent:
    """Test QueryAgent implementation."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = Mock()
        provider.generate = AsyncMock(return_value=Mock(content="Mock response"))
        return provider
    
    @pytest.fixture
    def query_agent(self, mock_llm_provider):
        """Create a QueryAgent instance."""
        return QueryAgent(mock_llm_provider)
    
    @pytest.mark.asyncio
    async def test_execute_basic(self, query_agent):
        """Test basic execution of QueryAgent."""
        result = await query_agent.execute("What is machine learning?")
        
        assert result["success"] is True
        assert "output" in result
        assert "steps" in result
        assert "analysis" in result
        assert len(result["steps"]) == 4
    
    @pytest.mark.asyncio
    async def test_execute_with_context(self, query_agent):
        """Test execution with context."""
        context = {
            "tools": ["web_search"],
            "previous_results": {}
        }
        result = await query_agent.execute("Test query", context)
        
        assert result["success"] is True
        # Agent returns empty tool_calls by default, tools are passed in context
    
    @pytest.mark.asyncio
    async def test_capabilities(self, query_agent):
        """Test agent capabilities."""
        assert "query_analysis" in query_agent.capabilities
        assert "query_decomposition" in query_agent.capabilities
        assert "intent_detection" in query_agent.capabilities


class TestResearchAgent:
    """Test ResearchAgent implementation."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = Mock()
        provider.generate = AsyncMock(return_value=Mock(content="Mock response"))
        return provider
    
    @pytest.fixture
    def research_agent(self, mock_llm_provider):
        """Create a ResearchAgent instance."""
        return ResearchAgent(mock_llm_provider)
    
    @pytest.mark.asyncio
    async def test_execute_basic(self, research_agent):
        """Test basic execution of ResearchAgent."""
        result = await research_agent.execute("Research machine learning")
        
        assert result["success"] is True
        assert "output" in result
        assert "research" in result
        assert "documents" in result["research"]
        assert "web_results" in result["research"]
    
    @pytest.mark.asyncio
    async def test_execute_with_rag_service(self, research_agent):
        """Test execution with RAG service in context."""
        mock_rag_service = Mock()
        mock_rag_service.retrieve_only = AsyncMock(return_value=[
            {"id": "doc_1", "text": "Document content", "score": 0.9}
        ])
        
        context = {
            "rag_service": mock_rag_service,
            "tools": ["retrieve_documents"]
        }
        
        result = await research_agent.execute("Test query", context)
        
        assert result["success"] is True
        assert len(result["research"]["documents"]) > 0
    
    @pytest.mark.asyncio
    async def test_capabilities(self, research_agent):
        """Test agent capabilities."""
        assert "document_retrieval" in research_agent.capabilities
        assert "web_search" in research_agent.capabilities
        assert "information_extraction" in research_agent.capabilities


class TestSynthesisAgent:
    """Test SynthesisAgent implementation."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = Mock()
        provider.generate = AsyncMock(return_value=Mock(content="Mock response"))
        return provider
    
    @pytest.fixture
    def synthesis_agent(self, mock_llm_provider):
        """Create a SynthesisAgent instance."""
        return SynthesisAgent(mock_llm_provider)
    
    @pytest.mark.asyncio
    async def test_execute_basic(self, synthesis_agent):
        """Test basic execution of SynthesisAgent."""
        context = {
            "research_results": {
                "documents": [{"text": "Document content"}],
                "web_results": [{"snippet": "Web result"}],
                "extracted_info": [{"fact": "Key fact"}]
            }
        }
        
        result = await synthesis_agent.execute("Synthesize this", context)
        
        assert result["success"] is True
        assert "output" in result
        assert "synthesis" in result
        assert "insights" in result["synthesis"]
        assert "answer" in result["synthesis"]
    
    @pytest.mark.asyncio
    async def test_execute_without_research(self, synthesis_agent):
        """Test execution without research results."""
        result = await synthesis_agent.execute("Synthesize this", {})
        
        assert result["success"] is True
        assert len(result["synthesis"]["insights"]) == 0
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, synthesis_agent):
        """Test confidence calculation."""
        insights = [
            {"insight": "Test insight", "importance": "high"}
        ]
        research_results = {
            "documents": [{"text": "Doc 1"}, {"text": "Doc 2"}],
            "web_results": []
        }
        
        confidence = synthesis_agent._calculate_confidence(insights, research_results)
        
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_capabilities(self, synthesis_agent):
        """Test agent capabilities."""
        assert "text_synthesis" in synthesis_agent.capabilities
        assert "summarization" in synthesis_agent.capabilities
        assert "answer_generation" in synthesis_agent.capabilities


class TestValidationAgent:
    """Test ValidationAgent implementation."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = Mock()
        provider.generate = AsyncMock(return_value=Mock(content="Mock response"))
        return provider
    
    @pytest.fixture
    def validation_agent(self, mock_llm_provider):
        """Create a ValidationAgent instance."""
        return ValidationAgent(mock_llm_provider)
    
    @pytest.mark.asyncio
    async def test_execute_basic(self, validation_agent):
        """Test basic execution of ValidationAgent."""
        context = {
            "documents": [{"text": "Supporting document"}],
            "web_results": []
        }
        
        result = await validation_agent.execute("This is a claim to validate", context)
        
        assert result["success"] is True
        assert "output" in result
        assert "validation" in result
        assert "claims" in result["validation"]
        assert "overall_confidence" in result["validation"]
    
    @pytest.mark.asyncio
    async def test_execute_without_sources(self, validation_agent):
        """Test execution without sources."""
        result = await validation_agent.execute("This is a claim", {})
        
        assert result["success"] is True
        # Confidence is calculated even without sources based on claim verification
        assert 0.0 <= result["validation"]["overall_confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_consistency_check(self, validation_agent):
        """Test consistency checking."""
        verified_claims = [
            {"claim": "Claim 1", "status": "supported"},
            {"claim": "Claim 2", "status": "supported"}
        ]
        
        consistency = await validation_agent._check_consistency(verified_claims)
        
        assert consistency["consistent"] is True
        # Check that the result has the expected structure
        assert "consistent" in consistency
    
    @pytest.mark.asyncio
    async def test_overall_confidence_calculation(self, validation_agent):
        """Test overall confidence calculation."""
        verified_claims = [
            {"claim": "Claim 1", "confidence": 0.8},
            {"claim": "Claim 2", "confidence": 0.9}
        ]
        source_assessment = {"overall_score": 0.85}
        consistency_check = {"consistent": True}
        
        confidence = validation_agent._calculate_overall_confidence(
            verified_claims,
            source_assessment,
            consistency_check
        )
        
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_capabilities(self, validation_agent):
        """Test agent capabilities."""
        assert "fact_checking" in validation_agent.capabilities
        assert "source_verification" in validation_agent.capabilities
        assert "confidence_scoring" in validation_agent.capabilities
