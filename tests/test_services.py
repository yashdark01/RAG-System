"""
Unit tests for core application services: DocumentService, RAGService, ChatService, and AgentService.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from app.services.document_service import DocumentService
from app.services.rag_service import RAGService
from app.services.chat_service import ChatService
from app.services.agent_service import AgentService
from app.rag.retriever import RetrievalResult
from app.llm.provider import LLMResponse


class TestDocumentService:
    """Test suite for DocumentService."""

    @pytest.fixture
    def mock_processor(self):
        processor = Mock()
        processor.process_document = AsyncMock(return_value=[
            {
                "id": "doc_1_0",
                "text": "Chunk 1 content.",
                "metadata": {"chunk_id": "doc_1_0", "source": "test.txt"}
            },
            {
                "id": "doc_1_1",
                "text": "Chunk 2 content.",
                "metadata": {"chunk_id": "doc_1_1", "source": "test.txt"}
            }
        ])
        return processor

    @pytest.fixture
    def mock_vector_store(self):
        store = Mock()
        store.add_document = AsyncMock(return_value=True)
        return store

    @pytest.mark.asyncio
    async def test_process_and_index_success(self, mock_processor, mock_vector_store):
        """Verify document is parsed into chunks and added to vector store."""
        service = DocumentService(processor=mock_processor, vector_store=mock_vector_store)
        result = await service.process_and_index(
            content=b"Sample content for indexing",
            filename="sample.txt",
            metadata={"department": "engineering"}
        )

        assert "document_id" in result
        assert result["chunks_count"] == 2
        assert result["indexed_count"] == 2
        assert mock_vector_store.add_document.await_count == 2
        assert service._documents_processed == 1


class TestRAGService:
    """Test suite for RAGService."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = Mock()
        retriever.retrieve = AsyncMock(return_value=[
            RetrievalResult(
                id="doc_1",
                text="LangChain 1.3 provides unified agentic orchestrators and tools.",
                score=0.95,
                metadata={"title": "LangChain Docs", "source": "docs.txt"}
            )
        ])
        return retriever

    @pytest.fixture
    def mock_llm_provider(self):
        llm = Mock()
        llm.generate = AsyncMock(return_value=LLMResponse(
            content="LangChain 1.3 simplifies building RAG systems with native components.",
            model="test-model"
        ))
        return llm

    @pytest.mark.asyncio
    async def test_rag_query(self, mock_retriever, mock_llm_provider):
        """Verify full RAG query flow: retrieval -> context -> generation."""
        rag = RAGService(retriever=mock_retriever, llm_provider=mock_llm_provider)
        response = await rag.query("What is LangChain 1.3?")

        assert "answer" in response
        assert "LangChain 1.3" in response["answer"]
        assert response["retrieval_count"] == 1
        assert len(response["sources"]) == 1
        assert response["confidence"] > 0

    @pytest.mark.asyncio
    async def test_retrieve_only(self, mock_retriever, mock_llm_provider):
        """Verify retrieve_only returns documents without calling LLM."""
        rag = RAGService(retriever=mock_retriever, llm_provider=mock_llm_provider)
        docs = await rag.retrieve_only("Query without generation", top_k=1)

        assert len(docs) == 1
        assert docs[0]["id"] == "doc_1"
        assert mock_llm_provider.generate.await_count == 0

    @pytest.mark.asyncio
    async def test_batch_query(self, mock_retriever, mock_llm_provider):
        """Verify concurrent batch query processing."""
        rag = RAGService(retriever=mock_retriever, llm_provider=mock_llm_provider)
        results = await rag.batch_query(["Question 1", "Question 2"])

        assert len(results) == 2
        assert all("answer" in r for r in results)


class TestChatService:
    """Test suite for ChatService."""

    @pytest.fixture
    def mock_rag_service(self):
        rag = Mock()
        rag.query = AsyncMock(return_value={
            "answer": "RAG provides grounded context.",
            "sources": [{"title": "Doc1", "snippet": "Context", "url": "doc.txt"}],
            "confidence": 0.9
        })
        return rag

    @pytest.fixture
    def mock_llm_provider(self):
        llm = Mock()
        llm.generate = AsyncMock(return_value=LLMResponse(
            content="Hello! Here is your answer with retrieved context.",
            model="test-chat-model"
        ))
        return llm

    @pytest.mark.asyncio
    async def test_chat_with_rag(self, mock_rag_service, mock_llm_provider):
        """Verify chat turn with RAG integration enabled."""
        chat_service = ChatService(rag_service=mock_rag_service, llm_provider=mock_llm_provider)
        response = await chat_service.chat(
            message="Explain RAG",
            conversation_id="conv-1",
            use_rag=True
        )

        assert "response" in response
        assert response["conversation_id"] == "conv-1"
        assert mock_rag_service.query.await_count == 1
        assert len(chat_service._conversations["conv-1"]) >= 2

    @pytest.mark.asyncio
    async def test_chat_without_rag(self, mock_rag_service, mock_llm_provider):
        """Verify direct LLM chat when use_rag is False."""
        chat_service = ChatService(rag_service=mock_rag_service, llm_provider=mock_llm_provider)
        response = await chat_service.chat(
            message="Say hi",
            conversation_id="conv-2",
            use_rag=False
        )

        assert "response" in response
        assert mock_rag_service.query.await_count == 0


class TestAgentService:
    """Test suite for AgentService."""

    @pytest.fixture
    def mock_llm_provider(self):
        llm = Mock()
        llm.generate = AsyncMock(return_value=LLMResponse(
            content='{"analysis": "User wants overview", "complexity": "low"}',
            model="test-agent-model"
        ))
        return llm

    @pytest.mark.asyncio
    async def test_execute_query_agent_task(self, mock_llm_provider):
        """Verify single query agent task execution."""
        agent_service = AgentService(llm_provider=mock_llm_provider)
        result = await agent_service.execute_task(
            task="Analyze the question: What are transformers?",
            agent_type="query"
        )

        assert "output" in result
        assert result["agent_type"] == "query"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_unknown_agent_error(self, mock_llm_provider):
        """Verify error result when invalid agent type is specified."""
        agent_service = AgentService(llm_provider=mock_llm_provider)
        result = await agent_service.execute_task(
            task="Invalid task",
            agent_type="nonexistent_agent"
        )

        assert "Error executing task" in result["output"]

