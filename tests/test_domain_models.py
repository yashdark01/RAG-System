"""
Unit tests for domain models.
"""

import pytest
from datetime import datetime
from app.models.domain.agent import Agent, AgentTask, AgentWorkflow, AgentType, AgentStatus, TaskStatus, WorkflowType
from app.models.domain.chat import Message, Conversation, ConversationMetadata, MessageRole
from app.models.domain.document import Document, DocumentChunk, DocumentMetadata, FileType, DocumentStatus
from app.models.domain.rag import RetrievalResult, RetrievalConfig, RAGQuery, RetrievalMetrics, RAGConfig, RetrievalStrategy


class TestAgentModels:
    """Test agent-related domain models."""
    
    def test_agent_creation(self):
        """Test creating an Agent instance."""
        agent = Agent(
            id="agent_123",
            name="Test Agent",
            agent_type=AgentType.QUERY,
            config={"temperature": 0.7},
            capabilities=["query_analysis"]
        )
        assert agent.id == "agent_123"
        assert agent.name == "Test Agent"
        assert agent.agent_type == AgentType.QUERY
        assert agent.status == AgentStatus.IDLE
        assert len(agent.capabilities) == 1
    
    def test_agent_task_creation(self):
        """Test creating an AgentTask instance."""
        task = AgentTask(
            id="task_123",
            agent_id="agent_123",
            task="Analyze this query",
            context={"user_id": "user_123"}
        )
        assert task.id == "task_123"
        assert task.agent_id == "agent_123"
        assert task.status == TaskStatus.PENDING
        assert task.processing_time is None
    
    def test_agent_workflow_creation(self):
        """Test creating an AgentWorkflow instance."""
        workflow = AgentWorkflow(
            id="workflow_123",
            name="Test Workflow",
            agents_config=[{"agent_type": "query", "task": "Analyze"}],
            workflow_type=WorkflowType.SEQUENTIAL
        )
        assert workflow.id == "workflow_123"
        assert workflow.workflow_type == WorkflowType.SEQUENTIAL
        assert workflow.max_iterations == 3
        assert workflow.status == TaskStatus.PENDING


class TestChatModels:
    """Test chat-related domain models."""
    
    def test_message_creation(self):
        """Test creating a Message instance."""
        message = Message(
            id="msg_123",
            role=MessageRole.USER,
            content="Hello, how can you help me?"
        )
        assert message.id == "msg_123"
        assert message.role == MessageRole.USER
        assert message.content == "Hello, how can you help me?"
        assert message.token_count is None
    
    def test_conversation_creation(self):
        """Test creating a Conversation instance."""
        conversation = Conversation(
            id="conv_123"
        )
        assert conversation.id == "conv_123"
        assert len(conversation.messages) == 0
        assert conversation.is_active is True
    
    def test_add_message(self):
        """Test adding a message to a conversation."""
        conversation = Conversation(id="conv_123")
        message = Message(
            id="msg_123",
            role=MessageRole.USER,
            content="Hello"
        )
        conversation.add_message(message)
        assert len(conversation.messages) == 1
        assert conversation.metadata.message_count == 1
    
    def test_get_last_n_messages(self):
        """Test getting the last n messages."""
        conversation = Conversation(id="conv_123")
        for i in range(5):
            conversation.add_message(
                Message(
                    id=f"msg_{i}",
                    role=MessageRole.USER,
                    content=f"Message {i}"
                )
            )
        last_3 = conversation.get_last_n_messages(3)
        assert len(last_3) == 3
        assert last_3[0].content == "Message 2"
    
    def test_get_messages_by_role(self):
        """Test getting messages by role."""
        conversation = Conversation(id="conv_123")
        conversation.add_message(Message(id="msg_1", role=MessageRole.USER, content="User message"))
        conversation.add_message(Message(id="msg_2", role=MessageRole.ASSISTANT, content="Assistant message"))
        conversation.add_message(Message(id="msg_3", role=MessageRole.USER, content="Another user message"))
        
        user_messages = conversation.get_messages_by_role(MessageRole.USER)
        assert len(user_messages) == 2


class TestDocumentModels:
    """Test document-related domain models."""
    
    def test_document_metadata_creation(self):
        """Test creating DocumentMetadata."""
        metadata = DocumentMetadata(
            source="document.pdf",
            file_type=FileType.PDF,
            title="Research Paper"
        )
        assert metadata.source == "document.pdf"
        assert metadata.file_type == FileType.PDF
        assert metadata.title == "Research Paper"
    
    def test_document_chunk_creation(self):
        """Test creating DocumentChunk."""
        chunk = DocumentChunk(
            id="chunk_123",
            document_id="doc_123",
            text="This is a chunk of text",
            chunk_index=0
        )
        assert chunk.id == "chunk_123"
        assert chunk.document_id == "doc_123"
        assert chunk.chunk_index == 0
        assert chunk.embedding is None
    
    def test_document_creation(self):
        """Test creating Document."""
        document = Document(
            id="doc_123",
            filename="document.pdf",
            metadata=DocumentMetadata(
                source="document.pdf",
                file_type=FileType.PDF
            )
        )
        assert document.id == "doc_123"
        assert document.filename == "document.pdf"
        assert document.status == DocumentStatus.PENDING
        assert document.indexed is False
    
    def test_add_chunk(self):
        """Test adding a chunk to a document."""
        document = Document(
            id="doc_123",
            filename="document.pdf",
            metadata=DocumentMetadata(
                source="document.pdf",
                file_type=FileType.PDF
            )
        )
        chunk = DocumentChunk(
            id="chunk_123",
            document_id="doc_123",
            text="Chunk text",
            chunk_index=0
        )
        document.add_chunk(chunk)
        assert document.chunk_count == 1
        assert len(document.chunks) == 1
    
    def test_get_total_tokens(self):
        """Test getting total tokens."""
        document = Document(
            id="doc_123",
            filename="document.pdf",
            metadata=DocumentMetadata(
                source="document.pdf",
                file_type=FileType.PDF
            )
        )
        document.add_chunk(
            DocumentChunk(
                id="chunk_1",
                document_id="doc_123",
                text="Chunk 1",
                chunk_index=0,
                token_count=100
            )
        )
        document.add_chunk(
            DocumentChunk(
                id="chunk_2",
                document_id="doc_123",
                text="Chunk 2",
                chunk_index=1,
                token_count=150
            )
        )
        assert document.get_total_tokens() == 250


class TestRAGModels:
    """Test RAG-related domain models."""
    
    def test_retrieval_result_creation(self):
        """Test creating RetrievalResult."""
        result = RetrievalResult(
            id="result_123",
            text="Retrieved text",
            score=0.95,
            rank=1,
            retrieval_method="semantic"
        )
        assert result.id == "result_123"
        assert result.score == 0.95
        assert result.rank == 1
    
    def test_retrieval_config_creation(self):
        """Test creating RetrievalConfig."""
        config = RetrievalConfig(
            top_k=10,
            similarity_threshold=0.8,
            strategy=RetrievalStrategy.SEMANTIC
        )
        assert config.top_k == 10
        assert config.similarity_threshold == 0.8
        assert config.strategy == RetrievalStrategy.SEMANTIC
    
    def test_rag_query_creation(self):
        """Test creating RAGQuery."""
        query = RAGQuery(
            id="query_123",
            query="What is machine learning?",
            answer="Machine learning is a subset of AI..."
        )
        assert query.id == "query_123"
        assert query.query == "What is machine learning?"
        assert query.confidence == 0.0
        assert query.from_cache is False
    
    def test_retrieval_metrics_creation(self):
        """Test creating RetrievalMetrics."""
        metrics = RetrievalMetrics(
            total_queries=100,
            avg_retrieval_time=0.5,
            cache_hits=60,
            cache_misses=40,
            cache_hit_rate=0.6
        )
        assert metrics.total_queries == 100
        assert metrics.cache_hit_rate == 0.6
    
    def test_rag_config_creation(self):
        """Test creating RAGConfig."""
        config = RAGConfig(
            max_context_tokens=4000,
            min_relevance_threshold=0.6,
            enable_validation=True
        )
        assert config.max_context_tokens == 4000
        assert config.enable_validation is True
