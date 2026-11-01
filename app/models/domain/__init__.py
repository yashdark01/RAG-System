"""
Domain models for the RAG system.
"""

from app.models.domain.agent import (
    Agent,
    AgentTask,
    AgentWorkflow,
    AgentType,
    AgentStatus,
    TaskStatus,
    WorkflowType
)
from app.models.domain.chat import (
    Message,
    Conversation,
    ConversationMetadata,
    MessageRole
)
from app.models.domain.document import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    FileType,
    DocumentStatus
)
from app.models.domain.rag import (
    RetrievalResult,
    RetrievalConfig,
    RAGQuery,
    RetrievalMetrics,
    RAGConfig,
    RetrievalStrategy
)

__all__ = [
    # Agent models
    "Agent",
    "AgentTask",
    "AgentWorkflow",
    "AgentType",
    "AgentStatus",
    "TaskStatus",
    "WorkflowType",
    # Chat models
    "Message",
    "Conversation",
    "ConversationMetadata",
    "MessageRole",
    # Document models
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "FileType",
    "DocumentStatus",
    # RAG models
    "RetrievalResult",
    "RetrievalConfig",
    "RAGQuery",
    "RetrievalMetrics",
    "RAGConfig",
    "RetrievalStrategy"
]
