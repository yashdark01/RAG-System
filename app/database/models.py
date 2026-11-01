"""
SQLAlchemy ORM models for the RAG system.
"""

from datetime import datetime
from typing import Optional, List
import uuid

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    Text, JSON, ForeignKey, Enum as SAEnum, Index, Table
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

import enum


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class DocumentStatus(str, enum.Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class AgentType(str, enum.Enum):
    """Types of agents."""
    QUERY = "query"
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


# Association table for conversation messages
conversation_messages = Table(
    "conversation_messages",
    Base.metadata,
    Column("conversation_id", String(36), ForeignKey("conversations.id")),
    Column("message_id", String(36), ForeignKey("messages.id"))
)


class Document(Base):
    """Document model for uploaded documents."""
    
    __tablename__ = "documents"
    __table_args__ = (
        Index("idx_document_status", "status"),
        Index("idx_document_created", "created_at"),
    )
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column(Integer)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64))
    
    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(DocumentStatus), default=DocumentStatus.UPLOADED
    )
    
    title: Mapped[Optional[str]] = mapped_column(String(500))
    description: Mapped[Optional[str]] = mapped_column(Text)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, default={})
    
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}')>"


class DocumentChunk(Base):
    """Document chunk model for indexed text chunks."""
    
    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("idx_chunk_document", "document_id"),
        Index("idx_chunk_index", "chunk_index"),
    )
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id"), nullable=False
    )
    
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    meta: Mapped[Optional[dict]] = mapped_column(JSON, default={})
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    
    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"


class Conversation(Base):
    """Conversation model for chat sessions."""
    
    __tablename__ = "conversations"
    __table_args__ = (
        Index("idx_conversation_created", "created_at"),
    )
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    title: Mapped[Optional[str]] = mapped_column(String(500))
    user_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    status: Mapped[str] = mapped_column(String(20), default="active")
    
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    
    meta: Mapped[Optional[dict]] = mapped_column(JSON, default={})
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Relationships
    messages: Mapped[List["Message"]] = relationship(
        "Message", secondary=conversation_messages, back_populates="conversations"
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, messages={self.message_count})>"


class Message(Base):
    """Message model for chat messages."""
    
    __tablename__ = "messages"
    __table_args__ = (
        Index("idx_message_created", "created_at"),
    )
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    
    sources: Mapped[Optional[list]] = mapped_column(JSON)  # RAG sources
    meta: Mapped[Optional[dict]] = mapped_column(JSON, default={})
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", secondary=conversation_messages, back_populates="messages"
    )
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role='{self.role}')>"


class AgentExecution(Base):
    """Agent execution log model."""
    
    __tablename__ = "agent_executions"
    __table_args__ = (
        Index("idx_agent_execution_type", "agent_type"),
        Index("idx_agent_execution_created", "created_at"),
    )
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    agent_type: Mapped[AgentType] = mapped_column(SAEnum(AgentType), nullable=False)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    
    input_context: Mapped[Optional[dict]] = mapped_column(JSON)
    output_result: Mapped[Optional[dict]] = mapped_column(JSON)
    
    steps: Mapped[Optional[list]] = mapped_column(JSON)
    tool_calls: Mapped[Optional[list]] = mapped_column(JSON)
    
    status: Mapped[str] = mapped_column(String(20), default="running")
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    token_usage: Mapped[Optional[dict]] = mapped_column(JSON)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    def __repr__(self) -> str:
        return f"<AgentExecution(id={self.id}, type='{self.agent_type}')>"


class QueryLog(Base):
    """Query logging model for analytics."""
    
    __tablename__ = "query_logs"
    __table_args__ = (
        Index("idx_query_log_created", "created_at"),
        Index("idx_query_log_user", "user_id"),
    )
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    user_id: Mapped[Optional[str]] = mapped_column(String(100))
    query: Mapped[str] = mapped_column(Text, nullable=False)
    
    query_type: Mapped[str] = mapped_column(String(50))  # rag, chat, agent
    response: Mapped[Optional[str]] = mapped_column(Text)
    
    retrieval_count: Mapped[Optional[int]] = mapped_column(Integer)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    token_usage: Mapped[Optional[dict]] = mapped_column(JSON)
    
    error: Mapped[Optional[str]] = mapped_column(Text)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, default={})
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    def __repr__(self) -> str:
        return f"<QueryLog(id={self.id}, query='{self.query[:50]}')>"


class APIKey(Base):
    """API key model for authentication."""
    
    __tablename__ = "api_keys"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    
    user_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    permissions: Mapped[Optional[list]] = mapped_column(JSON)
    
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name='{self.name}')>"


class CacheEntry(Base):
    """Cache entry model for persistent caching."""
    
    __tablename__ = "cache_entries"
    __table_args__ = (
        Index("idx_cache_key", "cache_key"),
        Index("idx_cache_expires", "expires_at"),
    )
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    
    cache_key: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    cache_value: Mapped[str] = mapped_column(Text, nullable=False)
    
    ttl: Mapped[int] = mapped_column(Integer, default=3600)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    def __repr__(self) -> str:
        return f"<CacheEntry(key='{self.cache_key}')>"