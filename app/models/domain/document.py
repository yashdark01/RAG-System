"""
Domain models for document-related entities.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class FileType(str, Enum):
    """Supported file types for documents."""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    XML = "xml"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    UNKNOWN = "unknown"


class DocumentStatus(str, Enum):
    """Status of document processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    source: str = Field(..., description="Source of the document (filename or URL)")
    file_type: FileType = Field(..., description="Type of file")
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Upload timestamp")
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    language: Optional[str] = Field(None, description="Document language")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": "document.pdf",
                "file_type": "pdf",
                "uploaded_at": "2024-01-01T00:00:00Z",
                "title": "Research Paper",
                "tags": ["research", "ai"]
            }
        }
    )


class DocumentChunk(BaseModel):
    """Represents a chunk of a document."""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="ID of the parent document")
    text: str = Field(..., description="Chunk text content")
    embedding: Optional[List[float]] = Field(None, description="Embedding vector")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    chunk_index: int = Field(..., description="Index of chunk in document")
    token_count: Optional[int] = Field(None, description="Estimated token count")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "chunk_123",
                "document_id": "doc_123",
                "text": "This is a chunk of text...",
                "chunk_index": 0,
                "token_count": 150
            }
        }
    )


class Document(BaseModel):
    """Represents a document in the system."""
    id: str = Field(..., description="Unique document identifier")
    content: Optional[str] = Field(None, description="Document content (for text files)")
    filename: str = Field(..., description="Original filename")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Document chunks")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Processing status")
    chunk_count: int = Field(default=0, description="Number of chunks")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    indexed: bool = Field(default=False, description="Whether document is indexed in vector store")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "doc_123",
                "filename": "document.pdf",
                "status": "completed",
                "chunk_count": 10,
                "indexed": True
            }
        }
    )

    def add_chunk(self, chunk: DocumentChunk) -> None:
        """Add a chunk to the document."""
        self.chunks.append(chunk)
        self.chunk_count = len(self.chunks)
        self.updated_at = datetime.now(timezone.utc)

    def get_chunk_by_index(self, index: int) -> Optional[DocumentChunk]:
        """Get a chunk by its index."""
        for chunk in self.chunks:
            if chunk.chunk_index == index:
                return chunk
        return None

    def get_total_tokens(self) -> int:
        """Get total token count across all chunks."""
        return sum(chunk.token_count or 0 for chunk in self.chunks)
