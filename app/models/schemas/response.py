"""
Pydantic response models for API responses.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    version: Optional[str] = Field(None, description="Service version")


class RAGResponse(BaseModel):
    """Response model for RAG queries."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field([], description="Source documents")
    retrieval_count: int = Field(0, description="Number of documents retrieved")
    confidence: Optional[float] = Field(None, description="Confidence score")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class RetrievalResult(BaseModel):
    """Single retrieval result."""
    id: str = Field(..., description="Document ID")
    text: str = Field(..., description="Document text")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field({}, description="Document metadata")
    rank: int = Field(..., description="Result rank")


class RetrievalResponse(BaseModel):
    """Response model for document retrieval."""
    query: str = Field(..., description="Search query")
    results: List[RetrievalResult] = Field([], description="Retrieved documents")
    total_found: int = Field(0, description="Total documents found")
    timestamp: str = Field(..., description="Response timestamp")


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status")
    chunks_count: int = Field(0, description="Number of chunks")
    metadata: Dict[str, Any] = Field({}, description="Document metadata")
    created_at: Optional[str] = Field(None, description="Creation timestamp")


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: List[DocumentResponse] = Field([], description="Document list")
    total: int = Field(0, description="Total documents")
    limit: int = Field(10, description="Page size")
    offset: int = Field(0, description="Page offset")


class ChatResponse(BaseModel):
    """Response model for chat."""
    message: str = Field(..., description="AI response message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    sources: List[Dict[str, Any]] = Field([], description="Source documents")
    processing_time: Optional[float] = Field(None, description="Processing time")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage")
    timestamp: str = Field(..., description="Response timestamp")


class AgentStep(BaseModel):
    """Single step in agent execution."""
    step_number: int = Field(..., description="Step number")
    action: str = Field(..., description="Action taken")
    observation: Optional[str] = Field(None, description="Action result")
    tool_call: Optional[Dict[str, Any]] = Field(None, description="Tool call details")


class AgentResponse(BaseModel):
    """Response model for agent tasks."""
    task: str = Field(..., description="Original task")
    agent_type: str = Field(..., description="Agent type used")
    result: str = Field(..., description="Final result")
    steps: List[AgentStep] = Field([], description="Execution steps")
    tool_calls: List[Dict[str, Any]] = Field([], description="Tool calls made")
    processing_time: Optional[float] = Field(None, description="Processing time")
    timestamp: str = Field(..., description="Response timestamp")


class AgentWorkflowResponse(BaseModel):
    """Response model for agent workflows."""
    task: str = Field(..., description="Overall task")
    workflow_type: str = Field(..., description="Workflow type")
    final_result: str = Field(..., description="Final synthesized result")
    agent_results: List[AgentResponse] = Field([], description="Individual agent results")
    iterations: int = Field(0, description="Number of iterations")
    total_processing_time: Optional[float] = Field(None, description="Total processing time")
    timestamp: str = Field(..., description="Response timestamp")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    detail: Optional[str] = Field(None, description="Error detail")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())