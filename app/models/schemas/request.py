"""
Pydantic request models for API validation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class RAGQueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., description="The query to process", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=100)
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters for retrieval")
    temperature: Optional[float] = Field(0.7, description="LLM temperature", ge=0, le=2)
    stream: bool = Field(False, description="Whether to stream the response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "top_k": 5,
                "temperature": 0.7,
                "stream": False
            }
        }


class BatchRAGQueryRequest(BaseModel):
    """Request model for batch RAG queries."""
    queries: List[RAGQueryRequest] = Field(..., description="List of queries", min_items=1, max_items=10)


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    chunk_size: Optional[int] = Field(None, description="Override chunk size", ge=100, le=5000)
    chunk_overlap: Optional[int] = Field(None, description="Override chunk overlap", ge=0, le=1000)


class DocumentMetadataUpdate(BaseModel):
    """Request model for updating document metadata."""
    metadata: Dict[str, Any] = Field(..., description="Updated metadata")
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        if not v:
            raise ValueError('Metadata must not be empty')
        return v


class ChatRequest(BaseModel):
    """Request model for chat."""
    message: str = Field(..., description="User message", min_length=1)
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    history: Optional[List[Dict[str, str]]] = Field(None, description="Previous messages")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    use_rag: bool = Field(True, description="Whether to use RAG for context")
    temperature: Optional[float] = Field(0.7, description="LLM temperature", ge=0, le=2)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello! Can you help me?",
                "use_rag": True,
                "temperature": 0.7
            }
        }


class ChatHistoryRequest(BaseModel):
    """Request model for chat history."""
    conversation_id: str = Field(..., description="Conversation ID")


class AgentTaskRequest(BaseModel):
    """Request model for agent tasks."""
    task: str = Field(..., description="Task description", min_length=1)
    agent_type: str = Field("query", description="Type of agent to use")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    tools: Optional[List[str]] = Field(None, description="Specific tools to use")


class AgentConfig(BaseModel):
    """Configuration for a single agent in workflow."""
    agent_type: str = Field(..., description="Type of agent")
    task: str = Field(..., description="Agent-specific task")
    tools: Optional[List[str]] = Field(None, description="Tools for this agent")
    depends_on: Optional[List[str]] = Field(None, description="Dependencies on other agents")


class AgentWorkflowRequest(BaseModel):
    """Request model for agent workflows."""
    task: str = Field(..., description="Overall task description", min_length=1)
    agents: List[AgentConfig] = Field(..., description="Agent configurations", min_items=1)
    workflow_type: str = Field("sequential", description="Workflow type (sequential, parallel, adaptive)")
    max_iterations: int = Field(3, description="Maximum iterations", ge=1, le=10)
    
    @field_validator('workflow_type')
    @classmethod
    def validate_workflow_type(cls, v):
        allowed = ["sequential", "parallel", "adaptive"]
        if v not in allowed:
            raise ValueError(f"Workflow type must be one of: {allowed}")
        return v