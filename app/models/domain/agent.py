"""
Domain models for agent-related entities.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class AgentType(str, Enum):
    """Types of agents available in the system."""
    QUERY = "query"
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


class AgentStatus(str, Enum):
    """Status of an agent."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Status of an agent task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowType(str, Enum):
    """Types of agent workflows."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


class Agent(BaseModel):
    """Represents an AI agent in the system."""
    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    agent_type: AgentType = Field(..., description="Type of agent")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Current status")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "agent_123",
                "name": "Query Agent",
                "agent_type": "query",
                "config": {"temperature": 0.7},
                "status": "idle",
                "capabilities": ["query_analysis", "query_decomposition"]
            }
        }
    )


class AgentTask(BaseModel):
    """Represents a task assigned to an agent."""
    id: str = Field(..., description="Unique task identifier")
    agent_id: str = Field(..., description="ID of the agent assigned to this task")
    task: str = Field(..., description="Task description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the task")
    result: Optional[Dict[str, Any]] = Field(None, description="Task execution result")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    error: Optional[str] = Field(None, description="Error message if task failed")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Execution steps")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls made during execution")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "task_123",
                "agent_id": "agent_123",
                "task": "Analyze this query",
                "context": {"user_id": "user_123"},
                "status": "completed",
                "processing_time": 2.5
            }
        }
    )


class AgentWorkflow(BaseModel):
    """Represents a multi-agent workflow."""
    id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    agents_config: List[Dict[str, Any]] = Field(..., description="Configuration for each agent")
    workflow_type: WorkflowType = Field(default=WorkflowType.SEQUENTIAL, description="Workflow execution type")
    max_iterations: int = Field(default=3, ge=1, le=10, description="Maximum iterations")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Workflow status")
    final_result: Optional[str] = Field(None, description="Final workflow result")
    agent_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results from each agent")
    iterations: int = Field(default=0, description="Number of iterations executed")
    total_time: Optional[float] = Field(None, description="Total execution time in seconds")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "workflow_123",
                "name": "Research Workflow",
                "workflow_type": "sequential",
                "max_iterations": 3,
                "agents_config": [
                    {"agent_type": "query", "task": "Analyze query"},
                    {"agent_type": "research", "task": "Gather information"}
                ]
            }
        }
    )
