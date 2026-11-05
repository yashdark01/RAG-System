"""
Agent-related exceptions.
"""

from app.exceptions.base import RAGSystemException


class AgentException(RAGSystemException):
    """Base exception for agent errors."""
    
    def __init__(
        self,
        message: str = "Agent operation failed",
        agent_type: str = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if agent_type:
            details['agent_type'] = agent_type
        super().__init__(
            message=message,
            error_code="AGENT_ERROR",
            status_code=500,
            details=details,
            **kwargs
        )


class AgentExecutionException(AgentException):
    """Exception for agent execution failures."""
    
    def __init__(
        self,
        message: str = "Agent execution failed",
        task: str = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if task:
            details['task'] = task[:200]
        super().__init__(
            message=message,
            error_code="AGENT_EXECUTION_ERROR",
            details=details,
            **kwargs
        )


class AgentTimeoutException(AgentException):
    """Exception when agent execution times out."""
    
    def __init__(
        self,
        timeout: int = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if timeout:
            details['timeout'] = timeout
        super().__init__(
            message=f"Agent execution timed out after {timeout}s",
            error_code="AGENT_TIMEOUT",
            status_code=504,
            details=details,
            **kwargs
        )


class AgentConfigurationException(AgentException):
    """Exception for invalid agent configuration."""
    
    def __init__(
        self,
        message: str = "Invalid agent configuration",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="AGENT_CONFIGURATION_ERROR",
            status_code=400,
            **kwargs
        )


class AgentToolException(AgentException):
    """Exception for tool execution errors."""
    
    def __init__(
        self,
        message: str = "Tool execution failed",
        tool_name: str = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if tool_name:
            details['tool_name'] = tool_name
        super().__init__(
            message=message,
            error_code="AGENT_TOOL_ERROR",
            details=details,
            **kwargs
        )


class WorkflowException(AgentException):
    """Exception for workflow errors."""
    
    def __init__(
        self,
        message: str = "Workflow execution failed",
        workflow_type: str = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if workflow_type:
            details['workflow_type'] = workflow_type
        super().__init__(
            message=message,
            error_code="WORKFLOW_ERROR",
            details=details,
            **kwargs
        )


class AgentLimitExceededException(AgentException):
    """Exception when agent iteration limit is exceeded."""
    
    def __init__(
        self,
        max_iterations: int = None,
        **kwargs
    ):
        super().__init__(
            message=f"Agent iteration limit exceeded: {max_iterations}",
            error_code="AGENT_LIMIT_EXCEEDED",
            status_code=429,
            **kwargs
        )