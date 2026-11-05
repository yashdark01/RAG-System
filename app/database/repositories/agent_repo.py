"""
Agent repository for agent execution logging.
"""

from typing import List, Optional
import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database.repositories.base import BaseRepository
from app.database.models import AgentExecution, AgentType


class AgentRepository(BaseRepository[AgentExecution]):
    """Repository for AgentExecution model operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(AgentExecution, session)
    
    async def log_execution(
        self,
        agent_type: AgentType,
        task: str,
        input_context: Optional[dict] = None,
        status: str = "running"
    ) -> AgentExecution:
        """
        Log a new agent execution.
        
        Args:
            agent_type: Type of agent
            task: Task description
            input_context: Input context
            status: Initial status
            
        Returns:
            Created execution log
        """
        return await self.create(
            agent_type=agent_type,
            task=task,
            input_context=input_context,
            status=status
        )
    
    async def update_execution(
        self,
        execution_id: uuid.UUID,
        output_result: Optional[dict] = None,
        steps: Optional[list] = None,
        tool_calls: Optional[list] = None,
        status: Optional[str] = None,
        error_message: Optional[str] = None,
        processing_time: Optional[float] = None,
        token_usage: Optional[dict] = None
    ) -> Optional[AgentExecution]:
        """
        Update agent execution with results.
        
        Args:
            execution_id: Execution ID
            output_result: Output result
            steps: Execution steps
            tool_calls: Tool calls made
            status: Final status
            error_message: Error message if failed
            processing_time: Processing time
            token_usage: Token usage stats
            
        Returns:
            Updated execution log
        """
        update_data = {}
        
        if output_result is not None:
            update_data["output_result"] = output_result
        if steps is not None:
            update_data["steps"] = steps
        if tool_calls is not None:
            update_data["tool_calls"] = tool_calls
        if status is not None:
            update_data["status"] = status
        if error_message is not None:
            update_data["error_message"] = error_message
        if processing_time is not None:
            update_data["processing_time"] = processing_time
        if token_usage is not None:
            update_data["token_usage"] = token_usage
        
        if status in ("completed", "failed"):
            update_data["completed_at"] = datetime.utcnow()
        
        return await self.update(execution_id, **update_data)
    
    async def get_recent_executions(
        self,
        limit: int = 20,
        agent_type: Optional[AgentType] = None
    ) -> List[AgentExecution]:
        """
        Get recent agent executions.
        
        Args:
            limit: Maximum results
            agent_type: Optional agent type filter
            
        Returns:
            List of executions
        """
        query = select(AgentExecution)
        
        if agent_type:
            query = query.where(AgentExecution.agent_type == agent_type)
        
        query = query.order_by(AgentExecution.created_at.desc()).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_execution_stats(self) -> dict:
        """
        Get agent execution statistics.
        
        Returns:
            Statistics dictionary
        """
        # Total executions
        total_query = select(func.count(AgentExecution.id))
        total_result = await self.session.execute(total_query)
        total = total_result.scalar()
        
        # Executions by type
        type_query = (
            select(AgentExecution.agent_type, func.count(AgentExecution.id))
            .group_by(AgentExecution.agent_type)
        )
        type_result = await self.session.execute(type_query)
        by_type = {agent_type: count for agent_type, count in type_result}
        
        # Executions by status
        status_query = (
            select(AgentExecution.status, func.count(AgentExecution.id))
            .group_by(AgentExecution.status)
        )
        status_result = await self.session.execute(status_query)
        by_status = {status: count for status, count in status_result}
        
        # Average processing time
        avg_time_query = select(func.avg(AgentExecution.processing_time))
        avg_time_result = await self.session.execute(avg_time_query)
        avg_time = avg_time_result.scalar() or 0
        
        # Success rate
        success_query = select(func.count(AgentExecution.id)).where(
            AgentExecution.status == "completed"
        )
        success_result = await self.session.execute(success_query)
        success_count = success_result.scalar()
        success_rate = success_count / total if total > 0 else 0
        
        return {
            "total_executions": total,
            "by_type": by_type,
            "by_status": by_status,
            "avg_processing_time": avg_time,
            "success_rate": success_rate
        }