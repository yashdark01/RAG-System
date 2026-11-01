"""
Multi-agent workflow endpoints.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import StreamingResponse

from app.dependencies import dependencies
from app.models.schemas.request import AgentTaskRequest, AgentWorkflowRequest
from app.models.schemas.response import AgentResponse, AgentWorkflowResponse
from app.logging.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/execute",
    response_model=AgentResponse,
    summary="Execute agent task",
    description="Execute a task using a single agent"
)
async def execute_agent_task(
    request: AgentTaskRequest,
    agent_service = Depends(dependencies.get_agent_service)
) -> AgentResponse:
    """
    Execute a task with an AI agent.
    
    - **task**: Task description
    - **agent_type**: Type of agent to use (query, research, synthesis, validation)
    - **context**: Additional context for the agent
    - **tools**: Specific tools the agent can use
    """
    try:
        result = await agent_service.execute_task(
            task=request.task,
            agent_type=request.agent_type,
            context=request.context,
            tools=request.tools
        )
        
        return AgentResponse(
            task=request.task,
            agent_type=request.agent_type,
            result=result["output"],
            steps=result.get("steps", []),
            tool_calls=result.get("tool_calls", []),
            processing_time=result.get("processing_time"),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Agent task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/execute/stream",
    summary="Stream agent execution tokens",
    description="Execute agent task and stream real-time tokens via Server-Sent Events"
)
async def stream_agent_task(
    request: AgentTaskRequest,
    agent_service = Depends(dependencies.get_agent_service)
):
    """Stream agent reasoning tokens in real-time."""
    async def token_generator():
        try:
            session_id = request.context.get("session_id", "default_session") if request.context else "default_session"
            async for token in agent_service.orchestrator.astream_tokens(
                task=request.task,
                session_id=session_id
            ):
                yield f"data: {token}\n\n"
        except Exception as e:
            logger.error(f"Agent token streaming failed: {e}")
            yield f"data: [Error: {str(e)}]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@router.post(
    "/workflow",
    response_model=AgentWorkflowResponse,
    summary="Execute agent workflow",
    description="Execute a multi-agent orchestrated workflow"
)
async def execute_workflow(
    request: AgentWorkflowRequest,
    agent_service = Depends(dependencies.get_agent_service)
) -> AgentWorkflowResponse:
    """
    Execute a multi-agent workflow.
    
    - **task**: Overall task description
    - **agents**: Configuration for each agent in workflow
    - **workflow_type**: Type of workflow (sequential, parallel, adaptive)
    - **max_iterations**: Maximum workflow iterations
    """
    try:
        result = await agent_service.execute_workflow(
            task=request.task,
            agents_config=request.agents,
            workflow_type=request.workflow_type,
            max_iterations=request.max_iterations
        )
        
        return AgentWorkflowResponse(
            task=request.task,
            workflow_type=request.workflow_type,
            final_result=result["final_result"],
            agent_results=result.get("agent_results", []),
            iterations=result.get("iterations", 0),
            total_processing_time=result.get("total_time"),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/research",
    response_model=AgentResponse,
    summary="Research task",
    description="Execute a research task using the research agent"
)
async def research_task(
    query: str = Body(...),
    depth: int = Body(3),
    sources: Optional[List[str]] = Body(None),
    agent_service = Depends(dependencies.get_agent_service)
) -> AgentResponse:
    """
    Execute a research task.
    
    - **query**: Research query
    - **depth**: Research depth level
    - **sources**: Specific sources to search
    """
    try:
        result = await agent_service.execute_task(
            task=query,
            agent_type="research",
            context={"depth": depth, "sources": sources}
        )
        
        return AgentResponse(
            task=query,
            agent_type="research",
            result=result["output"],
            steps=result.get("steps", []),
            processing_time=result.get("processing_time"),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Research task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/agents",
    summary="List available agents",
    description="Get list of available agent types and their capabilities"
)
async def list_agents() -> Dict[str, Any]:
    """List available agent types."""
    agents = {
        "query": {
            "description": "Query understanding and decomposition agent",
            "capabilities": ["query_analysis", "query_decomposition", "query_reformulation"]
        },
        "research": {
            "description": "Information gathering and research agent",
            "capabilities": ["web_search", "document_retrieval", "information_extraction"]
        },
        "synthesis": {
            "description": "Information synthesis and summarization agent",
            "capabilities": ["text_synthesis", "summarization", "answer_generation"]
        },
        "validation": {
            "description": "Fact-checking and validation agent",
            "capabilities": ["fact_checking", "source_verification", "confidence_scoring"]
        }
    }
    
    return {
        "agents": agents,
        "total": len(agents),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get(
    "/tools",
    summary="List available tools",
    description="Get list of tools available to agents"
)
async def list_tools() -> Dict[str, Any]:
    """List available tools for agents."""
    tools = {
        "rag_tools": [
            {"name": "retrieve_documents", "description": "Retrieve documents from knowledge base"},
            {"name": "search_similar", "description": "Search for similar documents"},
            {"name": "get_context", "description": "Get formatted context for query"}
        ],
        "search_tools": [
            {"name": "web_search", "description": "Search the web for information"},
            {"name": "news_search", "description": "Search recent news articles"},
            {"name": "academic_search", "description": "Search academic papers"}
        ],
        "verification_tools": [
            {"name": "verify_claim", "description": "Verify a factual claim"},
            {"name": "check_sources", "description": "Check source credibility"},
            {"name": "cross_reference", "description": "Cross-reference information"}
        ]
    }
    
    return {
        "tools": tools,
        "total_categories": len(tools),
        "timestamp": datetime.utcnow().isoformat()
    }