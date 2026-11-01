"""
Multi-agent orchestration service.
"""

from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import time
import asyncio
import uuid

from app.config import settings
from app.logging.logger import get_logger
from app.llm.provider import LLMProvider, Message
from app.llm.prompt_templates import PromptManager
from app.workflow.base import BaseAgent
from app.workflow.agents.query_agent import QueryAgent
from app.workflow.agents.research_agent import ResearchAgent
from app.workflow.agents.synthesis_agent import SynthesisAgent
from app.workflow.agents.validation_agent import ValidationAgent
from app.workflow.orchestrator import AgentOrchestrator

logger = get_logger(__name__)


class WorkflowType(str, Enum):
    """Types of agent workflows."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


class AgentService:
    """
    Service for managing and executing agent workflows.
    Supports multiple agent types and orchestration patterns.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[PromptManager] = None,
        max_iterations: int = 3,
        timeout: int = 300
    ):
        """
        Initialize agent service.
        
        Args:
            llm_provider: LLM provider for agents
            prompt_manager: Optional prompt manager
            max_iterations: Maximum workflow iterations
            timeout: Execution timeout in seconds
        """
        self.llm = llm_provider
        self.prompt_manager = prompt_manager or PromptManager()
        self.max_iterations = max_iterations
        self.timeout = timeout
        
        # Initialize agent classes
        self._agent_classes = {
            "query": QueryAgent,
            "research": ResearchAgent,
            "synthesis": SynthesisAgent,
            "validation": ValidationAgent
        }
        
        # Initialize orchestrator
        self.orchestrator = AgentOrchestrator(
            llm_provider=llm_provider,
            max_iterations=max_iterations
        )
        
        # Metrics
        self._tasks_executed = 0
        self._workflows_executed = 0
        self._total_time = 0.0
        
        logger.info("AgentService initialized")
    
    async def execute_task(
        self,
        task: str,
        agent_type: str = "query",
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a task with a single agent.
        
        Args:
            task: Task description
            agent_type: Type of agent to use
            context: Additional context
            tools: Specific tools to use
            **kwargs: Additional parameters
            
        Returns:
            Execution result
        """
        start_time = time.time()
        self._tasks_executed += 1
        
        try:
            # Check if delegating to LangChain tool-calling orchestrator
            if agent_type in ("orchestrator", "tool_calling", "auto"):
                session_id = context.get("session_id", "default_session") if context else "default_session"
                return await self.orchestrator.execute(task=task, session_id=session_id)

            # Get agent class
            agent_class = self._agent_classes.get(agent_type)
            if not agent_class:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Create agent instance
            agent = agent_class(
                llm_provider=self.llm,
                prompt_manager=self.prompt_manager
            )
            
            # Prepare context
            execution_context = context or {}
            if tools:
                execution_context["tools"] = self._get_tools(tools)
            
            # Execute agent with timeout
            try:
                result = await asyncio.wait_for(
                    agent.execute(task=task, context=execution_context),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Agent execution timed out after {self.timeout}s")
                result = {
                    "output": "Task execution timed out. Please try a simpler query.",
                    "steps": [],
                    "tool_calls": [],
                    "success": False,
                    "error": "Timeout"
                }
            
            # Process result
            processing_time = time.time() - start_time
            self._total_time += processing_time
            
            return {
                "output": result.get("output", ""),
                "steps": result.get("steps", []),
                "tool_calls": result.get("tool_calls", []),
                "success": result.get("success", True),
                "processing_time": processing_time,
                "agent_type": agent_type
            }
            
        except Exception as e:
            logger.error(f"Agent task execution failed: {e}")
            
            return {
                "output": f"Error executing task: {str(e)}",
                "steps": [],
                "tool_calls": [],
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def execute_workflow(
        self,
        task: str,
        agents_config: List[Dict[str, Any]],
        workflow_type: str = "sequential",
        max_iterations: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a multi-agent workflow.
        
        Args:
            task: Overall task description
            agents_config: Configuration for each agent
            workflow_type: Type of workflow
            max_iterations: Maximum iterations
            **kwargs: Additional parameters
            
        Returns:
            Workflow execution result
        """
        start_time = time.time()
        self._workflows_executed += 1
        
        try:
            # Create agents from config
            agents = []
            for agent_config in agents_config:
                agent_type = agent_config.get("agent_type", "query")
                agent_class = self._agent_classes.get(agent_type)
                
                if not agent_class:
                    logger.warning(f"Unknown agent type: {agent_type}, skipping")
                    continue
                
                agent = agent_class(
                    llm_provider=self.llm,
                    prompt_manager=self.prompt_manager
                )
                
                agents.append({
                    "agent": agent,
                    "name": agent_config.get("name", f"{agent_type}_{len(agents)}"),
                    "task": agent_config.get("task", task),
                    "tools": self._get_tools(agent_config.get("tools", [])),
                    "depends_on": agent_config.get("depends_on", [])
                })
            
            if not agents:
                raise ValueError("No valid agents configured")
            
            # Execute workflow based on type
            iterations = max_iterations or self.max_iterations
            
            if workflow_type == "sequential":
                result = await self._execute_sequential(agents, task, iterations)
            elif workflow_type == "parallel":
                result = await self._execute_parallel(agents, task, iterations)
            elif workflow_type == "adaptive":
                result = await self._execute_adaptive(agents, task, iterations)
            else:
                result = await self._execute_sequential(agents, task, iterations)
            
            # Process result
            total_time = time.time() - start_time
            self._total_time += total_time
            
            return {
                "final_result": result.get("final_result", ""),
                "agent_results": result.get("agent_results", []),
                "iterations": result.get("iterations", 1),
                "total_time": total_time,
                "workflow_type": workflow_type,
                "success": result.get("success", True)
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            
            return {
                "final_result": f"Workflow execution failed: {str(e)}",
                "agent_results": [],
                "iterations": 0,
                "total_time": time.time() - start_time,
                "workflow_type": workflow_type,
                "success": False,
                "error": str(e)
            }
    
    async def _execute_sequential(
        self,
        agents: List[Dict[str, Any]],
        task: str,
        max_iterations: int
    ) -> Dict[str, Any]:
        """Execute agents sequentially, passing results between them."""
        agent_results = []
        context = {"task": task}
        
        for iteration in range(max_iterations):
            iteration_results = []
            
            for agent_config in agents:
                agent = agent_config["agent"]
                agent_task = agent_config.get("task", task)
                
                # Execute agent
                result = await agent.execute(
                    task=agent_task,
                    context=context
                )
                
                iteration_results.append({
                    "agent_name": agent_config["name"],
                    "output": result.get("output", ""),
                    "success": result.get("success", True)
                })
                
                # Update context with this agent's output
                context[agent_config["name"]] = result.get("output", "")
            
            agent_results.extend(iteration_results)
            
            # Check if we should continue
            if self._should_stop(iteration_results):
                break
        
        # Synthesize final result
        final_result = await self._synthesize_results(agent_results, task)
        
        return {
            "final_result": final_result,
            "agent_results": agent_results,
            "iterations": min(max_iterations, len(agent_results) // len(agents)),
            "success": True
        }
    
    async def _execute_parallel(
        self,
        agents: List[Dict[str, Any]],
        task: str,
        max_iterations: int
    ) -> Dict[str, Any]:
        """Execute agents in parallel and merge results."""
        all_results = []
        
        for iteration in range(max_iterations):
            # Execute all agents in parallel
            tasks = []
            for agent_config in agents:
                agent = agent_config["agent"]
                agent_task = agent_config.get("task", task)
                
                tasks.append(agent.execute(
                    task=agent_task,
                    context={"task": task}
                ))
            
            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            iteration_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    iteration_results.append({
                        "agent_name": agents[i]["name"],
                        "output": f"Error: {str(result)}",
                        "success": False
                    })
                else:
                    iteration_results.append({
                        "agent_name": agents[i]["name"],
                        "output": result.get("output", ""),
                        "success": result.get("success", True)
                    })
            
            all_results.extend(iteration_results)
            
            # Check if we should continue
            if self._should_stop(iteration_results):
                break
        
        # Merge and synthesize results
        final_result = await self._synthesize_results(all_results, task)
        
        return {
            "final_result": final_result,
            "agent_results": all_results,
            "iterations": min(max_iterations, len(all_results) // len(agents)),
            "success": True
        }
    
    async def _execute_adaptive(
        self,
        agents: List[Dict[str, Any]],
        task: str,
        max_iterations: int
    ) -> Dict[str, Any]:
        """Execute agents adaptively, choosing next agent based on results."""
        agent_results = []
        context = {"task": task}
        
        for iteration in range(max_iterations):
            # Determine which agent to run next
            next_agent = await self._select_next_agent(agents, agent_results, task)
            
            if not next_agent:
                break
            
            # Execute selected agent
            result = await next_agent["agent"].execute(
                task=next_agent.get("task", task),
                context=context
            )
            
            agent_results.append({
                "agent_name": next_agent["name"],
                "output": result.get("output", ""),
                "success": result.get("success", True)
            })
            
            # Update context
            context[next_agent["name"]] = result.get("output", "")
            
            # Check if task is complete
            if self._is_task_complete(agent_results, task):
                break
        
        final_result = await self._synthesize_results(agent_results, task)
        
        return {
            "final_result": final_result,
            "agent_results": agent_results,
            "iterations": len(agent_results),
            "success": True
        }
    
    async def _select_next_agent(
        self,
        agents: List[Dict[str, Any]],
        previous_results: List[Dict[str, Any]],
        task: str
    ) -> Optional[Dict[str, Any]]:
        """Select the next agent to execute adaptively."""
        # Simple selection: choose agent that hasn't run yet
        executed_agents = {r["agent_name"] for r in previous_results}
        
        for agent_config in agents:
            if agent_config["name"] not in executed_agents:
                return agent_config
        
        # All agents executed
        return None
    
    def _should_stop(self, results: List[Dict[str, Any]]) -> bool:
        """Determine if workflow should stop."""
        # Stop if all agents failed
        if all(not r.get("success", False) for r in results):
            return True
        
        # Stop if any agent indicates completion
        for result in results:
            output = result.get("output", "").lower()
            if "task complete" in output or "final answer" in output:
                return True
        
        return False
    
    def _is_task_complete(
        self,
        results: List[Dict[str, Any]],
        task: str
    ) -> bool:
        """Check if the overall task is complete."""
        if not results:
            return False
        
        # Check last result for completion indicators
        last_output = results[-1].get("output", "").lower()
        completion_indicators = [
            "final answer",
            "task complete",
            "here is the final",
            "in conclusion"
        ]
        
        return any(indicator in last_output for indicator in completion_indicators)
    
    async def _synthesize_results(
        self,
        agent_results: List[Dict[str, Any]],
        task: str
    ) -> str:
        """Synthesize results from multiple agents."""
        if not agent_results:
            return "No results generated."
        
        if len(agent_results) == 1:
            return agent_results[0].get("output", "")
        
        # Combine results
        combined = "\n\n".join([
            f"[{r['agent_name']}]: {r.get('output', '')}"
            for r in agent_results
        ])
        
        # Synthesize using LLM
        messages = [
            Message(role="system", content="Synthesize the following agent outputs into a coherent final answer."),
            Message(role="user", content=f"Task: {task}\n\nAgent outputs:\n{combined}\n\nSynthesized answer:")
        ]
        
        response = await self.llm.generate(messages=messages)
        return response.content
    
    def _get_tools(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """Get tool configurations by name."""
        available_tools = {
            "retrieve_documents": {
                "name": "retrieve_documents",
                "description": "Retrieve relevant documents from knowledge base",
                "parameters": {"query": "string", "top_k": "integer"}
            },
            "web_search": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {"query": "string", "num_results": "integer"}
            },
            "verify_claim": {
                "name": "verify_claim",
                "description": "Verify a factual claim",
                "parameters": {"claim": "string"}
            },
            "summarize": {
                "name": "summarize",
                "description": "Summarize text content",
                "parameters": {"text": "string", "max_length": "integer"}
            }
        }
        
        return [available_tools[name] for name in tool_names if name in available_tools]
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List available agent types."""
        return [
            {
                "type": "query",
                "description": "Query understanding and decomposition",
                "capabilities": ["query_analysis", "query_decomposition", "intent_detection"]
            },
            {
                "type": "research",
                "description": "Information gathering and research",
                "capabilities": ["document_retrieval", "web_search", "information_extraction"]
            },
            {
                "type": "synthesis",
                "description": "Information synthesis and summarization",
                "capabilities": ["text_synthesis", "summarization", "answer_generation"]
            },
            {
                "type": "validation",
                "description": "Fact-checking and validation",
                "capabilities": ["fact_checking", "source_verification", "confidence_scoring"]
            }
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent service statistics."""
        avg_time = self._total_time / max(self._tasks_executed + self._workflows_executed, 1)
        
        return {
            "tasks_executed": self._tasks_executed,
            "workflows_executed": self._workflows_executed,
            "avg_execution_time": avg_time,
            "max_iterations": self.max_iterations,
            "timeout": self.timeout
        }