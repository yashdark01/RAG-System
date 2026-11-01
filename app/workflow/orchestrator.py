"""
LangChain Native Multi-Agent Orchestrator.
Creates Tool-Calling Agents using create_tool_calling_agent and AgentExecutor,
backed by RunnableWithMessageHistory for multi-turn sessions and real-time streaming.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
import asyncio

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from app.llm.provider import LLMProvider
from app.llm.prompt_templates import agent_prompt
from app.workflow.tools.search_tools import get_tavily_tool
from app.workflow.tools.verification_tools import verify_claim_tool
from app.logging.logger import get_logger

logger = get_logger(__name__)

# In-memory store for session histories
_session_histories: Dict[str, ChatMessageHistory] = {}


def get_chat_history_for_session(session_id: str) -> ChatMessageHistory:
    """Retrieve or create chat history for a session."""
    if session_id not in _session_histories:
        _session_histories[session_id] = ChatMessageHistory()
    return _session_histories[session_id]


class AgentOrchestrator:
    """
    Production-grade LangChain Agent Orchestrator.
    Executes tool-calling reasoning loops with automatic retry, parsing safety,
    session memory, and fine-grained streaming.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Optional[List[BaseTool]] = None,
        prompt: Optional[ChatPromptTemplate] = None,
        max_iterations: int = 5,
        verbose: bool = True
    ):
        self.llm_provider = llm_provider
        self.raw_model: BaseChatModel = llm_provider.model
        self.prompt = prompt or agent_prompt
        self.max_iterations = max_iterations
        
        # Default tools: Tavily Search + Claim Verification
        self.tools: List[BaseTool] = tools or [get_tavily_tool(), verify_claim_tool]
        
        # Build LangChain Tool Calling Agent
        self._agent = create_tool_calling_agent(
            llm=self.raw_model,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Build LangChain Agent Executor
        self._executor = AgentExecutor(
            agent=self._agent,
            tools=self.tools,
            max_iterations=self.max_iterations,
            verbose=verbose,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        # Wrap with conversation history
        self._agent_with_history = RunnableWithMessageHistory(
            self._executor,
            get_session_history=get_chat_history_for_session,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        
        logger.info(f"LangChain AgentOrchestrator initialized with {len(self.tools)} tools.")

    def add_tools(self, extra_tools: List[BaseTool]):
        """Dynamically add tools and rebuild executor."""
        self.tools.extend(extra_tools)
        self._agent = create_tool_calling_agent(
            llm=self.raw_model,
            tools=self.tools,
            prompt=self.prompt
        )
        self._executor = AgentExecutor(
            agent=self._agent,
            tools=self.tools,
            max_iterations=self.max_iterations,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        self._agent_with_history = RunnableWithMessageHistory(
            self._executor,
            get_session_history=get_chat_history_for_session,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    async def execute(
        self,
        task: str,
        session_id: str = "default_session",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute agent reasoning loop on a task.
        """
        config = {"configurable": {"session_id": session_id}}
        try:
            result = await self._agent_with_history.ainvoke(
                {"input": task},
                config=config
            )
            intermediate_steps = result.get("intermediate_steps", [])
            steps_summary = []
            for action, observation in intermediate_steps:
                tool_name = getattr(action, "tool", "unknown_tool")
                steps_summary.append({
                    "tool": tool_name,
                    "input": getattr(action, "tool_input", {}),
                    "output": str(observation)[:200]
                })

            return {
                "success": True,
                "output": result.get("output", ""),
                "steps": steps_summary,
                "tool_calls": [s["tool"] for s in steps_summary],
                "iterations": len(steps_summary) + 1
            }
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "success": False,
                "output": f"Agent task failed: {str(e)}",
                "steps": [],
                "tool_calls": [],
                "error": str(e)
            }

    async def astream_tokens(
        self,
        task: str,
        session_id: str = "default_session"
    ) -> AsyncIterator[str]:
        """
        Stream fine-grained LLM tokens using LangChain astream_events v2.
        """
        config = {"configurable": {"session_id": session_id}}
        try:
            async for event in self._agent_with_history.astream_events(
                {"input": task},
                config=config,
                version="v2"
            ):
                if event.get("event") == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield chunk.content
        except Exception as e:
            logger.error(f"Error during agent token streaming: {e}")
            yield f"\n[Stream Error: {e}]"

    async def orchestrate(
        self,
        task: str,
        agents: Optional[List[Dict[str, Any]]] = None,
        workflow_type: str = "sequential"
    ) -> Dict[str, Any]:
        """Backward-compatible orchestrate method."""
        return await self.execute(task=task)
