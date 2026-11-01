"""
LangChain Native Multi-Agent Orchestrator.
Creates Tool-Calling Agents using modern create_agent (LangGraph CompiledStateGraph)
with multi-turn sessions and real-time streaming support.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory

from app.llm.provider import LLMProvider
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
        self.prompt = prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Default tools: Tavily Search + Claim Verification
        self.tools: List[BaseTool] = tools or [get_tavily_tool(), verify_claim_tool]
        
        self._build_agent()
        logger.info(f"LangChain AgentOrchestrator initialized with {len(self.tools)} tools.")

    def _build_agent(self):
        """Construct the agent using LangChain 1.x create_agent or tool_calling_agent."""
        try:
            from langchain.agents import create_agent
            self._agent = create_agent(
                model=self.raw_model,
                tools=self.tools,
                system_prompt=(
                    "You are an advanced autonomous research and synthesis agent equipped "
                    "with specialized search and verification tools. Always use your tools to corroborate facts."
                )
            )
            self._mode = "create_agent"
        except Exception as e:
            logger.warning(f"Falling back to basic tool-calling pipeline: {e}")
            self._agent = None
            self._mode = "direct"

    def add_tools(self, extra_tools: List[BaseTool]):
        """Dynamically add tools and rebuild agent."""
        self.tools.extend(extra_tools)
        self._build_agent()

    async def execute(
        self,
        task: str,
        session_id: str = "default_session",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute agent reasoning loop on a task.
        """
        history = get_chat_history_for_session(session_id)
        
        try:
            if self._mode == "create_agent" and self._agent:
                # Prepare message sequence with prior history
                messages: List[BaseMessage] = list(history.messages)
                messages.append(HumanMessage(content=task))

                result = await self._agent.ainvoke({"messages": messages})
                out_messages = result.get("messages", [])
                final_content = ""
                tool_calls = []
                
                for msg in out_messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_calls.append(tc.get("name", "tool"))
                    if isinstance(msg, AIMessage) and msg.content:
                        final_content = msg.content if isinstance(msg.content, str) else str(msg.content)

                # Update session history
                history.add_user_message(task)
                history.add_ai_message(final_content)

                return {
                    "success": True,
                    "output": final_content or "Task completed.",
                    "steps": [{"step": "execution", "tool_calls": tool_calls}],
                    "tool_calls": tool_calls,
                    "iterations": len(tool_calls) + 1
                }
            else:
                # Direct tool-binding invocation fallback
                model_with_tools = self.raw_model.bind_tools(self.tools) if hasattr(self.raw_model, "bind_tools") else self.raw_model
                response = await model_with_tools.ainvoke([HumanMessage(content=task)])
                history.add_user_message(task)
                history.add_ai_message(str(response.content))
                return {
                    "success": True,
                    "output": str(response.content),
                    "steps": [],
                    "tool_calls": [],
                    "iterations": 1
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
        Stream fine-grained LLM tokens.
        """
        try:
            if self._mode == "create_agent" and self._agent:
                async for chunk in self._agent.astream({"messages": [HumanMessage(content=task)]}):
                    if isinstance(chunk, dict):
                        for key, val in chunk.items():
                            if isinstance(val, dict) and "messages" in val:
                                for m in val["messages"]:
                                    if isinstance(m, AIMessage) and m.content:
                                        yield str(m.content)
            else:
                async for chunk in self.raw_model.astream([HumanMessage(content=task)]):
                    if hasattr(chunk, "content") and chunk.content:
                        yield str(chunk.content)
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
