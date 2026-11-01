"""
LangChain-Native LLM Provider Abstraction.
Provides unified access to Google Gemini, Groq, OpenAI, Anthropic, and Ollama
using native LangChain chat models with automatic fallbacks and token streaming.
"""

from typing import List, Dict, Any, Optional, AsyncIterator, Union
from dataclasses import dataclass, field
import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
from app.config import settings
from app.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """Chat message structure compatible with LangChain."""
    role: str  # system, user, assistant, tool
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_langchain(self) -> BaseMessage:
        """Convert to LangChain message object."""
        if self.role == "system":
            return SystemMessage(content=self.content)
        elif self.role in ("user", "human"):
            return HumanMessage(content=self.content)
        elif self.role in ("assistant", "ai"):
            return AIMessage(content=self.content)
        elif self.role in ("tool", "function"):
            return ToolMessage(content=self.content, tool_call_id=self.tool_call_id or "default")
        return HumanMessage(content=self.content)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id")
        )


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str = "default"
    provider: str = "langchain"
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_used(self) -> int:
        return sum(self.usage.values()) if self.usage else 0


@dataclass
class StreamingChunk:
    """Streaming response chunk."""
    content: str
    finish_reason: Optional[str] = None
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    streaming: bool = True
) -> BaseChatModel:
    """
    Factory creating native LangChain chat models with automatic cross-provider fallbacks.
    Supported providers: gemini / google, groq, openai, anthropic, ollama, fake/mock.
    """
    provider = (provider or settings.LLM_PROVIDER).lower()
    temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE

    if provider in ("fake", "mock"):
        from langchain_community.chat_models import FakeListChatModel
        return FakeListChatModel(responses=["This is an automated test response from the assistant."])
    
    primary: Optional[BaseChatModel] = None
    
    # 1. Google Gemini
    if provider in ("gemini", "google"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = model or settings.LLM_MODEL or "gemini-1.5-flash"
            primary = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                streaming=streaming
            )
            logger.info(f"Initialized Google Gemini chat model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize ChatGoogleGenerativeAI: {e}")

    # 2. Groq
    elif provider == "groq":
        try:
            from langchain_groq import ChatGroq
            model_name = model or settings.LLM_MODEL or "llama3-70b-8192"
            primary = ChatGroq(
                model_name=model_name,
                temperature=temperature,
                groq_api_key=settings.GROQ_API_KEY,
                streaming=streaming
            )
            logger.info(f"Initialized Groq chat model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize ChatGroq: {e}")

    # 3. OpenAI
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            model_name = model or settings.LLM_MODEL or "gpt-4o-mini"
            primary = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=settings.OPENAI_API_KEY,
                streaming=streaming
            )
            logger.info(f"Initialized OpenAI chat model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize ChatOpenAI: {e}")

    # 4. Anthropic
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            model_name = model or settings.LLM_MODEL or "claude-3-5-sonnet-20241022"
            primary = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                api_key=settings.ANTHROPIC_API_KEY,
                streaming=streaming
            )
            logger.info(f"Initialized Anthropic chat model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize ChatAnthropic: {e}")

    # 5. Local Ollama
    elif provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            model_name = model or settings.LLM_MODEL or "llama3"
            primary = ChatOllama(
                model=model_name,
                temperature=temperature
            )
            logger.info(f"Initialized Ollama chat model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize ChatOllama: {e}")

    # Build fallbacks if primary failed or for runtime resilience
    fallbacks: List[BaseChatModel] = []
    
    # Try adding Gemini fallback if not primary
    if provider not in ("gemini", "google") and settings.GOOGLE_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            fallbacks.append(ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                streaming=streaming
            ))
        except Exception:
            pass

    # Try adding Groq fallback if not primary
    if provider != "groq" and settings.GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            fallbacks.append(ChatGroq(
                model_name="llama3-70b-8192",
                temperature=temperature,
                groq_api_key=settings.GROQ_API_KEY,
                streaming=streaming
            ))
        except Exception:
            pass

    # Try adding OpenAI fallback if not primary
    if provider != "openai" and settings.OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            fallbacks.append(ChatOpenAI(
                model="gpt-4o-mini",
                temperature=temperature,
                api_key=settings.OPENAI_API_KEY,
                streaming=streaming
            ))
        except Exception:
            pass

    if primary is None:
        if fallbacks:
            logger.warning("Primary provider initialization failed; using available fallback model")
            primary = fallbacks.pop(0)
        else:
            # Fallback to local stub if no keys available
            from langchain_community.chat_models import FakeListChatModel
            logger.warning("No valid LLM credentials configured. Using FakeListChatModel for local offline development.")
            primary = FakeListChatModel(responses=["This is a mock offline LLM response. Please configure API keys in .env."])

    if fallbacks:
        return primary.with_fallbacks(fallbacks)
    return primary


class LLMProvider:
    """
    Unified LLM provider powered by native LangChain chat models.
    Supports streaming, fallbacks, tool calling, and backward-compatible methods.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        self.provider_name = provider or settings.LLM_PROVIDER
        self.model_name = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        # Native LangChain BaseChatModel instance
        self._chat_model = create_chat_model(
            provider=self.provider_name,
            model=self.model_name,
            temperature=self.temperature
        )

    @property
    def model(self) -> BaseChatModel:
        """Access the underlying native LangChain chat model."""
        return self._chat_model

    def _convert_messages(self, messages: List[Union[Dict[str, Any], Message, BaseMessage]]) -> List[BaseMessage]:
        """Convert input messages to LangChain BaseMessage objects."""
        lc_messages: List[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                lc_messages.append(msg)
            elif isinstance(msg, Message):
                lc_messages.append(msg.to_langchain())
            elif isinstance(msg, dict):
                lc_messages.append(Message.from_dict(msg).to_langchain())
            elif isinstance(msg, str):
                lc_messages.append(HumanMessage(content=msg))
        return lc_messages

    async def generate(
        self,
        messages: List[Union[Dict[str, Any], Message, BaseMessage]],
        **kwargs
    ) -> LLMResponse:
        """Generate response asynchronously using native LangChain model."""
        lc_messages = self._convert_messages(messages)
        try:
            ai_msg = await self._chat_model.ainvoke(lc_messages, **kwargs)
            content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
            
            tool_calls = []
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                tool_calls = ai_msg.tool_calls

            usage = {}
            if hasattr(ai_msg, "response_metadata") and "token_usage" in ai_msg.response_metadata:
                usage = ai_msg.response_metadata["token_usage"]

            return LLMResponse(
                content=content,
                model=self.model_name,
                provider=self.provider_name,
                usage=usage,
                tool_calls=tool_calls
            )
        except Exception as e:
            logger.error(f"LangChain generation failed: {e}")
            raise

    async def stream(
        self,
        messages: List[Union[Dict[str, Any], Message, BaseMessage]],
        **kwargs
    ) -> AsyncIterator[StreamingChunk]:
        """Stream response tokens asynchronously using LangChain astream."""
        lc_messages = self._convert_messages(messages)
        try:
            async for chunk in self._chat_model.astream(lc_messages, **kwargs):
                text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                if text:
                    yield StreamingChunk(content=text)
        except Exception as e:
            logger.error(f"LangChain streaming failed: {e}")
            raise

    async def check_health(self) -> Dict[str, bool]:
        """Health check for LLM provider."""
        try:
            res = await self._chat_model.ainvoke([HumanMessage(content="ping")])
            return {self.provider_name: bool(res.content)}
        except Exception as e:
            logger.warning(f"Health check failed for {self.provider_name}: {e}")
            return {self.provider_name: False}
