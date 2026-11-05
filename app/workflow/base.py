"""Base agent class."""
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from app.llm.provider import LLMProvider
from app.llm.prompt_templates import PromptManager
from app.logging.logger import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[PromptManager] = None,
        name: Optional[str] = None
    ):
        self.llm = llm_provider
        self.prompt_manager = prompt_manager or PromptManager()
        self.name = name or self.__class__.__name__
        
    @abstractmethod
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the agent's task."""
        pass
    
    async def _generate_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate response using LLM."""
        response = await self.llm.generate(messages=messages, **kwargs)
        return response.content