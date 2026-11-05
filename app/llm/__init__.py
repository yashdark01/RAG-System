"""
LLM (Large Language Model) module.
Provides abstraction layer for multiple LLM providers with unified interface.
"""

from app.llm.provider import LLMProvider
from app.llm.prompt_templates import PromptTemplate, PromptManager

__all__ = [
    "LLMProvider",
    "PromptTemplate",
    "PromptManager",
]