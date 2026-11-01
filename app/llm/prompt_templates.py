"""
LangChain-Native Prompt Template Management.
Provides pre-built ChatPromptTemplate instances with MessagesPlaceholder support
for RAG chains and AgentExecutor tool-calling pipelines.
"""

from typing import Dict, Any, Optional, List, Union
from enum import Enum

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate as LCPromptTemplate
)
from app.logging.logger import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------------------
# Core LangChain Chat Prompt Templates
# -------------------------------------------------------------------------

# 1. Standard RAG QA Prompt
RAG_SYSTEM_INSTRUCTION = (
    "You are an expert AI knowledge assistant.\n"
    "Use the following authoritative retrieved context to accurately answer the user's question.\n"
    "Guidelines:\n"
    "- Ground your answer strictly in the provided context.\n"
    "- Cite the source document/chunk when referencing facts (e.g., [Source: filename]).\n"
    "- If the context does not contain sufficient information, state that clearly rather than hallucinating.\n"
    "- Be comprehensive, objective, and well-structured.\n\n"
    "Context:\n{context}"
)

rag_chat_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_INSTRUCTION),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}")
])


# 2. Tool-Calling Agent Prompt (Required keys: input, chat_history, agent_scratchpad)
AGENT_SYSTEM_INSTRUCTION = (
    "You are an advanced autonomous research and synthesis agent.\n"
    "You have access to specialized tools including internal knowledge base retrieval and web search.\n"
    "Always plan your actions, use tools when needed to verify facts, and synthesize grounded answers."
)

agent_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", AGENT_SYSTEM_INSTRUCTION),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# 3. Query Decomposition & Expansion Prompt
QUERY_DECOMPOSITION_INSTRUCTION = (
    "You are a query analysis expert. Decompose the following complex user query into 1 to 3 "
    "focused, self-contained sub-queries optimal for vector similarity and keyword search.\n"
    "Return only the sub-queries, one per line.\n\n"
    "User Query: {query}"
)

query_expansion_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", "You decompose and optimize search queries."),
    ("human", QUERY_DECOMPOSITION_INSTRUCTION)
])


# 4. Factual Verification & Claim Validation Prompt
VERIFICATION_INSTRUCTION = (
    "You are a strict fact-checking and validation expert.\n"
    "Evaluate whether the Claim is fully supported by the Reference Context.\n"
    "Output a JSON object with: 'is_supported' (bool), 'confidence' (float 0.0-1.0), and 'explanation' (str).\n\n"
    "Claim: {claim}\n\n"
    "Reference Context:\n{context}"
)

verification_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are a meticulous factual verification engine."),
    ("human", VERIFICATION_INSTRUCTION)
])


# -------------------------------------------------------------------------
# Backward-Compatible PromptManager
# -------------------------------------------------------------------------

class PromptTemplate:
    """Wrapper holding template text and metadata."""
    def __init__(self, template: str, name: str = "", required_variables: Optional[List[str]] = None, **kwargs):
        self.template = template
        self.name = name
        self.required_variables = required_variables or []

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    def to_langchain(self) -> LCPromptTemplate:
        return LCPromptTemplate.from_template(self.template)


class PromptManager:
    """Centralized prompt manager providing templates and LangChain prompts."""

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._init_defaults()

    def _init_defaults(self):
        self._templates["rag_system"] = PromptTemplate(
            name="rag_system",
            template=RAG_SYSTEM_INSTRUCTION,
            required_variables=["context"]
        )
        self._templates["rag_query"] = PromptTemplate(
            name="rag_query",
            template="Question: {query}\n\nAnswer based on context:",
            required_variables=["query"]
        )
        self._templates["agent_reasoning"] = PromptTemplate(
            name="agent_reasoning",
            template="Task: {task}\nTools: {tools}\nContext: {context}",
            required_variables=["task"]
        )

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        return self._templates.get(name)

    def render(self, name: str, **kwargs) -> str:
        tpl = self._templates.get(name)
        if not tpl:
            return ""
        return tpl.format(**kwargs)

    def get_rag_prompt(self) -> ChatPromptTemplate:
        return rag_chat_prompt

    def get_agent_prompt(self) -> ChatPromptTemplate:
        return agent_prompt

    def get_verification_prompt(self) -> ChatPromptTemplate:
        return verification_prompt
