"""
Tests for native LangChain components in the RAG & Agent system.
Validates ChatModel factory, prompt templates, document splitters, and tools.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from app.llm.provider import create_chat_model, LLMProvider, Message
from app.llm.prompt_templates import rag_chat_prompt, agent_prompt, PromptManager
from app.rag.document_processor import DocumentProcessor
from app.workflow.tools.search_tools import get_tavily_tool
from app.workflow.tools.verification_tools import verify_claim_tool


def test_create_chat_model_fallback():
    """Verify create_chat_model returns a valid LangChain model (even without API keys)."""
    model = create_chat_model(provider="fake")
    assert model is not None
    # Test invoke
    res = model.invoke([HumanMessage(content="Hello")])
    assert res.content is not None


def test_message_conversion():
    """Verify Message conversions to LangChain message types."""
    user_msg = Message(role="user", content="Test user query").to_langchain()
    assert isinstance(user_msg, HumanMessage)
    assert user_msg.content == "Test user query"

    ai_msg = Message(role="assistant", content="Test response").to_langchain()
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == "Test response"

    sys_msg = Message(role="system", content="System instruction").to_langchain()
    assert isinstance(sys_msg, SystemMessage)
    assert sys_msg.content == "System instruction"


def test_langchain_prompts():
    """Verify LangChain prompt templates format properly."""
    assert isinstance(rag_chat_prompt, ChatPromptTemplate)
    formatted = rag_chat_prompt.format_messages(
        context="Doc 1 text",
        question="What is this?"
    )
    assert len(formatted) >= 2
    assert "Doc 1 text" in formatted[0].content

    assert isinstance(agent_prompt, ChatPromptTemplate)
    formatted_agent = agent_prompt.format_messages(
        input="Analyze Q3 earnings",
        agent_scratchpad=[]
    )
    assert len(formatted_agent) >= 2


@pytest.mark.asyncio
async def test_langchain_document_processor():
    """Verify DocumentProcessor uses RecursiveCharacterTextSplitter properly."""
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    text = "Paragraph one with some interesting text.\n\nParagraph two with more details on RAG systems."
    chunks = await processor.process_document(content=text, filename="sample.txt")
    
    assert len(chunks) > 0
    assert "id" in chunks[0]
    assert "text" in chunks[0]
    assert "metadata" in chunks[0]
    assert chunks[0]["metadata"]["source"] == "sample.txt"

    # Convert to LangChain documents
    lc_docs = processor.to_langchain_documents(chunks)
    assert len(lc_docs) == len(chunks)
    assert lc_docs[0].page_content == chunks[0]["text"]


def test_tavily_tool_fallback():
    """Verify Tavily tool initialization returns a valid BaseTool."""
    tool = get_tavily_tool()
    assert isinstance(tool, BaseTool)
    assert tool.name in ("fallback_web_search", "tavily_search_results_json")


def test_verification_tool():
    """Verify verify_claim_tool executes correctly."""
    context = "Google Antigravity is an agentic AI coding platform built for advanced developers."
    claim = "Google Antigravity is an agentic platform"
    res = verify_claim_tool.invoke({"claim": claim, "context": context})
    assert "Verified" in res

