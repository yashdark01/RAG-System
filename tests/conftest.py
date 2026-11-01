"""
Pytest configuration and shared fixtures for the RAG & Multi-Agent test suite.
Provides fast offline mocks for embeddings, LLM chat models, and databases.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
from typing import List
from unittest.mock import Mock, AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_models import FakeListChatModel
from fastapi.testclient import TestClient

# Ensure test environment settings
os.environ["APP_ENV"] = "development"
os.environ["DEBUG"] = "false"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["RATE_LIMIT_ENABLED"] = "false"


class MockEmbeddings:
    """Fast, deterministic in-memory mock embeddings without downloading models."""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            # Deterministic pseudo-vector based on hash
            np.random.seed(abs(hash(text)) % (2**32))
            vec = np.random.randn(self.dimension)
            norm = np.linalg.norm(vec)
            vec = (vec / norm).tolist() if norm > 0 else vec.tolist()
            results.append(vec)
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@pytest.fixture(autouse=True)
def mock_huggingface_embeddings():
    """Globally mock HuggingFaceEmbeddings, SentenceTransformer, and CrossEncoder to avoid downloads."""
    with patch("langchain_huggingface.HuggingFaceEmbeddings") as mock_hf, \
         patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("sentence_transformers.CrossEncoder") as mock_ce:
        
        mock_instance = Mock()
        mock_instance.embed_documents.side_effect = lambda texts: MockEmbeddings().embed_documents(texts)
        mock_instance.embed_query.side_effect = lambda q: MockEmbeddings().embed_query(q)
        mock_hf.return_value = mock_instance

        mock_st_instance = Mock()
        mock_st_instance.encode.side_effect = lambda texts: np.array(MockEmbeddings().embed_documents(texts if isinstance(texts, list) else [texts]))
        mock_st.return_value = mock_st_instance

        mock_ce_instance = Mock()
        mock_ce_instance.predict.side_effect = lambda pairs: np.array([0.95 - (i * 0.05) for i in range(len(pairs))])
        mock_ce.return_value = mock_ce_instance

        yield mock_instance


@pytest.fixture
def fake_chat_model():
    """Return a fast offline FakeListChatModel."""
    return FakeListChatModel(responses=[
        "This is an automated test response from the assistant.",
        "Secondary test response with facts and citations.",
        "Third test response."
    ])


@pytest.fixture
def temp_dir():
    """Create a temporary directory cleaned up after test."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_app():
    """Create a FastAPI app instance for testing."""
    from app.main import create_app
    return create_app()


@pytest.fixture
def client(mock_app):
    """FastAPI TestClient with lifespan management."""
    with TestClient(mock_app) as test_client:
        yield test_client
