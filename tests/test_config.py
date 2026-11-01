"""
Unit tests for configuration and environment settings.
"""

import pytest
from pydantic import ValidationError
from app.config import Settings


class TestConfig:
    """Test suite for application configuration."""

    def test_default_settings(self):
        """Verify default settings initialization."""
        settings = Settings()
        assert settings.APP_NAME == "RAG System"
        assert settings.EMBEDDING_DIMENSION == 384
        assert settings.CHUNK_SIZE > 0
        assert settings.TOP_K_RETRIEVAL > 0

    def test_debug_validator_boolean_and_strings(self):
        """Verify DEBUG field validator parses boolean and string values safely."""
        s1 = Settings(DEBUG=True)
        assert s1.DEBUG is True

        s2 = Settings(DEBUG=False)
        assert s2.DEBUG is False

        s3 = Settings(DEBUG="true")
        assert s3.DEBUG is True

        s4 = Settings(DEBUG="1")
        assert s4.DEBUG is True

        s5 = Settings(DEBUG="release")
        assert s5.DEBUG is False

        s6 = Settings(DEBUG="dev")
        assert s6.DEBUG is True

    def test_gemini_api_key_fallback(self):
        """Verify GEMINI_API_KEY fallback to GOOGLE_API_KEY."""
        s = Settings(GOOGLE_API_KEY="test-google-key", GEMINI_API_KEY=None)
        assert s.GEMINI_API_KEY == "test-google-key"

        s2 = Settings(GOOGLE_API_KEY=None, GEMINI_API_KEY="test-gemini-key")
        assert s2.GEMINI_API_KEY == "test-gemini-key"

    def test_extra_fields_ignored(self):
        """Verify extra unconfigured environment variables are ignored without error."""
        s = Settings(UNKNOWN_NEW_SETTING="arbitrary_value")
        assert not hasattr(s, "UNKNOWN_NEW_SETTING") or s.model_extra.get("UNKNOWN_NEW_SETTING") is not None
