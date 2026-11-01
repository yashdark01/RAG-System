"""
Unit tests for DocumentProcessor and LangChain document chunking.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from langchain_core.documents import Document as LCDocument

from app.rag.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test suite for DocumentProcessor."""

    @pytest.fixture
    def processor(self):
        return DocumentProcessor(chunk_size=100, chunk_overlap=20)

    def test_initialization_and_stats(self, processor):
        """Verify processor initializes with correct attributes and stats."""
        stats = processor.get_stats()
        assert stats["documents_processed"] == 0
        assert stats["total_chunks"] == 0
        assert stats["chunk_size"] == 100
        assert stats["chunk_overlap"] == 20

    def test_clean_text(self, processor):
        """Verify text cleaning normalizes whitespace and line breaks."""
        dirty = "  Hello   world!\r\n\r\nThis  is   a   test.\n\n\n\nNew paragraph.  "
        cleaned = processor._clean_text(dirty)
        assert "\r" not in cleaned
        assert "   " not in cleaned
        assert "\n\n\n" not in cleaned
        assert cleaned.startswith("Hello world!")
        assert "New paragraph." in cleaned

    def test_clean_text_empty(self, processor):
        """Verify cleaning empty or None text returns empty string."""
        assert processor._clean_text("") == ""
        assert processor._clean_text(None) == ""

    @pytest.mark.asyncio
    async def test_extract_text_string(self, processor):
        """Verify extracting text from string."""
        raw_text = "This is a direct string input."
        extracted = await processor.extract_text(raw_text)
        assert extracted == raw_text

    @pytest.mark.asyncio
    async def test_extract_text_bytes(self, processor):
        """Verify extracting text from utf-8 bytes."""
        byte_data = b"UTF-8 encoded document content."
        extracted = await processor.extract_text(byte_data)
        assert extracted == "UTF-8 encoded document content."

    @pytest.mark.asyncio
    async def test_extract_text_pdf_fallback(self, processor):
        """Verify extracting text from PDF bytes falls back safely."""
        pdf_bytes = b"%PDF-1.4 mock content"
        extracted = await processor.extract_text(pdf_bytes, filename="sample.pdf")
        assert isinstance(extracted, str)

    @pytest.mark.asyncio
    async def test_process_document_chunking(self, processor):
        """Verify process_document creates chunks with rich metadata."""
        sample_text = (
            "Artificial intelligence (AI) is intelligence demonstrated by machines. "
            "Unlike natural intelligence displayed by humans, machine intelligence "
            "involves perception, synthesis, and deduction. "
            "Applications include advanced web search engines, recommendation systems, "
            "understanding human speech, and self-driving cars."
        )
        chunks = await processor.process_document(
            content=sample_text,
            filename="ai_overview.txt",
            metadata={"category": "technology", "author": "Alice"}
        )

        assert len(chunks) > 0
        first_chunk = chunks[0]
        assert "id" in first_chunk
        assert "text" in first_chunk
        assert "metadata" in first_chunk
        meta = first_chunk["metadata"]
        assert meta["source"] == "ai_overview.txt"
        assert meta["category"] == "technology"
        assert meta["author"] == "Alice"
        assert meta["chunk_index"] == 0
        assert meta["is_first_chunk"] is True
        assert meta["total_chunks"] == len(chunks)

        # Check stats update
        stats = processor.get_stats()
        assert stats["documents_processed"] == 1
        assert stats["total_chunks"] == len(chunks)

    @pytest.mark.asyncio
    async def test_process_empty_document(self, processor):
        """Verify processing an empty document returns an empty list."""
        chunks = await processor.process_document(content="", filename="empty.txt")
        assert chunks == []

    def test_to_langchain_documents(self, processor):
        """Verify conversion to LangChain Document instances."""
        chunks = [
            {"id": "c1", "text": "LangChain integration test chunk 1", "metadata": {"source": "test.txt"}},
            {"id": "c2", "text": "LangChain integration test chunk 2", "metadata": {"source": "test.txt"}}
        ]
        lc_docs = processor.to_langchain_documents(chunks)
        assert len(lc_docs) == 2
        assert isinstance(lc_docs[0], LCDocument)
        assert lc_docs[0].page_content == "LangChain integration test chunk 1"
        assert lc_docs[0].metadata["source"] == "test.txt"
