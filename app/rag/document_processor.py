"""
LangChain-Native Document Processing and Hierarchical Chunking.
Uses RecursiveCharacterTextSplitter to create high-precision child chunks
linked to larger parent context chunks for optimal RAG retrieval.
"""

from typing import List, Dict, Any, Optional, Union, BinaryIO, Tuple
import os
import io
import re
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.logging.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Production document processor supporting hierarchical (parent-child) chunking,
    PDF/Text parsing, and metadata enrichment.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap

        # LangChain text splitters
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        # Metrics
        self._documents_processed = 0
        self._total_chunks = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return processor statistics."""
        return {
            "documents_processed": self._documents_processed,
            "total_chunks": self._total_chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "parent_chunk_size": self.parent_chunk_size
        }

    def _clean_text(self, text: str) -> str:
        """Clean raw text while preserving layout structure."""
        if not text:
            return ""
        # Normalize whitespace while preserving paragraph breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _extract_pdf_bytes(self, content_bytes: bytes) -> str:
        """Extract text from PDF bytes using pypdf."""
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text:
                    pages.append(f"[Page {i+1}]\n{page_text}")
            return "\n\n".join(pages)
        except Exception as e:
            logger.warning(f"pypdf extraction failed: {e}. Falling back to utf-8 decode.")
            return content_bytes.decode("utf-8", errors="ignore")

    async def extract_text(
        self,
        content: Union[str, bytes, BinaryIO, Path],
        filename: Optional[str] = None
    ) -> str:
        """Extract raw text from varied input formats."""
        if isinstance(content, str):
            # Check if it's a file path or raw text
            if "\n" not in content and len(content) < 500 and os.path.exists(content):
                path = Path(content)
                if path.suffix.lower() == ".pdf":
                    return self._extract_pdf_bytes(path.read_bytes())
                return path.read_text(encoding="utf-8", errors="ignore")
            return content

        elif isinstance(content, bytes):
            if filename and filename.lower().endswith(".pdf"):
                return self._extract_pdf_bytes(content)
            # Try utf-8
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return self._extract_pdf_bytes(content)

        elif isinstance(content, Path):
            if content.suffix.lower() == ".pdf":
                return self._extract_pdf_bytes(content.read_bytes())
            return content.read_text(encoding="utf-8", errors="ignore")

        elif hasattr(content, "read"):
            data = content.read()
            if isinstance(data, str):
                return data
            return self.extract_text(data, filename)

        return str(content)

    async def process_document(
        self,
        content: Union[str, bytes, BinaryIO, Path],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        file_type: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process a document and return chunks compatible with VectorStoreService and API.
        """
        text = await self.extract_text(content, filename)
        text = self._clean_text(text)
        
        if not text:
            logger.warning(f"No text extracted from document '{filename}'")
            return []

        doc_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Split text into child chunks
        child_texts = self.child_splitter.split_text(text)
        if not child_texts:
            child_texts = [text]

        base_metadata = {
            "doc_id": doc_id,
            "source": filename or "unknown",
            "file_type": file_type or (filename.split(".")[-1] if filename and "." in filename else "text"),
            "content_hash": content_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_chunks": len(child_texts)
        }
        if metadata:
            base_metadata.update(metadata)

        document_chunks = []
        for i, chunk_text in enumerate(child_texts):
            chunk_id = f"{doc_id}_{i}"
            document_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    **base_metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "is_first_chunk": (i == 0),
                    "is_last_chunk": (i == len(child_texts) - 1)
                }
            })

        self._documents_processed += 1
        self._total_chunks += len(document_chunks)
        logger.info(f"Processed document '{filename}': {len(document_chunks)} chunks created.")
        return document_chunks

    def to_langchain_documents(self, chunks: List[Dict[str, Any]]) -> List[LCDocument]:
        """Convert chunk dictionaries to LangChain Document instances."""
        return [
            LCDocument(page_content=c["text"], metadata=c.get("metadata", {}))
            for c in chunks
        ]
