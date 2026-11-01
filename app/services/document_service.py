"""
Document processing and management service.
"""

from typing import Dict, Any, List, Optional, BinaryIO
from pathlib import Path
import uuid
import asyncio
from datetime import datetime

from app.config import settings
from app.logging.logger import get_logger
from app.rag.document_processor import DocumentProcessor
from app.rag.vector_store import VectorStoreService
from app.rag.embeddings import EmbeddingService

logger = get_logger(__name__)


class DocumentService:
    """
    Service for document ingestion, processing, and management.
    """
    
    def __init__(
        self,
        processor: DocumentProcessor,
        vector_store: VectorStoreService,
        embedding_service: Optional[EmbeddingService] = None
    ):
        """
        Initialize document service.
        
        Args:
            processor: Document processor
            vector_store: Vector store for indexing
            embedding_service: Optional embedding service
        """
        self.processor = processor
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        # Processing queue for large documents
        self._processing_queue = asyncio.Queue()
        self._processing_task = None
        
        # Metrics
        self._documents_processed = 0
        self._chunks_indexed = 0
        self._errors = 0
        
        logger.info("DocumentService initialized")
    
    async def process_and_index(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a document and index it in the vector store.
        
        Args:
            content: Document content as bytes
            filename: Original filename
            metadata: Additional metadata
            chunk_size: Override chunk size
            chunk_overlap: Override chunk overlap
            **kwargs: Additional processing options
            
        Returns:
            Processing result with document ID and chunk count
        """
        try:
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Prepare base metadata
            base_metadata = {
                "source": filename,
                "document_id": document_id,
                "uploaded_at": datetime.utcnow().isoformat(),
                "file_type": self._detect_file_type(filename)
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            # Process document into chunks
            if chunk_size or chunk_overlap:
                # Create custom processor for this document
                custom_processor = DocumentProcessor(
                    chunk_size=chunk_size or settings.CHUNK_SIZE,
                    chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP
                )
                chunks = await custom_processor.process_document(
                    content=content,
                    filename=filename,
                    metadata=base_metadata,
                    **kwargs
                )
            else:
                chunks = await self.processor.process_document(
                    content=content,
                    filename=filename,
                    metadata=base_metadata,
                    **kwargs
                )
            
            # Index chunks in vector store
            indexed_count = 0
            for chunk in chunks:
                success = await self.vector_store.add_document(
                    document_id=chunk["id"],
                    text=chunk["text"],
                    metadata=chunk["metadata"]
                )
                if success:
                    indexed_count += 1
            
            # Update metrics
            self._documents_processed += 1
            self._chunks_indexed += indexed_count
            
            logger.info(
                f"Document '{filename}' processed: {len(chunks)} chunks, "
                f"{indexed_count} indexed"
            )
            
            return {
                "document_id": document_id,
                "filename": filename,
                "status": "processed",
                "chunks_count": len(chunks),
                "indexed_count": indexed_count,
                "metadata": base_metadata
            }
            
        except Exception as e:
            self._errors += 1
            logger.error(f"Document processing failed: {e}")
            raise
    
    async def process_and_index_batch(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process and index multiple documents.
        
        Args:
            documents: List of document specs with content, filename, metadata
            **kwargs: Additional processing options
            
        Returns:
            List of processing results
        """
        tasks = [
            self.process_and_index(
                content=doc["content"],
                filename=doc["filename"],
                metadata=doc.get("metadata"),
                **kwargs
            )
            for doc in documents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch document {i} failed: {result}")
                processed_results.append({
                    "filename": documents[i]["filename"],
                    "status": "error",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        logger.info(f"Batch processed {len(documents)} documents")
        return processed_results
    
    async def get_document(
        self,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document data or None
        """
        return await self.vector_store.get_document(document_id)
    
    async def delete_document(
        self,
        document_id: str
    ) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if deleted
        """
        return await self.vector_store.delete_document(document_id)
    
    async def update_document(
        self,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None
    ) -> bool:
        """
        Update document metadata or content.
        
        Args:
            document_id: Document ID
            metadata: Updated metadata
            text: Updated text content
            
        Returns:
            True if updated
        """
        return await self.vector_store.update_document(
            document_id=document_id,
            metadata=metadata,
            text=text
        )
    
    async def search_documents(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents by query.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of matching documents
        """
        results = await self.vector_store.search(
            query=query,
            top_k=top_k,
            filter_metadata=filters
        )
        
        return results
    
    async def list_documents(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List all documents with pagination.
        
        Args:
            limit: Number of documents per page
            offset: Pagination offset
            
        Returns:
            List of documents
        """
        # Get all document IDs
        all_ids = list(self.vector_store._metadata_index.keys())
        
        # Apply pagination
        page_ids = all_ids[offset:offset + limit]
        
        # Get documents
        documents = []
        for doc_id in page_ids:
            doc = await self.vector_store.get_document(doc_id)
            if doc:
                documents.append({
                    "document_id": doc["id"],
                    "metadata": doc.get("metadata", {}),
                    "created_at": doc.get("created_at"),
                    "index_id": doc.get("index_id")
                })
        
        return documents
    
    async def get_document_count(self) -> int:
        """Get total number of documents."""
        return await self.vector_store.count_documents()
    
    async def reindex_document(
        self,
        document_id: str
    ) -> bool:
        """
        Reindex a document (delete and reprocess).
        
        Args:
            document_id: Document ID
            
        Returns:
            True if reindexed
        """
        doc = await self.vector_store.get_document(document_id)
        if not doc:
            return False
        
        # Delete existing
        await self.vector_store.delete_document(document_id)
        
        # Re-add with same metadata
        return await self.vector_store.add_document(
            document_id=document_id,
            text=doc["text"],
            metadata=doc.get("metadata", {})
        )
    
    def _detect_file_type(self, filename: str) -> str:
        """Detect file type from extension."""
        ext = Path(filename).suffix.lower()
        
        type_map = {
            ".txt": "text",
            ".pdf": "pdf",
            ".md": "markdown",
            ".py": "python",
            ".js": "javascript",
            ".json": "json",
            ".csv": "csv",
            ".html": "html",
            ".xml": "xml"
        }
        
        return type_map.get(ext, "unknown")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document service statistics."""
        return {
            "documents_processed": self._documents_processed,
            "chunks_indexed": self._chunks_indexed,
            "errors": self._errors,
            "processor_stats": self.processor.get_stats(),
            "vector_store_stats": self.vector_store.get_stats()
        }