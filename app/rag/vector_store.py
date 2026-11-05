"""
Vector store service for managing document embeddings and similarity search.
Uses FAISS for efficient vector similarity search with metadata storage.
"""

import os
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
from pathlib import Path
import uuid

import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings
from app.logging.logger import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """
    Service for managing vector embeddings storage and retrieval.
    Supports CRUD operations, batch processing, and hybrid search.
    """
    
    def __init__(
        self,
        embedding_service,
        store_path: str = "./data/vector_store",
        index_type: str = "Flat",
        metric: str = "cosine"
    ):
        """
        Initialize the vector store service.
        
        Args:
            embedding_service: Service for generating embeddings
            store_path: Path to persist the vector store
            index_type: FAISS index type (Flat, IVF, HNSW)
            metric: Distance metric (cosine, l2, ip)
        """
        self.embedding_service = embedding_service
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.index_type = index_type
        self.metric = metric
        
        # Initialize indices
        self._index: Optional[faiss.Index] = None
        self._documents: List[Dict[str, Any]] = []
        self._metadata_index: Dict[str, int] = {}
        
        # Performance metrics
        self._search_count = 0
        self._total_search_time = 0.0
        
        # Load existing store if available
        self._load_store()
        
        logger.info(
            f"VectorStoreService initialized with {index_type} index, "
            f"metric={metric}, documents={len(self._documents)}"
        )
    
    def _create_index(self, dimension: int) -> faiss.Index:
        """
        Create a FAISS index based on configuration.
        
        Args:
            dimension: Embedding dimension
            
        Returns:
            FAISS index
        """
        if self.metric == "cosine":
            # For cosine similarity, use inner product on normalized vectors
            index = faiss.IndexFlatIP(dimension)
        elif self.metric == "l2":
            index = faiss.IndexFlatL2(dimension)
        else:
            index = faiss.IndexFlatIP(dimension)
        
        # Convert to IndexIDMap to support custom IDs
        return faiss.IndexIDMap(index)
    
    def _get_index_path(self) -> Path:
        """Get path for index file."""
        return self.store_path / "faiss.index"
    
    def _get_metadata_path(self) -> Path:
        """Get path for metadata file."""
        return self.store_path / "metadata.json"
    
    def _get_documents_path(self) -> Path:
        """Get path for documents file."""
        return self.store_path / "documents.pkl"
    
    def _save_store(self):
        """Persist vector store to disk."""
        try:
            # Save FAISS index
            if self._index is not None and self._index.ntotal > 0:
                faiss.write_index(self._index, str(self._get_index_path()))
            
            # Save documents
            with open(self._get_documents_path(), 'wb') as f:
                pickle.dump(self._documents, f)
            
            # Save metadata index
            with open(self._get_metadata_path(), 'w') as f:
                json.dump(self._metadata_index, f)
            
            logger.info(f"Vector store saved: {len(self._documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def _load_store(self):
        """Load vector store from disk."""
        try:
            index_path = self._get_index_path()
            docs_path = self._get_documents_path()
            meta_path = self._get_metadata_path()
            
            if all(p.exists() for p in [index_path, docs_path, meta_path]):
                # Load FAISS index
                self._index = faiss.read_index(str(index_path))
                
                # Load documents
                with open(docs_path, 'rb') as f:
                    self._documents = pickle.load(f)
                
                # Load metadata
                with open(meta_path, 'r') as f:
                    self._metadata_index = json.load(f)
                
                logger.info(f"Vector store loaded: {len(self._documents)} documents")
            else:
                logger.info("No existing vector store found, starting fresh")
                
        except Exception as e:
            logger.warning(f"Failed to load vector store: {e}. Starting fresh.")
            self._index = None
            self._documents = []
            self._metadata_index = {}
    
    async def add_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Add a single document to the vector store.
        
        Args:
            document_id: Unique document identifier
            text: Document text content
            metadata: Additional document metadata
            embedding: Pre-computed embedding (optional)
            
        Returns:
            True if successful
        """
        try:
            # Generate embedding if not provided
            if embedding is None:
                embedding = await self.embedding_service.embed_document(text)
            
            # Ensure embedding is 2D array
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Initialize index if needed
            if self._index is None:
                dimension = embedding.shape[1]
                self._index = self._create_index(dimension)
            
            # Add to FAISS index
            idx = len(self._documents)
            self._index.add_with_ids(
                embedding.astype(np.float32),
                np.array([idx], dtype=np.int64)
            )
            
            # Store document and metadata
            doc_data = {
                "id": document_id,
                "text": text,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "index_id": idx
            }
            self._documents.append(doc_data)
            self._metadata_index[document_id] = idx
            
            # Persist periodically
            if len(self._documents) % 100 == 0:
                self._save_store()
            
            logger.debug(f"Added document {document_id} at index {idx}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {document_id}: {e}")
            return False
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        texts: Optional[List[str]] = None,
        embeddings: Optional[np.ndarray] = None
    ) -> int:
        """
        Add multiple documents to the vector store.
        
        Args:
            documents: List of document dicts with 'id', 'text', 'metadata'
            texts: List of texts (alternative to documents)
            embeddings: Pre-computed embeddings
            
        Returns:
            Number of documents added successfully
        """
        if texts:
            documents = [
                {"id": str(uuid.uuid4()), "text": text, "metadata": {}}
                for i, text in enumerate(texts)
            ]
        
        if not documents:
            return 0
        
        try:
            # Extract texts for embedding if embeddings not provided
            texts_to_embed = [doc["text"] for doc in documents]
            
            if embeddings is None:
                embeddings = await self.embedding_service.embed_documents(texts_to_embed)
            
            # Ensure 2D array
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Initialize index if needed
            if self._index is None:
                dimension = embeddings.shape[1]
                self._index = self._create_index(dimension)
            
            # Add to FAISS index
            start_idx = len(self._documents)
            indices = np.arange(start_idx, start_idx + len(documents), dtype=np.int64)
            
            self._index.add_with_ids(
                embeddings.astype(np.float32),
                indices
            )
            
            # Store documents
            for i, doc in enumerate(documents):
                doc_data = {
                    "id": doc.get("id", str(uuid.uuid4())),
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "created_at": datetime.utcnow().isoformat(),
                    "index_id": start_idx + i
                }
                self._documents.append(doc_data)
                self._metadata_index[doc_data["id"]] = start_idx + i
            
            # Persist
            self._save_store()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to add documents batch: {e}")
            return 0
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0,
        return_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using query text.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Metadata filters
            min_similarity: Minimum similarity threshold
            return_embeddings: Whether to include embeddings in results
            
        Returns:
            List of search results with scores and metadata
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_query(query)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search in FAISS
            k = min(top_k * 2, self._index.ntotal)  # Get extra for filtering
            distances, indices = self._index.search(query_embedding, k)
            
            # Process results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1 or idx >= len(self._documents):
                    continue
                
                doc = self._documents[idx]
                score = float(distance)
                
                # Apply minimum similarity filter
                if score < min_similarity:
                    continue
                
                # Apply metadata filters
                if filter_metadata:
                    if not self._matches_metadata(doc["metadata"], filter_metadata):
                        continue
                
                result = {
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score,
                    "rank": len(results) + 1
                }
                
                if return_embeddings and hasattr(self, '_last_embeddings'):
                    result["embedding"] = self._last_embeddings[idx]
                
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            # Update metrics
            self._search_count += 1
            self._total_search_time += asyncio.get_event_loop().time() - start_time
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def search_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using embedding vector.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results
            filter_metadata: Metadata filters
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        
        try:
            # Ensure 2D array
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            embedding = embedding.astype(np.float32)
            
            # Search
            k = min(top_k, self._index.ntotal)
            distances, indices = self._index.search(embedding, k)
            
            # Process results
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1 or idx >= len(self._documents):
                    continue
                
                score = float(distance)
                if score < min_similarity:
                    continue
                
                doc = self._documents[idx]
                
                if filter_metadata and not self._matches_metadata(doc["metadata"], filter_metadata):
                    continue
                
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score,
                    "rank": len(results) + 1
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return []
    
    def _matches_metadata(
        self,
        doc_metadata: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """
        Check if document metadata matches filters.
        
        Args:
            doc_metadata: Document metadata
            filters: Filter conditions
            
        Returns:
            True if all filters match
        """
        for key, value in filters.items():
            if key not in doc_metadata:
                return False
            
            doc_value = doc_metadata[key]
            
            # Handle list values (OR condition)
            if isinstance(value, list):
                if doc_value not in value:
                    return False
            # Handle exact match
            elif doc_value != value:
                return False
        
        return True
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            if document_id not in self._metadata_index:
                logger.warning(f"Document {document_id} not found")
                return False
            
            idx = self._metadata_index[document_id]
            
            # Remove from FAISS index
            # Note: FAISS doesn't support direct deletion, we need to rebuild
            self._documents[idx] = None  # Mark as deleted
            
            # Remove from metadata index
            del self._metadata_index[document_id]
            
            # Rebuild index periodically if too many deletions
            deleted_count = sum(1 for doc in self._documents if doc is None)
            if deleted_count > len(self._documents) * 0.2:  # 20% threshold
                await self._rebuild_index()
            
            logger.info(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def update_document(
        self,
        document_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document in the vector store.
        
        Args:
            document_id: Document ID to update
            text: New text content
            metadata: Updated metadata
            
        Returns:
            True if updated successfully
        """
        try:
            # Delete existing
            await self.delete_document(document_id)
            
            # Get existing doc data
            old_doc = None
            for doc in self._documents:
                if doc and doc["id"] == document_id:
                    old_doc = doc
                    break
            
            if old_doc is None:
                # Add as new document
                return await self.add_document(
                    document_id=document_id,
                    text=text or "",
                    metadata=metadata or {}
                )
            
            # Add updated version
            return await self.add_document(
                document_id=document_id,
                text=text or old_doc["text"],
                metadata=metadata or old_doc["metadata"]
            )
            
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False
    
    async def _rebuild_index(self):
        """Rebuild FAISS index to remove deleted documents."""
        try:
            # Filter out deleted documents
            active_docs = [(i, doc) for i, doc in enumerate(self._documents) if doc is not None]
            
            if not active_docs:
                self._index = None
                self._documents = []
                self._metadata_index = {}
                return
            
            # Re-embed documents
            texts = [doc["text"] for _, doc in active_docs]
            embeddings = await self.embedding_service.embed_documents(texts)
            
            # Create new index
            self._index = self._create_index(embeddings.shape[1])
            
            # Add all active documents
            new_indices = np.arange(len(active_docs), dtype=np.int64)
            self._index.add_with_ids(embeddings.astype(np.float32), new_indices)
            
            # Update documents list and metadata index
            new_documents = []
            new_metadata = {}
            for new_idx, (old_idx, doc) in enumerate(active_docs):
                doc["index_id"] = new_idx
                new_documents.append(doc)
                new_metadata[doc["id"]] = new_idx
            
            self._documents = new_documents
            self._metadata_index = new_metadata
            
            self._save_store()
            
            logger.info(f"Rebuilt index with {len(active_docs)} documents")
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        if document_id not in self._metadata_index:
            return None
        
        idx = self._metadata_index[document_id]
        return self._documents[idx] if idx < len(self._documents) else None
    
    async def get_documents(
        self,
        document_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get multiple documents by IDs.
        
        Args:
            document_ids: List of document IDs
            
        Returns:
            List of document data
        """
        documents = []
        for doc_id in document_ids:
            doc = await self.get_document(doc_id)
            if doc:
                documents.append(doc)
        return documents
    
    async def count_documents(self) -> int:
        """Get total number of documents in store."""
        return len([doc for doc in self._documents if doc is not None])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        active_docs = len([doc for doc in self._documents if doc is not None])
        
        return {
            "total_documents": len(self._documents),
            "active_documents": active_docs,
            "deleted_documents": len(self._documents) - active_docs,
            "index_type": self.index_type,
            "metric": self.metric,
            "dimension": self._index.d if self._index else None,
            "total_vectors": self._index.ntotal if self._index else 0,
            "search_count": self._search_count,
            "avg_search_time": (
                self._total_search_time / self._search_count
                if self._search_count > 0 else 0
            ),
            "store_path": str(self.store_path)
        }
    
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search with similarity scores.
        
        Args:
            query: Search query
            k: Number of results
            filter: Metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        results = await self.search(query, top_k=k, filter_metadata=filter)
        return [(result, result["score"]) for result in results]
    
    def clear(self):
        """Clear all documents from the store."""
        self._index = None
        self._documents = []
        self._metadata_index = {}
        
        # Remove persisted files
        for path in [self._get_index_path(), self._get_documents_path(), self._get_metadata_path()]:
            if path.exists():
                path.unlink()
        
        logger.info("Vector store cleared")

    def to_langchain_documents(self) -> List[Any]:
        """Convert all active stored documents to LangChain Document objects."""
        from langchain_core.documents import Document
        return [
            Document(page_content=doc["text"], metadata=doc.get("metadata", {}))
            for doc in self._documents if not doc.get("metadata", {}).get("is_deleted", False)
        ]