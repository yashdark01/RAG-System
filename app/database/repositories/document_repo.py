"""
Document repository for document-related database operations.
"""

from typing import List, Optional
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from app.database.repositories.base import BaseRepository
from app.database.models import Document, DocumentChunk, DocumentStatus


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document model operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(Document, session)
    
    async def get_with_chunks(self, document_id: uuid.UUID) -> Optional[Document]:
        """
        Get document with all its chunks.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document with chunks loaded
        """
        from sqlalchemy.orm import selectinload
        
        query = (
            select(Document)
            .where(Document.id == document_id)
            .options(selectinload(Document.chunks))
        )
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_status(
        self,
        status: DocumentStatus,
        limit: int = 100
    ) -> List[Document]:
        """
        Get documents by status.
        
        Args:
            status: Document status
            limit: Maximum results
            
        Returns:
            List of documents
        """
        query = (
            select(Document)
            .where(Document.status == status)
            .order_by(Document.created_at.desc())
            .limit(limit)
        )
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_by_filename(self, filename: str) -> Optional[Document]:
        """
        Get document by filename.
        
        Args:
            filename: Document filename
            
        Returns:
            Document or None
        """
        query = select(Document).where(Document.filename == filename)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_recent_documents(
        self,
        limit: int = 10,
        status: Optional[DocumentStatus] = None
    ) -> List[Document]:
        """
        Get recently uploaded documents.
        
        Args:
            limit: Maximum results
            status: Optional status filter
            
        Returns:
            List of documents
        """
        query = select(Document)
        
        if status:
            query = query.where(Document.status == status)
        
        query = query.order_by(Document.created_at.desc()).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_total_stats(self) -> dict:
        """
        Get document statistics.
        
        Returns:
            Dictionary with statistics
        """
        # Total documents
        total_query = select(func.count(Document.id))
        total_result = await self.session.execute(total_query)
        total_docs = total_result.scalar()
        
        # Documents by status
        status_query = (
            select(Document.status, func.count(Document.id))
            .group_by(Document.status)
        )
        status_result = await self.session.execute(status_query)
        status_counts = {status: count for status, count in status_result}
        
        # Total chunks
        chunks_query = select(func.count(DocumentChunk.id))
        chunks_result = await self.session.execute(chunks_query)
        total_chunks = chunks_result.scalar()
        
        # Total tokens
        tokens_query = select(func.sum(Document.token_count))
        tokens_result = await self.session.execute(tokens_query)
        total_tokens = tokens_result.scalar() or 0
        
        return {
            "total_documents": total_docs,
            "by_status": status_counts,
            "total_chunks": total_chunks,
            "total_tokens": total_tokens
        }
    
    async def search_documents(
        self,
        query: str,
        file_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Document]:
        """
        Search documents by filename or metadata.
        
        Args:
            query: Search query
            file_type: Optional file type filter
            limit: Maximum results
            
        Returns:
            List of matching documents
        """
        conditions = [
            Document.filename.ilike(f"%{query}%"),
            Document.title.ilike(f"%{query}%"),
            Document.description.ilike(f"%{query}%")
        ]
        
        query = select(Document).where(or_(*conditions))
        
        if file_type:
            query = query.where(Document.file_type == file_type)
        
        query = query.order_by(Document.updated_at.desc()).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())