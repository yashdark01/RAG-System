"""
RAG-specific exceptions.
"""

from app.exceptions.base import RAGSystemException


class RetrievalException(RAGSystemException):
    """Exception for retrieval errors."""
    
    def __init__(
        self,
        message: str = "Document retrieval failed",
        query: str = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if query:
            details['query'] = query[:100]
        super().__init__(
            message=message,
            error_code="RETRIEVAL_ERROR",
            status_code=500,
            details=details,
            **kwargs
        )


class EmbeddingException(RAGSystemException):
    """Exception for embedding generation errors."""
    
    def __init__(
        self,
        message: str = "Embedding generation failed",
        model: str = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if model:
            details['model'] = model
        super().__init__(
            message=message,
            error_code="EMBEDDING_ERROR",
            status_code=500,
            details=details,
            **kwargs
        )


class VectorStoreException(RAGSystemException):
    """Exception for vector store operations."""
    
    def __init__(
        self,
        message: str = "Vector store operation failed",
        operation: str = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if operation:
            details['operation'] = operation
        super().__init__(
            message=message,
            error_code="VECTOR_STORE_ERROR",
            status_code=500,
            details=details,
            **kwargs
        )


class DocumentProcessingException(RAGSystemException):
    """Exception for document processing errors."""
    
    def __init__(
        self,
        message: str = "Document processing failed",
        filename: str = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if filename:
            details['filename'] = filename
        super().__init__(
            message=message,
            error_code="DOCUMENT_PROCESSING_ERROR",
            status_code=400,
            details=details,
            **kwargs
        )


class RerankingException(RAGSystemException):
    """Exception for reranking errors."""
    
    def __init__(
        self,
        message: str = "Reranking failed",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="RERANKING_ERROR",
            status_code=500,
            **kwargs
        )


class DocumentNotFoundError(RAGSystemException):
    """Exception when document is not found."""
    
    def __init__(
        self,
        document_id: str = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if document_id:
            details['document_id'] = document_id
        super().__init__(
            message=f"Document not found: {document_id}",
            error_code="DOCUMENT_NOT_FOUND",
            status_code=404,
            details=details,
            **kwargs
        )


class InvalidDocumentError(RAGSystemException):
    """Exception for invalid document format or content."""
    
    def __init__(
        self,
        message: str = "Invalid document",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="INVALID_DOCUMENT",
            status_code=400,
            **kwargs
        )


class IndexingException(RAGSystemException):
    """Exception for document indexing errors."""
    
    def __init__(
        self,
        message: str = "Document indexing failed",
        document_id: str = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if document_id:
            details['document_id'] = document_id
        super().__init__(
            message=message,
            error_code="INDEXING_ERROR",
            status_code=500,
            details=details,
            **kwargs
        )