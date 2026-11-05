"""
Base exception class for the RAG system.
"""

from typing import Optional, Dict, Any


class RAGSystemException(Exception):
    """
    Base exception for all RAG system errors.
    Includes error codes and metadata for better error handling.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize exception.
        
        Args:
            message: Error message
            error_code: Machine-readable error code
            status_code: HTTP status code
            details: Additional error details
            original_error: Original exception if wrapped
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.status_code = status_code
        self.details = details or {}
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API responses.
        
        Returns:
            Dictionary representation
        """
        error_dict = {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code
        }
        
        if self.details:
            error_dict["details"] = self.details
        
        if self.original_error and hasattr(self.original_error, 'message'):
            error_dict["original_error"] = str(self.original_error)
        
        return error_dict
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"