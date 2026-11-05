"""
Application constants and enums.
"""

from enum import Enum


class ErrorCodes(str, Enum):
    """Standard error codes for the application."""
    
    # General errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    
    # RAG errors
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"
    DOCUMENT_PROCESSING_ERROR = "DOCUMENT_PROCESSING_ERROR"
    RERANKING_ERROR = "RERANKING_ERROR"
    
    # Agent errors
    AGENT_ERROR = "AGENT_ERROR"
    AGENT_EXECUTION_ERROR = "AGENT_EXECUTION_ERROR"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    AGENT_CONFIGURATION_ERROR = "AGENT_CONFIGURATION_ERROR"
    WORKFLOW_ERROR = "WORKFLOW_ERROR"
    
    # LLM errors
    LLM_ERROR = "LLM_ERROR"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
    
    # Document errors
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    INVALID_DOCUMENT = "INVALID_DOCUMENT"
    DOCUMENT_TOO_LARGE = "DOCUMENT_TOO_LARGE"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"


class ContentTypes(str, Enum):
    """Supported content types."""
    
    TEXT = "text/plain"
    PDF = "application/pdf"
    MARKDOWN = "text/markdown"
    JSON = "application/json"
    HTML = "text/html"
    CSV = "text/csv"
    XML = "application/xml"


class StatusCodes:
    """Common HTTP status codes with descriptions."""
    
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


class FileSizeLimits:
    """File size limits in bytes."""
    
    MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50 MB
    MAX_PDF_SIZE = 100 * 1024 * 1024      # 100 MB
    MAX_TEXT_SIZE = 10 * 1024 * 1024      # 10 MB
    MAX_BATCH_SIZE = 200 * 1024 * 1024    # 200 MB


class TokenLimits:
    """Token limits for different operations."""
    
    MAX_INPUT_TOKENS = 4096
    MAX_OUTPUT_TOKENS = 2048
    MAX_CONTEXT_TOKENS = 8192
    MAX_CHUNK_TOKENS = 512
    MAX_SUMMARY_TOKENS = 256


class CacheKeys:
    """Standard cache key prefixes."""
    
    RAG_QUERY = "rag:query"
    CHAT_RESPONSE = "chat:response"
    EMBEDDING = "embedding"
    DOCUMENT_CHUNK = "doc:chunk"
    AGENT_RESULT = "agent:result"
    
    @classmethod
    def rag_query_key(cls, query: str, top_k: int) -> str:
        """Generate RAG query cache key."""
        return f"{cls.RAG_QUERY}:{hash(query)}:{top_k}"
    
    @classmethod
    def chat_key(cls, conversation_id: str, message: str) -> str:
        """Generate chat cache key."""
        return f"{cls.CHAT_RESPONSE}:{conversation_id}:{hash(message)}"


class MetricsNames:
    """Prometheus-style metric names."""
    
    # Request metrics
    REQUEST_COUNT = "rag_requests_total"
    REQUEST_DURATION = "rag_request_duration_seconds"
    
    # RAG metrics
    RETRIEVAL_COUNT = "rag_retrieval_total"
    RETRIEVAL_DURATION = "rag_retrieval_duration_seconds"
    GENERATION_DURATION = "rag_generation_duration_seconds"
    
    # Cache metrics
    CACHE_HITS = "rag_cache_hits_total"
    CACHE_MISSES = "rag_cache_misses_total"
    
    # Error metrics
    ERROR_COUNT = "rag_errors_total"
    
    # Document metrics
    DOCUMENTS_PROCESSED = "rag_documents_processed_total"
    CHUNKS_CREATED = "rag_chunks_created_total"