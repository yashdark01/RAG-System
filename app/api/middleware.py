"""
API middleware components.
Includes CORS, logging, rate limiting, and authentication middleware.
"""

import time
import uuid
from typing import Callable, Dict, Optional
from collections import defaultdict
import asyncio

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.config import settings
from app.logging.logger import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all requests and responses."""
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with logging."""
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response {request_id}: {response.status_code} "
                f"took {process_time:.3f}s"
            )
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "request_id": request_id,
                    "detail": str(e) if settings.DEBUG else None
                }
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using sliding window algorithm."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.requests: Dict[str, list] = defaultdict(list)
        self.rate_limit = settings.RATE_LIMIT_REQUESTS
        self.window = settings.RATE_LIMIT_WINDOW_SECONDS
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with rate limiting."""
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        # Get client identifier
        client_id = request.client.host if request.client else "unknown"
        
        # Clean old entries
        now = time.time()
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window
        ]
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.rate_limit:
            logger.warning(f"Rate limit exceeded for {client_id}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "detail": f"Rate limit of {self.rate_limit} requests per {self.window}s exceeded"
                }
            )
        
        # Record request
        self.requests[client_id].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.rate_limit - len(self.requests[client_id])
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + self.window))
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with error handling."""
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "type": type(e).__name__,
                    "detail": str(e) if settings.DEBUG else "An unexpected error occurred"
                }
            )


def setup_middleware(app: FastAPI):
    """Configure all middleware for the application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_CREDENTIALS,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )
    
    # Custom middleware (order matters - last added = first executed)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("Middleware configured successfully")