"""
Main application entry point.
Configures and starts the FastAPI application.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

from app.config import settings
from app.api.v1.router import api_router
from app.api.middleware import setup_middleware
from app.logging.logger import setup_logging, get_logger
from app.database.session import init_db, close_db
from app.cache.redis_cache import init_redis, close_redis

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.APP_ENV}")
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Initialize Redis
        await init_redis()
        logger.info("Redis initialized")
        
        # Initialize logging
        setup_logging()
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    try:
        # Close database connections
        await close_db()
        logger.info("Database connections closed")
        
        # Close Redis connections
        await close_redis()
        logger.info("Redis connections closed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
    
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Advanced RAG System with Multi-Agent Workflow",
        docs_url="/docs" if settings.APP_ENV != "production" else None,
        redoc_url="/redoc" if settings.APP_ENV != "production" else None,
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Include routers
    app.include_router(
        api_router,
        prefix=settings.API_V1_PREFIX
    )
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=settings.APP_NAME,
            version=settings.APP_VERSION,
            description="Advanced RAG System with Multi-Agent Workflow",
            routes=app.routes,
        )
        
        # Add security schemes if needed
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            }
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV,
            "docs": "/docs",
            "health": f"{settings.API_V1_PREFIX}/health"
        }
    
    # Custom docs endpoint (optional)
    if settings.APP_ENV == "production":
        @app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui():
            return get_swagger_ui_html(
                openapi_url="/openapi.json",
                title=f"{settings.APP_NAME} - API Documentation"
            )
    
    return app


# Create application instance
app = create_app()


def main():
    """Run the application."""
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS if settings.APP_ENV == "production" else 1,
        reload=settings.APP_ENV == "development",
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG
    )


if __name__ == "__main__":
    main()