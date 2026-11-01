"""
Database session management.
Handles connection pooling, session lifecycle, and async database operations.
"""

from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
import asyncio

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    AsyncEngine,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from app.config import settings
from app.logging.logger import get_logger

logger = get_logger(__name__)

# Database engine and session factory
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker] = None

# Connection pool configuration
POOL_SIZE = settings.DATABASE_POOL_SIZE
MAX_OVERFLOW = settings.DATABASE_MAX_OVERFLOW
POOL_TIMEOUT = 30
POOL_RECYCLE = 3600  # Recycle connections after 1 hour


async def init_db() -> None:
    """
    Initialize database engine and create tables.
    Should be called during application startup.
    """
    global _engine, _session_factory
    
    try:
        # Create async engine
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DEBUG,
            pool_pre_ping=True,  # Verify connections before use
            poolclass=NullPool,  # Use NullPool for async compatibility
            connect_args={} if "sqlite" in settings.DATABASE_URL else {
                "command_timeout": 60,
                "server_settings": {
                    "application_name": settings.APP_NAME,
                    "timezone": "UTC"
                }
            }
        )
        
        # Create session factory
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False
        )
        
        # Verify connection
        async with _engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        
        # Create tables
        from app.database.models import Base
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info(
            f"Database initialized: pool_size={POOL_SIZE}, "
            f"max_overflow={MAX_OVERFLOW}"
        )
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_db() -> None:
    """
    Close database connections.
    Should be called during application shutdown.
    """
    global _engine, _session_factory
    
    try:
        if _engine:
            await _engine.dispose()
            _engine = None
            _session_factory = None
            logger.info("Database connections closed")
            
    except Exception as e:
        logger.error(f"Database shutdown error: {e}")
        raise


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session.
    Use as a FastAPI dependency for automatic session management.
    
    Yields:
        AsyncSession: Database session
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session as a context manager.
    
    Usage:
        async with get_db_context() as db:
            result = await db.execute(query)
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_health() -> bool:
    """
    Check database health by executing a simple query.
    
    Returns:
        bool: True if database is healthy
    """
    try:
        if _engine is None:
            return False
        
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        
        return True
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def get_db_stats() -> dict:
    """
    Get database connection pool statistics.
    
    Returns:
        dict: Pool statistics
    """
    if _engine is None:
        return {"status": "not_initialized"}
    
    try:
        pool = _engine.pool
        
        return {
            "status": "connected",
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "overflow": pool.overflow(),
            "total": pool.total()
        }
        
    except Exception as e:
        logger.error(f"Failed to get DB stats: {e}")
        return {"status": "error", "message": str(e)}