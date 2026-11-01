"""Redis cache implementation."""
from typing import Optional
import redis.asyncio as redis
from app.config import settings
from app.logging.logger import get_logger

logger = get_logger(__name__)

_redis_client: Optional[redis.Redis] = None

async def init_redis() -> None:
    """Initialize Redis connection."""
    global _redis_client
    try:
        _redis_client = redis.Redis(
            host=settings.REDIS_URL.split("://")[1].split(":")[0] if "://" in settings.REDIS_URL else "localhost",
            port=int(settings.REDIS_URL.split(":")[-1]) if ":" in settings.REDIS_URL.split("://")[1] else 6379,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            decode_responses=True
        )
        await _redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed, running without cache: {e}")
        _redis_client = None

async def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")

async def get_redis() -> redis.Redis:
    """Get Redis client."""
    if _redis_client is None:
        await init_redis()
    return _redis_client

class RedisCache:
    """Redis cache wrapper."""
    def __init__(self, redis_client=None):
        self.redis = redis_client
        
    async def get(self, key: str) -> Optional[str]:
        return await self.redis.get(key)
        
    async def set(self, key: str, value: str, ttl: int = 3600):
        await self.redis.set(key, value, ex=ttl)
        
    async def delete(self, key: str):
        await self.redis.delete(key)
        
    async def exists(self, key: str) -> bool:
        return await self.redis.exists(key) > 0