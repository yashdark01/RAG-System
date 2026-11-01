"""
Caching service using Redis for distributed caching.
"""

from typing import Any, Optional, Union
import json
import hashlib
from datetime import timedelta

from app.config import settings
from app.logging.logger import get_logger

logger = get_logger(__name__)


class CacheService:
    """
    Caching service with Redis backend.
    Supports TTL, namespaces, and serialization.
    """
    
    def __init__(
        self,
        redis_client,
        default_ttl: int = 3600,
        namespace: str = "rag"
    ):
        """
        Initialize cache service.
        
        Args:
            redis_client: Redis client instance
            default_ttl: Default time-to-live in seconds
            namespace: Cache key namespace
        """
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.namespace = namespace
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._sets = 0
        
        logger.info(f"CacheService initialized: namespace={namespace}, ttl={default_ttl}s")
    
    def _make_key(self, key: str) -> str:
        """
        Create namespaced cache key.
        
        Args:
            key: Original key
            
        Returns:
            Namespaced key
        """
        # Hash long keys
        if len(key) > 100:
            key = hashlib.md5(key.encode()).hexdigest()[:16]
        
        return f"{self.namespace}:{key}"
    
    async def get(self, key: str, default: Any = None) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        try:
            namespaced_key = self._make_key(key)
            value = await self.redis.get(namespaced_key)
            
            if value is None:
                self._misses += 1
                return default
            
            self._hits += 1
            
            # Deserialize
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value.decode('utf-8') if isinstance(value, bytes) else value
                
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            
        Returns:
            True if set successfully
        """
        try:
            namespaced_key = self._make_key(key)
            
            # Serialize
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value, default=str)
            
            # Set with TTL
            expire = ttl or self.default_ttl
            await self.redis.set(namespaced_key, value, ex=expire)
            
            self._sets += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        try:
            namespaced_key = self._make_key(key)
            await self.redis.delete(namespaced_key)
            return True
            
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if exists
        """
        try:
            namespaced_key = self._make_key(key)
            return await self.redis.exists(namespaced_key) > 0
            
        except Exception as e:
            logger.error(f"Cache exists check failed: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter in cache.
        
        Args:
            key: Cache key
            amount: Increment amount
            
        Returns:
            New value
        """
        try:
            namespaced_key = self._make_key(key)
            return await self.redis.incrby(namespaced_key, amount)
            
        except Exception as e:
            logger.error(f"Cache increment failed: {e}")
            return 0
    
    async def get_or_set(
        self,
        key: str,
        factory,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get from cache or compute and set.
        
        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Time-to-live
            
        Returns:
            Cached or computed value
        """
        value = await self.get(key)
        
        if value is not None:
            return value
        
        # Compute value
        if callable(factory):
            value = await factory() if hasattr(factory, '__call__') else factory
        else:
            value = factory
        
        # Cache it
        await self.set(key, value, ttl)
        
        return value
    
    async def clear_namespace(self) -> int:
        """
        Clear all keys in current namespace.
        
        Returns:
            Number of keys deleted
        """
        try:
            pattern = f"{self.namespace}:*"
            count = 0
            
            # Scan and delete keys
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(
                    cursor, match=pattern, count=100
                )
                
                if keys:
                    await self.redis.delete(*keys)
                    count += len(keys)
                
                if cursor == 0:
                    break
            
            logger.info(f"Cleared {count} keys from namespace '{self.namespace}'")
            return count
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return 0
    
    async def get_many(self, keys: list) -> dict:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs
        """
        try:
            namespaced_keys = [self._make_key(k) for k in keys]
            values = await self.redis.mget(namespaced_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        result[key] = value.decode('utf-8') if isinstance(value, bytes) else value
            
            return result
            
        except Exception as e:
            logger.error(f"Cache get_many failed: {e}")
            return {}
    
    async def set_many(
        self,
        mapping: dict,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set multiple values in cache.
        
        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time-to-live
            
        Returns:
            True if successful
        """
        try:
            expire = ttl or self.default_ttl
            
            # Use pipeline for batch operation
            pipeline = self.redis.pipeline()
            
            for key, value in mapping.items():
                namespaced_key = self._make_key(key)
                
                if not isinstance(value, (str, bytes)):
                    value = json.dumps(value, default=str)
                
                pipeline.set(namespaced_key, value, ex=expire)
            
            await pipeline.execute()
            return True
            
        except Exception as e:
            logger.error(f"Cache set_many failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "sets": self._sets,
            "hit_rate": hit_rate,
            "namespace": self.namespace,
            "default_ttl": self.default_ttl
        }