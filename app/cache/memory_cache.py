"""In-memory cache implementation."""
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import asyncio
from app.logging.logger import get_logger

logger = get_logger(__name__)

class MemoryCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            entry = self._cache[key]
            if entry["expires_at"] > datetime.utcnow():
                return entry["value"]
            else:
                del self._cache[key]
        return None
        
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache."""
        if len(self._cache) >= self._max_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["expires_at"])
            del self._cache[oldest_key]
            
        self._cache[key] = {
            "value": value,
            "expires_at": datetime.utcnow() + timedelta(seconds=ttl)
        }
        
    async def delete(self, key: str):
        """Delete from cache."""
        self._cache.pop(key, None)
        
    async def clear(self):
        """Clear all cache."""
        self._cache.clear()