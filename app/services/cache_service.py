import functools
import json
from typing import Any, Callable, Optional
import logging
from app.services.redis_service import redis_client
import hashlib

class CacheService:
    def __init__(self):
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            self.logger.error(f"Error getting from cache: {str(e)}")
            return None

    async def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set a value in cache with expiration."""
        try:
            await self.redis.set(key, json.dumps(value), ex=expire)
            return True
        except Exception as e:
            self.logger.error(f"Error setting cache: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            self.logger.error(f"Error deleting from cache: {str(e)}")
            return False

    async def clear(self) -> bool:
        """Clear all cache."""
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return False

def cache(expire: int = 3600):
    """Cache decorator for functions."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
            key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cache_service = CacheService()
            cached_result = await cache_service.get(key)
            if cached_result is not None:
                return cached_result
            
            # If not in cache, call function and cache result
            result = await func(*args, **kwargs)
            await cache_service.set(key, result, expire)
            return result
        return wrapper
    return decorator 