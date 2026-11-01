"""
General helper utilities.
"""

import uuid
import time
import asyncio
from typing import Any, Callable, TypeVar, List, Optional
from functools import wraps
import hashlib

from app.logging.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


def generate_id() -> str:
    """
    Generate a unique ID.
    
    Returns:
        Unique ID string
    """
    return str(uuid.uuid4())


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def async_timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure async function execution time.
    
    Args:
        func: Async function to measure
        
    Returns:
        Wrapped async function
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


async def retry_async(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exception types to catch
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_error = e
            
            if attempt < max_retries:
                wait_time = delay * (backoff ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    raise last_error


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split a list into chunks.
    
    Args:
        items: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def safe_get(data: dict, *keys: str, default: Any = None) -> Any:
    """
    Safely get a nested dictionary value.
    
    Args:
        data: Dictionary to search
        *keys: Key path
        default: Default value if not found
        
    Returns:
        Value or default
    """
    current = data
    
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
        
        if current is None:
            return default
    
    return current


def hash_content(content: str) -> str:
    """
    Generate a hash for content string.
    
    Args:
        content: Content to hash
        
    Returns:
        SHA256 hash string
    """
    return hashlib.sha256(content.encode()).hexdigest()


def merge_dicts(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def parse_bool(value: Any) -> bool:
    """
    Parse a value to boolean.
    
    Args:
        value: Value to parse
        
    Returns:
        Boolean value
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    
    return bool(value)


def chunks_to_batches(chunks: List[str], batch_size: int) -> List[List[str]]:
    """
    Split text chunks into batches for processing.
    
    Args:
        chunks: List of text chunks
        batch_size: Maximum batch size
        
    Returns:
        List of batches
    """
    batches = []
    current_batch = []
    current_size = 0
    
    for chunk in chunks:
        chunk_size = len(chunk)
        
        if current_size + chunk_size > batch_size and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        
        current_batch.append(chunk)
        current_size += chunk_size
    
    if current_batch:
        batches.append(current_batch)
    
    return batches