"""
Formatting utilities for consistent output.
"""

from typing import Any, Optional
from datetime import datetime, timedelta
import json


def format_response(
    data: Any,
    status: str = "success",
    message: Optional[str] = None,
    meta: Optional[dict] = None
) -> dict:
    """
    Format a standard API response.
    
    Args:
        data: Response data
        status: Response status
        message: Optional message
        meta: Optional metadata
        
    Returns:
        Formatted response dictionary
    """
    response = {
        "status": status,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if message:
        response["message"] = message
    
    if meta:
        response["meta"] = meta
    
    return response


def format_error_response(
    error: str,
    error_code: Optional[str] = None,
    details: Optional[dict] = None,
    status_code: int = 500
) -> dict:
    """
    Format an error response.
    
    Args:
        error: Error message
        error_code: Error code
        details: Additional details
        status_code: HTTP status code
        
    Returns:
        Formatted error dictionary
    """
    return {
        "status": "error",
        "error": {
            "code": error_code or "UNKNOWN_ERROR",
            "message": error,
            "details": details or {},
            "status_code": status_code
        },
        "timestamp": datetime.utcnow().isoformat()
    }


def truncate_text(
    text: str,
    max_length: int = 200,
    suffix: str = "..."
) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to append
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to break at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]
    
    return truncated + suffix


def format_timestamp(
    dt: Optional[datetime] = None,
    fmt: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """
    Format a datetime to string.
    
    Args:
        dt: Datetime object (default: now)
        fmt: Format string
        
    Returns:
        Formatted timestamp
    """
    dt = dt or datetime.utcnow()
    return dt.strftime(fmt)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


def format_number(num: int) -> str:
    """
    Format large numbers with commas.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted number string
    """
    return f"{num:,}"


def prettify_json(data: Any) -> str:
    """
    Pretty print JSON data.
    
    Args:
        data: Data to format
        
    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


def format_list(items: list, separator: str = ", ", last_separator: str = " and ") -> str:
    """
    Format a list as a human-readable string.
    
    Args:
        items: List of items
        separator: Separator between items
        last_separator: Separator before last item
        
    Returns:
        Formatted string
    """
    if not items:
        return ""
    
    if len(items) == 1:
        return str(items[0])
    
    if len(items) == 2:
        return f"{items[0]}{last_separator}{items[1]}"
    
    return separator.join(str(item) for item in items[:-1]) + last_separator + str(items[-1])