"""
Input validation utilities.
"""

import re
from typing import Optional, List
from pathlib import Path


def validate_query(query: str, min_length: int = 1, max_length: int = 2000) -> bool:
    """
    Validate a search query.
    
    Args:
        query: Query string
        min_length: Minimum query length
        max_length: Maximum query length
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If query is invalid
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if len(query) < min_length:
        raise ValueError(f"Query must be at least {min_length} characters")
    
    if len(query) > max_length:
        raise ValueError(f"Query must be at most {max_length} characters")
    
    # Check for potentially harmful content
    if re.search(r'[<>{}]', query):
        raise ValueError("Query contains invalid characters")
    
    return True


def validate_file_type(filename: str, allowed_types: Optional[List[str]] = None) -> bool:
    """
    Validate file type by extension.
    
    Args:
        filename: File name
        allowed_types: List of allowed extensions
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If file type is invalid
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    ext = Path(filename).suffix.lower()
    
    default_allowed = [
        '.txt', '.pdf', '.md', '.markdown',
        '.py', '.js', '.java', '.cpp', '.c',
        '.json', '.csv', '.html', '.htm',
        '.xml', '.yaml', '.yml', '.log'
    ]
    
    allowed = allowed_types or default_allowed
    
    if ext not in allowed:
        raise ValueError(
            f"File type '{ext}' is not supported. "
            f"Allowed types: {', '.join(allowed)}"
        )
    
    return True


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL string
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If URL is invalid
    """
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$',
        re.IGNORECASE
    )
    
    if not url_pattern.match(url):
        raise ValueError(f"Invalid URL format: {url}")
    
    return True


def sanitize_input(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input by removing potentially harmful content.
    
    Args:
        text: Input text
        max_length: Maximum length to truncate
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove script tags and content
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove SQL injection patterns
    text = re.sub(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)', '', text, flags=re.IGNORECASE)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()


def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email address
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If email is invalid
    """
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    if not email_pattern.match(email):
        raise ValueError(f"Invalid email format: {email}")
    
    return True


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format.
    
    Args:
        api_key: API key string
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If API key is invalid
    """
    if not api_key or len(api_key) < 16:
        raise ValueError("API key must be at least 16 characters")
    
    # Check for common patterns
    if re.match(r'^[a-zA-Z0-9_-]+$', api_key):
        return True
    
    raise ValueError("API key contains invalid characters")