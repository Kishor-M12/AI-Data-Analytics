"""
cache.py — Disk-based caching for embeddings and LLM responses.
Uses diskcache for persistent, TTL-aware caching.
"""
import hashlib
import json
from functools import wraps
from typing import Any, Callable, Optional
import diskcache
from config import get_settings

settings = get_settings()
_cache = diskcache.Cache(settings.cache_dir)


def make_key(*args, **kwargs) -> str:
    """Generate a deterministic cache key from arguments."""
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def cached(ttl: Optional[int] = None):
    """
    Decorator to cache function results on disk.

    Args:
        ttl: Time-to-live in seconds. Defaults to settings.cache_ttl.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key = f"{func.__qualname__}:{make_key(*args, **kwargs)}"
            result = _cache.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            _cache.set(key, result, expire=ttl or settings.cache_ttl)
            return result
        return wrapper
    return decorator


def clear_cache():
    """Clear all cached entries."""
    _cache.clear()


def cache_stats() -> dict:
    """Return cache size info."""
    return {"size": len(_cache), "volume_bytes": _cache.volume()}
