"""
PhishGuard AI - Utility Helpers
================================
General-purpose utilities: hashing, text sanitisation, timing, etc.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# ── Text Sanitisation ──────────────────────────────────────────────────────────

def sanitise_text(text: str, max_length: int = 50_000) -> str:
    """
    Basic input sanitisation for email text.
    - Strip leading/trailing whitespace
    - Limit maximum length
    - Remove null bytes
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    text = text.replace("\x00", "")        # Remove null bytes
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length]
    return text


def is_suspicious_input(text: str) -> bool:
    """
    Detect potential injection attempts in email text.
    Returns True if input looks suspicious.
    """
    patterns = [
        r"<script",               # XSS
        r"javascript:",           # JS injection
        r"\bDROP\b.*\bTABLE\b",  # SQL injection
        r"\$where",               # MongoDB injection
        r"\.\./",                  # Path traversal
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# ── Hashing ────────────────────────────────────────────────────────────────────

def hash_text(text: str) -> str:
    """Return SHA-256 hash of input text (used for deduplication)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Timing ─────────────────────────────────────────────────────────────────────

def timeit(func: F) -> F:
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        from utils.logger import app_logger
        app_logger.debug(f"{func.__qualname__} took {elapsed:.2f}ms")
        return result
    return wrapper  # type: ignore[return-value]


# ── Risk Classification ────────────────────────────────────────────────────────

def classify_risk(confidence: float, threshold_high: float = 0.8) -> str:
    """
    Map confidence score to a risk level string.

    Args:
        confidence: Probability of being phishing (0.0 – 1.0).
        threshold_high: Above this → 'high', else 'medium'.

    Returns:
        'low' | 'medium' | 'high'
    """
    if confidence < 0.4:
        return "low"
    elif confidence < threshold_high:
        return "medium"
    else:
        return "high"


# ── DateTime Helpers ───────────────────────────────────────────────────────────

def utcnow() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return utcnow().isoformat()


# ── Pagination ─────────────────────────────────────────────────────────────────

def paginate(
    items: list,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    """Return a slice of `items` with pagination metadata."""
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }
