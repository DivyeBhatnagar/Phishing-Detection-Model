"""
PhishGuard AI - Request Logging Middleware
==========================================
Logs every API request/response including timing,
status code, and user info to MongoDB + loguru.
"""

from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests with timing and status."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Compute response time
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Log to console/file
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} "
            f"[{elapsed_ms}ms] "
            f"client={request.client.host if request.client else 'unknown'}"
        )

        # Async DB log (fire-and-forget, non-blocking)
        try:
            from backend.db.models import LogDocument
            log_doc = LogDocument(
                endpoint=str(request.url.path),
                method=request.method,
                status_code=response.status_code,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                response_time_ms=elapsed_ms,
            )
            # Use create_task to avoid blocking
            import asyncio
            asyncio.create_task(log_doc.insert())
        except Exception:
            pass  # Non-critical, don't break the request

        # Add timing header
        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
        return response
