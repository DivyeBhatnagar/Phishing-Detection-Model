"""
PhishGuard AI - FastAPI Application Entry Point
================================================
Production-ready FastAPI application with:
- JWT authentication
- Rate limiting (slowapi)
- CORS
- Request logging middleware
- Prometheus metrics
- Swagger / ReDoc docs
- Structured startup / shutdown lifecycle
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.api.auth import router as auth_router
from backend.api.monitoring import health_router, history_router, metrics_router
from backend.api.predict import router as predict_router
from backend.api.train import router as train_router
from backend.db.connection import connect_db, disconnect_db
from backend.middleware.logging_middleware import RequestLoggingMiddleware
from backend.services.detector import get_detector
from config.settings import settings
from utils.logger import setup_logging

# ── Startup / Shutdown lifecycle ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan context manager.
    Handles startup (DB connect, model load) and shutdown (cleanup).
    """
    # ── STARTUP ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"  Environment: {settings.APP_ENV}")
    logger.info("=" * 60)

    # Setup logging
    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_dir=settings.LOG_DIR,
    )

    # Connect to MongoDB
    try:
        await connect_db()
    except Exception as exc:
        logger.warning(f"MongoDB connection failed (non-fatal): {exc}")

    # Load ML model
    detector = get_detector()
    if detector.is_loaded:
        logger.info(f"✅ Model ready: {detector.model_name}")
    else:
        logger.warning(
            "⚠️  No trained model found. "
            "POST /api/v1/train to train the model, "
            "or run: python ml_pipeline/pipeline.py"
        )

    logger.info("✅ PhishGuard API started successfully")
    yield

    # ── SHUTDOWN ───────────────────────────────────────────────────────────────
    logger.info("Shutting down PhishGuard API...")
    await disconnect_db()
    logger.info("Goodbye. 👋")


# ── Rate Limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── FastAPI Application ────────────────────────────────────────────────────────
app = FastAPI(
    title="PhishGuard AI",
    description="""
## 🛡️ AI-Powered Phishing Email Detection System

An enterprise-grade ML backend for detecting phishing and spam emails
using advanced NLP and machine learning techniques.

### Features
- 🤖 **Multi-model ML** (LightGBM, XGBoost, RF, LR, Naive Bayes)
- 🧠 **Advanced NLP** (TF-IDF, char n-grams, hand-crafted features)
- 🔍 **SHAP Explainability** for model transparency
- 🔒 **JWT Auth** + API key support
- ⚡ **Async MongoDB** integration
- 📊 **Prometheus metrics** ready

### Quick Start
1. Train the model: `POST /api/v1/train`
2. Predict an email: `POST /api/v1/predict`
3. View history: `GET /api/v1/history`
    """,
    version=settings.APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── Rate Limiting ──────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Response-Time"],
)

# ── GZip Compression ──────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Request Logging Middleware ─────────────────────────────────────────────────
app.add_middleware(RequestLoggingMiddleware)

# ── Prometheus Metrics ─────────────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ── API Routers ────────────────────────────────────────────────────────────────
API_PREFIX = "/api/v1"

app.include_router(health_router, prefix=API_PREFIX)
app.include_router(metrics_router, prefix=API_PREFIX)
app.include_router(history_router, prefix=API_PREFIX)
app.include_router(predict_router, prefix=API_PREFIX)
app.include_router(train_router, prefix=API_PREFIX)
app.include_router(auth_router, prefix=API_PREFIX)


# ── Global Exception Handlers ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler — prevents leaking stack traces in production."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    if settings.is_production:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "code": "INTERNAL_ERROR"},
        )
    # In development, include the exception message
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


# ── Root endpoint ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect hint for root URL."""
    return {
        "message": f"Welcome to {settings.APP_NAME}!",
        "version": settings.APP_VERSION,
        "docs": "/api/docs",
        "health": "/api/v1/health",
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=1 if settings.RELOAD else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=False,  # We handle logging ourselves
    )
