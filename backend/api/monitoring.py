"""
PhishGuard AI - Health, Metrics & History Routers
===================================================
GET /api/v1/health    → Service health check
GET /api/v1/metrics   → Model performance metrics
GET /api/v1/history   → Prediction history (paginated)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from loguru import logger

from backend.core.auth import get_current_user, get_optional_user, require_admin
from backend.db.models import PredictionDocument
from backend.schemas.schemas import (
    HealthResponse,
    ModelMetricsResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
)
from backend.services.detector import get_detector
from config.settings import settings
from ml_pipeline.models.model_store import get_model_metadata

# Track app start time for uptime calculation
_start_time = time.time()


# ── Health Router ──────────────────────────────────────────────────────────────

health_router = APIRouter(tags=["Health"])


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns overall service health including model and DB status.",
)
async def health_check():
    """Public health check endpoint."""
    detector = get_detector()

    # Check DB connectivity
    db_connected = False
    try:
        from backend.db.connection import get_database
        db = get_database()
        await db.command("ping")
        db_connected = True
    except Exception:
        db_connected = False

    overall_status = "healthy"
    if not detector.is_loaded:
        overall_status = "degraded"
    if not db_connected:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        model_loaded=detector.is_loaded,
        model_name=detector.model_name,
        model_threshold=detector.threshold if detector.is_loaded else None,
        database_connected=db_connected,
        uptime_seconds=round(time.time() - _start_time, 2),
        timestamp=datetime.now(tz=timezone.utc),
    )


# ── Metrics Router ─────────────────────────────────────────────────────────────

metrics_router = APIRouter(tags=["Metrics"])


@metrics_router.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    summary="Model performance metrics",
    description="Returns trained model evaluation metrics from the last training run.",
)
async def get_metrics(current_user: dict = Depends(get_current_user)):
    """Get model metrics (requires authentication)."""
    metadata = get_model_metadata(settings.MODEL_PATH)
    if not metadata:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail="No trained model found. Run the training pipeline first.",
        )

    m = metadata.get("metrics", {})
    cm = m.get("confusion_matrix", {})

    # Get prediction counts from DB
    total_predictions = await PredictionDocument.count()
    spam_predictions = await PredictionDocument.find(
        PredictionDocument.prediction == "spam"
    ).count()
    legit_predictions = total_predictions - spam_predictions

    return ModelMetricsResponse(
        model_name=metadata.get("model_name", "unknown"),
        saved_at=metadata.get("saved_at", ""),
        threshold=metadata.get("threshold", 0.5),
        precision=m.get("precision", 0),
        recall=m.get("recall", 0),
        f1=m.get("f1", 0),
        roc_auc=m.get("roc_auc", 0),
        accuracy=m.get("accuracy", 0),
        confusion_matrix=cm,
        total_predictions=total_predictions,
        spam_predictions=spam_predictions,
        legitimate_predictions=legit_predictions,
    )


# ── History Router ─────────────────────────────────────────────────────────────

history_router = APIRouter(tags=["History"])


@history_router.get(
    "/history",
    response_model=PredictionHistoryResponse,
    summary="Prediction history",
    description="Returns paginated prediction history. Admin sees all, users see their own.",
)
async def get_prediction_history(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    prediction: Optional[str] = Query(default=None, description="Filter: spam | legitimate"),
    risk_level: Optional[str] = Query(default=None, description="Filter: low | medium | high"),
    current_user: dict = Depends(get_current_user),
):
    """Get paginated prediction history."""
    is_admin = current_user.get("is_admin", False)
    user_id = current_user.get("sub")

    # Build query
    query = {}
    if not is_admin:
        query["user_id"] = user_id
    if prediction:
        query["prediction"] = prediction
    if risk_level:
        query["risk_level"] = risk_level

    # Get paginated results
    skip = (page - 1) * page_size

    docs = await PredictionDocument.find(query).sort(
        -PredictionDocument.created_at
    ).skip(skip).limit(page_size).to_list()

    total = await PredictionDocument.find(query).count()
    total_pages = (total + page_size - 1) // page_size

    items = [
        PredictionHistoryItem(
            id=str(doc.id),
            prediction=doc.prediction,
            confidence=doc.confidence_percent,
            risk_level=doc.risk_level,
            email_preview=doc.email_preview[:100],
            created_at=doc.created_at,
        )
        for doc in docs
    ]

    return PredictionHistoryResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
