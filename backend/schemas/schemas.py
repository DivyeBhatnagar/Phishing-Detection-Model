"""
PhishGuard AI - Pydantic Request / Response Schemas
====================================================
All API input validation and response serialisation schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Prediction ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Input schema for POST /predict."""

    email_text: str = Field(
        ...,
        min_length=5,
        max_length=50_000,
        description="Full email content (subject + body combined)",
        examples=["Congratulations! You have won a prize. Click here to claim."],
    )
    include_explanation: bool = Field(
        default=False,
        description="Include SHAP feature explanation in response",
    )

    @field_validator("email_text")
    @classmethod
    def no_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("email_text cannot be empty or whitespace")
        return v


class PredictResponse(BaseModel):
    """Output schema for POST /predict."""

    prediction: str          # 'spam' | 'legitimate'
    label: int               # 1 | 0
    confidence: float        # 0.0 – 100.0 (percentage)
    risk_level: str          # 'low' | 'medium' | 'high'
    model_name: str
    threshold: float
    phishing_keywords: List[str] = []
    top_features: Optional[List[Dict]] = None
    processing_time_ms: float
    prediction_id: Optional[str] = None


# ── Auth ────────────────────────────────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class UserLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int      # seconds


# ── Health ──────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str          # 'healthy' | 'degraded' | 'unhealthy'
    version: str
    environment: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_threshold: Optional[float] = None
    database_connected: bool
    uptime_seconds: float
    timestamp: datetime


# ── Metrics ─────────────────────────────────────────────────────────────────────

class ModelMetricsResponse(BaseModel):
    model_name: str
    saved_at: str
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    accuracy: float
    confusion_matrix: Dict[str, int]
    total_predictions: int
    spam_predictions: int
    legitimate_predictions: int


# ── History ─────────────────────────────────────────────────────────────────────

class PredictionHistoryItem(BaseModel):
    id: str
    prediction: str
    confidence: float
    risk_level: str
    email_preview: str
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    items: List[PredictionHistoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int


# ── Training ────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    max_features: int = Field(default=75_000, ge=1_000, le=200_000)
    use_char_ngrams: bool = True
    run_cv: bool = False
    min_recall: float = Field(default=0.95, ge=0.5, le=1.0)


class TrainResponse(BaseModel):
    task_id: Optional[str] = None
    status: str           # 'started' | 'completed' | 'failed'
    message: str
    best_model: Optional[str] = None
    metrics: Optional[Dict] = None


# ── Generic ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class SuccessResponse(BaseModel):
    message: str
    data: Optional[Any] = None
