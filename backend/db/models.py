"""
PhishGuard AI - Database Document Models (Beanie ODM)
======================================================
MongoDB collections:
- predictions  → stores prediction results + history
- users        → user accounts for auth
- logs         → API request/response logs
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from beanie import Document, Indexed
from pydantic import EmailStr, Field


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── Predictions Collection ─────────────────────────────────────────────────────

class PredictionDocument(Document):
    """Stores each email prediction result."""

    email_hash: Indexed(str)              # SHA-256 of email text (dedup key)
    email_preview: str                    # First 200 chars of email
    prediction: str                       # 'spam' | 'legitimate'
    label: int                            # 1 = spam, 0 = legitimate
    confidence: float                     # 0.0 – 1.0
    confidence_percent: float             # 0.0 – 100.0
    risk_level: str                       # 'low' | 'medium' | 'high'
    model_name: str                       # Model used for prediction
    threshold: float                      # Decision threshold
    phishing_keywords: List[str] = []    # Detected phishing keywords
    shap_top_features: List[Dict] = []   # Top SHAP features (if available)
    user_id: Optional[str] = None        # Auth user ID (nullable for anon)
    api_key_used: Optional[str] = None   # Which API key was used
    ip_address: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    processing_time_ms: float = 0.0

    class Settings:
        name = "predictions"
        indexes = [
            "email_hash",
            "prediction",
            "risk_level",
            "created_at",
            "user_id",
        ]


# ── Users Collection ───────────────────────────────────────────────────────────

class UserDocument(Document):
    """User accounts for authentication."""

    username: Indexed(str, unique=True)
    email: Indexed(EmailStr, unique=True)
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    api_key: Optional[Indexed(str)] = None
    prediction_count: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    last_login: Optional[datetime] = None

    class Settings:
        name = "users"
        indexes = ["username", "email", "api_key"]


# ── Logs Collection ────────────────────────────────────────────────────────────

class LogDocument(Document):
    """API request/response logs."""

    endpoint: str
    method: str
    status_code: int
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_body: Optional[Dict[str, Any]] = None
    response_time_ms: float = 0.0
    user_id: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)

    class Settings:
        name = "logs"
        indexes = ["endpoint", "status_code", "created_at"]
