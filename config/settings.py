"""
PhishGuard AI - Application Configuration
==========================================
Centralized configuration using Pydantic Settings.
All values are read from environment variables / .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    APP_NAME: str = "PhishGuard AI"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "change-me-in-production-use-openssl-rand-hex-32"
    ALLOWED_HOSTS: List[str] = ["*"]
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    # ── Server ─────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False

    # ── MongoDB ────────────────────────────────────────────────────────────────
    MONGO_URI: str = "mongodb://admin:password@localhost:27017"
    MONGO_DB_NAME: str = "phishguard"
    MONGO_USERNAME: str = "admin"
    MONGO_PASSWORD: str = "password"
    MONGO_AUTH_SOURCE: str = "admin"

    # ── Redis / Celery ─────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── JWT ────────────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = "jwt-secret-change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── API Keys ───────────────────────────────────────────────────────────────
    API_KEY_HEADER: str = "X-API-Key"
    ADMIN_API_KEY: str = "admin-api-key-change-me"

    # ── ML Model ───────────────────────────────────────────────────────────────
    MODEL_PATH: Path = Path("models/saved")
    DEFAULT_MODEL: str = "lightgbm"
    CONFIDENCE_THRESHOLD: float = 0.5
    HIGH_RISK_THRESHOLD: float = 0.8

    # ── Rate Limiting ──────────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60

    # ── NLTK ───────────────────────────────────────────────────────────────────
    NLTK_DATA_PATH: str = "./nltk_data"

    # ── Logging ────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = Path("./logs")
    LOG_ROTATION: str = "500 MB"
    LOG_RETENTION: str = "30 days"

    # ── Scheduler ──────────────────────────────────────────────────────────────
    SCHEDULER_TIMEZONE: str = "UTC"
    AUTO_RETRAIN_ENABLED: bool = False
    AUTO_RETRAIN_CRON: str = "0 2 * * 0"

    # ── Dataset ────────────────────────────────────────────────────────────────
    DATASET_PATH: Path = Path("./datasets")
    PRIMARY_DATASET: str = "phishing_email.csv"

    @field_validator("MODEL_PATH", "LOG_DIR", "DATASET_PATH", mode="before")
    @classmethod
    def create_dirs(cls, v: str | Path) -> Path:
        """Ensure required directories exist at startup."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton pattern)."""
    return Settings()


# Convenience alias
settings = get_settings()
