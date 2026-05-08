"""
PhishGuard AI - Logging System
================================
Structured JSON logging using Loguru with rotating file handlers.
Supports: API logs, prediction logs, training logs, error logs.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_dir: str | Path = "./logs",
    rotation: str = "500 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure Loguru with multiple sinks:
    - Console (colourized)
    - General app log (JSON)
    - Prediction-specific log
    - Training-specific log
    - Error-only log
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default Loguru sink
    logger.remove()

    # ── Console sink (development-friendly, colourized) ────────────────────────
    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=True,
        diagnose=True,
    )

    # ── General application log (JSON, all levels) ─────────────────────────────
    logger.add(
        log_dir / "app.log",
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        serialize=True,   # JSON format
        backtrace=True,
        diagnose=False,   # Disable in prod to avoid leaking secrets
        enqueue=True,     # Thread-safe async logging
    )

    # ── Prediction log (filter by custom context) ──────────────────────────────
    logger.add(
        log_dir / "predictions.log",
        level="INFO",
        rotation=rotation,
        retention=retention,
        compression="zip",
        serialize=True,
        filter=lambda record: "prediction" in record["extra"],
        enqueue=True,
    )

    # ── Training log ───────────────────────────────────────────────────────────
    logger.add(
        log_dir / "training.log",
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        compression="zip",
        serialize=True,
        filter=lambda record: "training" in record["extra"],
        enqueue=True,
    )

    # ── Error log (WARNING and above only) ────────────────────────────────────
    logger.add(
        log_dir / "errors.log",
        level="WARNING",
        rotation="100 MB",
        retention=retention,
        compression="zip",
        serialize=True,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )

    logger.info(
        "Logging system initialised",
        log_level=log_level,
        log_dir=str(log_dir),
    )


# ── Convenience loggers with bound context ─────────────────────────────────────

def get_prediction_logger():
    """Return a logger bound to the prediction context."""
    return logger.bind(prediction=True)


def get_training_logger():
    """Return a logger bound to the training context."""
    return logger.bind(training=True)


def get_api_logger():
    """Return a logger bound to the API context."""
    return logger.bind(api=True)


# Expose main logger
app_logger = logger
prediction_logger = get_prediction_logger()
training_logger = get_training_logger()
api_logger = get_api_logger()
