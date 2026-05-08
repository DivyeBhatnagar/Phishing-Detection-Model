"""
PhishGuard AI - Celery Background Task Worker
==============================================
Background tasks:
- Model training (async, non-blocking)
- Scheduled retraining
- Report generation
"""

from __future__ import annotations

from celery import Celery
from loguru import logger

from config.settings import settings

# ── Celery Application ─────────────────────────────────────────────────────────
celery_app = Celery(
    "phishguard",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,  # 24 hours
)


# ── Tasks ──────────────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="phishguard.train_model",
    max_retries=2,
    default_retry_delay=60,
)
def train_model_task(
    self,
    dataset_dir: str = "./datasets",
    model_dir: str = "./models/saved",
    max_features: int = 75_000,
    run_cv: bool = False,
):
    """
    Celery task: Train the phishing detection model.
    Runs in the background, updates progress via Celery state.
    """
    try:
        self.update_state(state="PROGRESS", meta={"step": "Loading datasets..."})
        logger.info(f"[Task {self.request.id}] Starting training pipeline...")

        from ml_pipeline.pipeline import run_training_pipeline

        self.update_state(state="PROGRESS", meta={"step": "Training models..."})
        results = run_training_pipeline(
            dataset_dir=dataset_dir,
            model_save_dir=model_dir,
            max_features=max_features,
            run_cv=run_cv,
        )

        # Reload the detector singleton
        self.update_state(state="PROGRESS", meta={"step": "Reloading model..."})
        from backend.services.detector import get_detector
        get_detector().reload()

        logger.info(f"[Task {self.request.id}] Training complete.")
        return {
            "status": "completed",
            "best_model": results["best_model"],
            "metrics": results["test_metrics"],
        }

    except Exception as exc:
        logger.error(f"[Task {self.request.id}] Training failed: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(name="phishguard.scheduled_retrain")
def scheduled_retrain_task():
    """Scheduled auto-retraining task (triggered by Celery Beat)."""
    from config.settings import settings
    if not settings.AUTO_RETRAIN_ENABLED:
        return {"status": "skipped", "reason": "Auto-retrain disabled"}

    logger.info("Running scheduled retraining...")
    return train_model_task.delay(
        dataset_dir=str(settings.DATASET_PATH),
        model_dir=str(settings.MODEL_PATH),
    )


# ── Celery Beat Schedule ───────────────────────────────────────────────────────
celery_app.conf.beat_schedule = {
    "scheduled-retrain": {
        "task": "phishguard.scheduled_retrain",
        "schedule": 604800,  # Weekly (7 days in seconds)
    },
}
