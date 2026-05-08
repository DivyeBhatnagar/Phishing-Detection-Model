"""
PhishGuard AI - Training API Router
=====================================
POST /api/v1/train → Trigger model training (admin only)
GET  /api/v1/train/status/{task_id} → Check training status
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from backend.core.auth import require_admin
from backend.schemas.schemas import TrainRequest, TrainResponse
from config.settings import settings

router = APIRouter(prefix="/train", tags=["Training"])


@router.post(
    "",
    response_model=TrainResponse,
    summary="Trigger model retraining",
    description="""
    **Admin only.** Kicks off an asynchronous model training pipeline
    using Celery. Returns a task ID to poll for progress.
    """,
)
async def trigger_training(
    body: TrainRequest,
    admin: dict = Depends(require_admin),
):
    """Start background model training."""
    try:
        from backend.services.celery_worker import train_model_task

        task = train_model_task.delay(
            dataset_dir=str(settings.DATASET_PATH),
            model_dir=str(settings.MODEL_PATH),
            max_features=body.max_features,
            run_cv=body.run_cv,
        )

        logger.info(f"Training task submitted: {task.id}")

        return TrainResponse(
            task_id=task.id,
            status="started",
            message=(
                f"Training started. Poll /api/v1/train/status/{task.id} "
                "for progress."
            ),
        )

    except Exception as exc:
        logger.error(f"Failed to submit training task: {exc}")
        # Fallback: synchronous training for systems without Redis
        logger.info("Falling back to synchronous training...")
        try:
            from ml_pipeline.pipeline import run_training_pipeline

            results = run_training_pipeline(
                dataset_dir=str(settings.DATASET_PATH),
                model_save_dir=str(settings.MODEL_PATH),
                max_features=body.max_features,
                run_cv=body.run_cv,
                min_recall=body.min_recall,
            )

            # Reload model
            from backend.services.detector import get_detector
            get_detector().reload()

            return TrainResponse(
                status="completed",
                message="Training completed (synchronous fallback).",
                best_model=results["best_model"],
                metrics=results["test_metrics"],
            )
        except Exception as train_exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Training failed: {train_exc}",
            )


@router.get(
    "/status/{task_id}",
    summary="Check training task status",
)
async def get_training_status(
    task_id: str,
    admin: dict = Depends(require_admin),
):
    """Poll the status of an async training task."""
    try:
        from backend.services.celery_worker import celery_app
        task = celery_app.AsyncResult(task_id)

        return {
            "task_id": task_id,
            "status": task.status,
            "info": task.info if task.status in ["PROGRESS", "SUCCESS"] else None,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve task status: {exc}",
        )
