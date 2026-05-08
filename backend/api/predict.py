"""
PhishGuard AI - Prediction Router
===================================
POST /api/v1/predict    → Single email prediction
POST /api/v1/predict/batch → Batch email prediction
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger

from backend.core.auth import get_current_user, get_optional_user
from backend.db.models import PredictionDocument
from backend.schemas.schemas import PredictRequest, PredictResponse
from backend.services.detector import get_detector
from utils.helpers import hash_text, is_suspicious_input, sanitise_text

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "",
    response_model=PredictResponse,
    summary="Predict if an email is phishing or legitimate",
    description="""
    Analyse an email and predict whether it is spam/phishing or legitimate.

    **Features:**
    - Advanced NLP preprocessing (HTML stripping, URL normalisation)
    - TF-IDF + hand-crafted phishing features
    - Confidence score (0–100%)
    - Risk level classification (low / medium / high)
    - Optional SHAP feature explanation
    """,
)
async def predict_email(
    request: Request,
    body: PredictRequest,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """Predict phishing status of a single email."""
    from utils.logger import prediction_logger

    # Input sanitisation
    email_text = sanitise_text(body.email_text)

    if is_suspicious_input(email_text):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input contains potentially malicious content.",
        )

    # Get detector
    detector = get_detector()
    if not detector.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded. Run the training pipeline first.",
        )

    # Run prediction
    try:
        result = detector.predict(
            email_text=email_text,
            include_shap=body.include_explanation,
        )
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please try again.",
        )

    # Persist to MongoDB
    prediction_doc = None
    try:
        user_id = current_user.get("sub") if current_user else None
        ip = request.client.host if request.client else None

        prediction_doc = PredictionDocument(
            email_hash=result["email_hash"],
            email_preview=email_text[:200],
            prediction=result["prediction"],
            label=result["label"],
            confidence=result["confidence"] / 100,  # Store as 0-1
            confidence_percent=result["confidence"],
            risk_level=result["risk_level"],
            model_name=result["model_name"] or "unknown",
            threshold=result["threshold"],
            phishing_keywords=result["phishing_keywords"],
            shap_top_features=result.get("top_features") or [],
            user_id=user_id,
            ip_address=ip,
            processing_time_ms=result["processing_time_ms"],
        )
        await prediction_doc.insert()
    except Exception as exc:
        logger.warning(f"Failed to save prediction to DB: {exc}")

    # Log prediction
    prediction_logger.info(
        "Prediction made",
        prediction=result["prediction"],
        confidence=result["confidence"],
        risk_level=result["risk_level"],
        processing_time_ms=result["processing_time_ms"],
    )

    return PredictResponse(
        prediction=result["prediction"],
        label=result["label"],
        confidence=result["confidence"],
        risk_level=result["risk_level"],
        model_name=result["model_name"] or "unknown",
        threshold=result["threshold"],
        phishing_keywords=result["phishing_keywords"],
        top_features=result.get("top_features"),
        processing_time_ms=result["processing_time_ms"],
        prediction_id=str(prediction_doc.id) if prediction_doc else None,
    )


class BatchPredictRequest(PredictRequest):
    pass


@router.post(
    "/batch",
    summary="Batch predict multiple emails",
    description="Predict phishing status for a list of email texts (max 50).",
)
async def batch_predict(
    emails: List[PredictRequest],
    current_user: dict = Depends(get_current_user),
):
    """Predict phishing status for multiple emails (requires authentication)."""
    if len(emails) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds maximum of 50 emails.",
        )

    detector = get_detector()
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for req in emails:
        text = sanitise_text(req.email_text)
        result = detector.predict(text)
        results.append({
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "risk_level": result["risk_level"],
        })

    return {"results": results, "count": len(results)}
