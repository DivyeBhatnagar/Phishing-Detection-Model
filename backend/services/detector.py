"""
PhishGuard AI - ML Model Service
===================================
Singleton service that:
- Loads trained model artefacts at startup
- Exposes predict() method for single and batch predictions
- Caches model in memory for low-latency inference
- Supports hot-reload after retraining
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ml_pipeline.models.model_store import (
    get_model_metadata,
    load_model_artefacts,
    model_artefacts_exist,
)
from ml_pipeline.preprocessing.text_preprocessor import TextPreprocessor, PHISHING_KEYWORDS
from utils.helpers import classify_risk, hash_text


class PhishingDetectorService:
    """
    Singleton inference service for phishing email detection.

    Usage::

        service = PhishingDetectorService()
        service.load()
        result = service.predict("Click here to claim your prize!")
    """

    def __init__(self, model_dir: str = "models/saved") -> None:
        self.model_dir = model_dir
        self._model: Optional[Any] = None
        self._feature_engineer: Optional[Any] = None
        self._metadata: Optional[Dict] = None
        self._preprocessor = TextPreprocessor(use_lemmatizer=True)
        self._loaded: bool = False

    def load(self) -> bool:
        """
        Load model artefacts from disk.

        Returns:
            True if loaded successfully, False if no artefacts found.
        """
        if not model_artefacts_exist(self.model_dir):
            logger.warning(
                f"No trained model found in '{self.model_dir}'. "
                "Run training pipeline first: python ml_pipeline/pipeline.py"
            )
            return False

        try:
            self._model, self._feature_engineer, self._metadata = (
                load_model_artefacts(self.model_dir)
            )
            self._loaded = True
            logger.info(
                f"✅ Model loaded: {self._metadata.get('model_name')} "
                f"(threshold={self._metadata.get('threshold'):.4f})"
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            return False

    def reload(self) -> bool:
        """Hot-reload model after retraining."""
        logger.info("Reloading model artefacts...")
        self._loaded = False
        return self.load()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> Optional[str]:
        return self._metadata.get("model_name") if self._metadata else None

    @property
    def threshold(self) -> float:
        return self._metadata.get("threshold", 0.5) if self._metadata else 0.5

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "Model not loaded. Call load() first or run the training pipeline."
            )

    def _detect_phishing_keywords(self, text: str) -> List[str]:
        """Return list of phishing keywords found in text."""
        words = set(text.lower().split())
        return sorted(words & PHISHING_KEYWORDS)

    def predict(
        self,
        email_text: str,
        include_shap: bool = False,
    ) -> Dict:
        """
        Predict whether an email is phishing or legitimate.

        Args:
            email_text: Raw email content (subject + body).
            include_shap: Whether to compute SHAP explanation.

        Returns:
            Dict with prediction, confidence, risk_level, etc.
        """
        self._require_loaded()

        start_time = time.perf_counter()

        # Feature extraction
        X = self._feature_engineer.transform_single(email_text)

        # Probability prediction
        y_prob = float(self._model.predict_proba(X)[0, 1])
        y_pred = int(y_prob >= self.threshold)

        # Map to labels
        prediction_label = "spam" if y_pred == 1 else "legitimate"
        confidence_pct = round(y_prob * 100, 2)
        risk = classify_risk(y_prob)

        # Phishing keyword detection
        keywords = self._detect_phishing_keywords(email_text)

        # SHAP (optional)
        top_features = None
        if include_shap:
            from ml_pipeline.evaluation.evaluator import generate_shap_explanation
            feature_names = self._feature_engineer.get_feature_names()
            shap_result = generate_shap_explanation(
                self._model, X, feature_names, max_samples=1
            )
            if shap_result:
                top_features = shap_result["top_features"][:10]

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "prediction": prediction_label,
            "label": y_pred,
            "confidence": confidence_pct,
            "risk_level": risk,
            "model_name": self.model_name,
            "threshold": self.threshold,
            "phishing_keywords": keywords,
            "top_features": top_features,
            "processing_time_ms": elapsed_ms,
            "email_hash": hash_text(email_text),
        }

    def predict_batch(self, email_texts: List[str]) -> List[Dict]:
        """Batch predict for multiple emails."""
        self._require_loaded()
        return [self.predict(text) for text in email_texts]

    def get_info(self) -> Dict:
        """Return service metadata."""
        return {
            "loaded": self._loaded,
            "model_name": self.model_name,
            "threshold": self.threshold,
            "metadata": self._metadata,
        }


# ── Global Singleton ───────────────────────────────────────────────────────────
_detector: Optional[PhishingDetectorService] = None


def get_detector() -> PhishingDetectorService:
    """Return the global PhishingDetectorService instance."""
    global _detector
    if _detector is None:
        _detector = PhishingDetectorService()
        _detector.load()
    return _detector
