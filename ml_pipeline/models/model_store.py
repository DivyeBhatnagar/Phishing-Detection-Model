"""
PhishGuard AI - Model Persistence
===================================
Handles saving and loading of:
- Trained classifier
- FeatureEngineer (TF-IDF vectorizers + scaler)
- Model metadata and metrics

All artefacts stored in models/saved/ as .joblib files.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
from loguru import logger


# ── File name constants ────────────────────────────────────────────────────────
MODEL_FILENAME = "model.joblib"
FEATURE_ENGINEER_FILENAME = "feature_engineer.joblib"
METADATA_FILENAME = "metadata.json"


def save_model_artefacts(
    model: Any,
    feature_engineer: Any,
    model_name: str,
    metrics: Dict,
    threshold: float,
    save_dir: str | Path = "models/saved",
) -> Path:
    """
    Save all model artefacts to disk.

    Args:
        model: Fitted sklearn-compatible classifier.
        feature_engineer: Fitted FeatureEngineer instance.
        model_name: Name of the model (e.g., 'lightgbm').
        metrics: Evaluation metrics dict.
        threshold: Optimal decision threshold.
        save_dir: Directory to save artefacts.

    Returns:
        Path to the saved directory.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = save_dir / MODEL_FILENAME
    joblib.dump(model, model_path, compress=3)
    logger.info(f"Model saved → {model_path} ({model_path.stat().st_size // 1024}KB)")

    # Save feature engineer
    fe_path = save_dir / FEATURE_ENGINEER_FILENAME
    joblib.dump(feature_engineer, fe_path, compress=3)
    logger.info(f"FeatureEngineer saved → {fe_path}")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        "threshold": threshold,
        "metrics": metrics,
        "files": {
            "model": MODEL_FILENAME,
            "feature_engineer": FEATURE_ENGINEER_FILENAME,
        },
    }
    meta_path = save_dir / METADATA_FILENAME
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Metadata saved → {meta_path}")

    logger.info(f"✅ All artefacts saved to: {save_dir}")
    return save_dir


def load_model_artefacts(
    load_dir: str | Path = "models/saved",
) -> Tuple[Any, Any, Dict]:
    """
    Load all model artefacts from disk.

    Args:
        load_dir: Directory containing saved artefacts.

    Returns:
        (model, feature_engineer, metadata)

    Raises:
        FileNotFoundError: If model files are missing.
    """
    load_dir = Path(load_dir)

    model_path = load_dir / MODEL_FILENAME
    fe_path = load_dir / FEATURE_ENGINEER_FILENAME
    meta_path = load_dir / METADATA_FILENAME

    for path in [model_path, fe_path, meta_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Model artefact not found: {path}\n"
                "Run training first: python scripts/train.py"
            )

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    logger.info(f"Loading FeatureEngineer from {fe_path}...")
    feature_engineer = joblib.load(fe_path)

    metadata = json.loads(meta_path.read_text())
    logger.info(
        f"Loaded model: {metadata.get('model_name')} "
        f"(saved: {metadata.get('saved_at')})"
    )

    return model, feature_engineer, metadata


def model_artefacts_exist(load_dir: str | Path = "models/saved") -> bool:
    """Check if trained model artefacts are present."""
    load_dir = Path(load_dir)
    return all(
        (load_dir / f).exists()
        for f in [MODEL_FILENAME, FEATURE_ENGINEER_FILENAME, METADATA_FILENAME]
    )


def get_model_metadata(load_dir: str | Path = "models/saved") -> Optional[Dict]:
    """Return metadata dict without loading heavy model objects."""
    meta_path = Path(load_dir) / METADATA_FILENAME
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())
