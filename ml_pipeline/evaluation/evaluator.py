"""
PhishGuard AI - Model Evaluator
=================================
Comprehensive evaluation suite:
- Confusion matrix
- Precision / Recall / F1 / Accuracy / ROC-AUC
- Optimal threshold selection (maximise Recall @ target precision)
- Classification report
- SHAP explainability (optional)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(
    model: Any,
    X: csr_matrix,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute a full evaluation report for a fitted classifier.

    Args:
        model: Fitted sklearn-compatible classifier.
        X: Feature matrix.
        y_true: Ground truth labels.
        threshold: Decision threshold for positive class.

    Returns:
        Dict containing all evaluation metrics.
    """
    # Probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.decision_function(X)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    y_pred = (y_prob >= threshold).astype(int)

    # Core metrics
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["Legitimate", "Phishing"],
            output_dict=True,
        ),
        "threshold": threshold,
    }


def find_optimal_threshold(
    model: Any,
    X: csr_matrix,
    y_true: np.ndarray,
    min_recall: float = 0.95,
) -> Tuple[float, Dict]:
    """
    Find the optimal decision threshold that achieves at least
    `min_recall` while maximising precision.

    Args:
        model: Fitted classifier.
        X: Feature matrix.
        y_true: Ground truth labels.
        min_recall: Minimum acceptable recall (default 0.95 for safety).

    Returns:
        (optimal_threshold, metrics_at_threshold)
    """
    y_prob = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    best_threshold = 0.5
    best_precision = 0.0

    for i, (p, r, t) in enumerate(zip(precisions[:-1], recalls[:-1], thresholds)):
        if r >= min_recall and p > best_precision:
            best_precision = p
            best_threshold = t

    metrics = evaluate_model(model, X, y_true, threshold=best_threshold)
    logger.info(
        f"Optimal threshold: {best_threshold:.4f} — "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f}"
    )
    return best_threshold, metrics


def print_comparison_table(results: List[Dict]) -> None:
    """Pretty-print model comparison table to console."""
    header = (
        f"{'Model':<25} {'Precision':>10} {'Recall':>10} "
        f"{'F1':>10} {'ROC-AUC':>10} {'Time(s)':>8} {'Best':>6}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    for row in results:
        if "error" in row:
            print(f"{row['model']:<25} {'ERROR':<55}")
            continue
        best_mark = "★" if row.get("is_best") else ""
        print(
            f"{row['model']:<25} "
            f"{row.get('precision', 0):>10.4f} "
            f"{row.get('recall', 0):>10.4f} "
            f"{row.get('f1', 0):>10.4f} "
            f"{row.get('roc_auc', 0):>10.4f} "
            f"{row.get('training_time_s', 0):>8.1f} "
            f"{best_mark:>6}"
        )

    print(separator + "\n")


def generate_shap_explanation(
    model: Any,
    X_sample: csr_matrix,
    feature_names: List[str],
    max_samples: int = 100,
) -> Optional[Dict]:
    """
    Generate SHAP values for model explainability.

    Args:
        model: Fitted classifier (tree-based models work best).
        X_sample: Feature matrix (subset for speed).
        feature_names: List of feature names.
        max_samples: Cap number of samples to explain.

    Returns:
        Dict with shap_values and top feature importances.
    """
    try:
        import shap  # type: ignore

        logger.info("Generating SHAP explanations...")

        # Use dense array for SHAP
        X_dense = X_sample[:max_samples].toarray()

        if hasattr(model, "predict_proba"):
            explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") \
                else shap.LinearExplainer(model, X_dense)
        else:
            explainer = shap.LinearExplainer(model, X_dense)

        shap_values = explainer.shap_values(X_dense)

        # If binary classification, take phishing class values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Top 20 most impactful features
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:20]

        top_features = [
            {
                "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                "importance": round(float(mean_abs[i]), 6),
            }
            for i in top_indices
        ]

        logger.info(f"SHAP top feature: {top_features[0]['feature']}")
        return {
            "top_features": top_features,
            "shap_values_shape": list(shap_values.shape),
        }

    except Exception as exc:
        logger.warning(f"SHAP explanation failed (non-critical): {exc}")
        return None
