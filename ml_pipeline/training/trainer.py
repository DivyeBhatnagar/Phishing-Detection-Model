"""
PhishGuard AI - Model Trainer
==============================
Trains and compares multiple classifiers:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Multinomial Naive Bayes

Optimises for Recall and F1-score (critical for phishing detection).
Auto-selects the best model by F1 score.

Usage::

    trainer = ModelTrainer()
    results = trainer.train_all(X_train, y_train, X_val, y_val)
    best_model = trainer.get_best_model()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ── Model Definitions ─────────────────────────────────────────────────────────

def get_model_configs() -> Dict[str, Any]:
    """
    Return optimised model configurations.
    All models are tuned to prioritise RECALL (phishing detection).
    """
    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",   # Handles imbalance
            solver="saga",
            n_jobs=-1,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,        # Set dynamically in train_all()
            use_label_encoder=False,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
        "naive_bayes": ComplementNB(alpha=0.1),  # ComplementNB excels for text
    }


class ModelTrainer:
    """Trains, evaluates, and selects the best phishing classifier."""

    def __init__(self, cv_folds: int = 5, scoring: str = "f1") -> None:
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.results: Dict[str, Dict] = {}
        self.trained_models: Dict[str, Any] = {}
        self._best_model_name: Optional[str] = None

    def _compute_class_weight(self, y: np.ndarray) -> float:
        """Compute scale_pos_weight for XGBoost (negatives / positives)."""
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        return round(n_neg / max(n_pos, 1), 4)

    def _train_single(
        self,
        name: str,
        model: Any,
        X_train: csr_matrix,
        y_train: np.ndarray,
    ) -> Tuple[Any, float]:
        """Train a single model and return (fitted_model, training_time_s)."""
        logger.info(f"  Training {name}...")
        start = time.perf_counter()

        if name == "xgboost":
            model.scale_pos_weight = self._compute_class_weight(y_train)

        model.fit(X_train, y_train)
        elapsed = round(time.perf_counter() - start, 2)
        logger.info(f"  {name} trained in {elapsed}s")
        return model, elapsed

    def _cross_validate(
        self,
        model: Any,
        X: csr_matrix,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Run stratified k-fold CV and return mean metric scores."""
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scoring = {
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }
        cv_result = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        return {
            metric: round(float(np.mean(cv_result[f"test_{metric}"])), 4)
            for metric in scoring
        }

    def train_all(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray,
        run_cv: bool = False,
    ) -> Dict[str, Dict]:
        """
        Train all configured models and evaluate on the validation set.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Validation feature matrix.
            y_val: Validation labels.
            run_cv: Whether to run cross-validation (slower but more reliable).

        Returns:
            Dict mapping model name → evaluation metrics.
        """
        from ml_pipeline.evaluation.evaluator import evaluate_model

        model_configs = get_model_configs()
        logger.info(f"Training {len(model_configs)} models...")

        for name, model in model_configs.items():
            try:
                fitted_model, train_time = self._train_single(
                    name, model, X_train, y_train
                )
                self.trained_models[name] = fitted_model

                # Evaluate on validation set
                metrics = evaluate_model(fitted_model, X_val, y_val)
                metrics["training_time_s"] = train_time

                # Optional cross-validation
                if run_cv:
                    logger.info(f"  Running {self.cv_folds}-fold CV for {name}...")
                    cv_metrics = self._cross_validate(model, X_train, y_train)
                    metrics["cv"] = cv_metrics

                self.results[name] = metrics
                logger.info(
                    f"  ✓ {name} — "
                    f"Precision: {metrics['precision']:.4f} | "
                    f"Recall: {metrics['recall']:.4f} | "
                    f"F1: {metrics['f1']:.4f} | "
                    f"ROC-AUC: {metrics['roc_auc']:.4f}"
                )

            except Exception as exc:
                logger.error(f"  ✗ {name} failed: {exc}")
                self.results[name] = {"error": str(exc)}

        self._select_best_model()
        return self.results

    def _select_best_model(self) -> None:
        """Select best model by F1 score (primary) + Recall (secondary)."""
        valid = {
            k: v for k, v in self.results.items()
            if "error" not in v and "f1" in v
        }
        if not valid:
            logger.warning("No valid models to select from!")
            return

        # Sort by F1 descending, then Recall descending
        self._best_model_name = max(
            valid,
            key=lambda k: (valid[k]["f1"], valid[k].get("recall", 0)),
        )
        logger.info(
            f"🏆 Best model selected: {self._best_model_name} "
            f"(F1={valid[self._best_model_name]['f1']:.4f})"
        )

    def get_best_model(self) -> Tuple[str, Any]:
        """Return (model_name, fitted_model) for the best model."""
        if not self._best_model_name:
            raise RuntimeError("No models trained yet. Call train_all() first.")
        return self._best_model_name, self.trained_models[self._best_model_name]

    def get_comparison_table(self) -> List[Dict]:
        """Return sorted model comparison as a list of dicts."""
        rows = []
        for name, metrics in self.results.items():
            if "error" in metrics:
                rows.append({"model": name, "error": metrics["error"]})
            else:
                rows.append({
                    "model": name,
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1": metrics.get("f1", 0),
                    "roc_auc": metrics.get("roc_auc", 0),
                    "accuracy": metrics.get("accuracy", 0),
                    "training_time_s": metrics.get("training_time_s", 0),
                    "is_best": name == self._best_model_name,
                })
        return sorted(rows, key=lambda r: r.get("f1", 0), reverse=True)
