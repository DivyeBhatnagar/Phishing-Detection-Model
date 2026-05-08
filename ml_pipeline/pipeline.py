"""
PhishGuard AI - Full Training Pipeline
========================================
Orchestrates the complete ML training workflow:

1. Load datasets
2. Preprocess text
3. Feature engineering
4. Train/Val/Test split
5. Train all models
6. Evaluate and compare
7. Select best model
8. Find optimal threshold
9. SHAP explanation
10. Save artefacts

Run directly:
    python ml_pipeline/pipeline.py

Or import for programmatic use:
    from ml_pipeline.pipeline import run_training_pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.data.loader import load_datasets, get_dataset_stats, split_data
from ml_pipeline.evaluation.evaluator import (
    evaluate_model,
    find_optimal_threshold,
    generate_shap_explanation,
    print_comparison_table,
)
from ml_pipeline.features.feature_engineer import FeatureEngineer
from ml_pipeline.models.model_store import save_model_artefacts
from ml_pipeline.preprocessing.text_preprocessor import TextPreprocessor
from ml_pipeline.training.trainer import ModelTrainer
from utils.logger import setup_logging, training_logger


def run_training_pipeline(
    dataset_dir: str | Path = "./datasets",
    model_save_dir: str | Path = "./models/saved",
    run_cv: bool = False,
    max_features: int = 75_000,
    use_char_ngrams: bool = True,
    min_recall: float = 0.95,
    log_dir: str | Path = "./logs",
) -> Dict:
    """
    Execute the full training pipeline.

    Args:
        dataset_dir: Path to directory containing dataset CSV files.
        model_save_dir: Where to save trained artefacts.
        run_cv: Whether to run cross-validation (slow but thorough).
        max_features: TF-IDF max feature count.
        use_char_ngrams: Include character n-gram features.
        min_recall: Minimum recall for threshold optimisation.
        log_dir: Directory for logs.

    Returns:
        Dict with training results and best model metrics.
    """
    setup_logging(log_dir=log_dir)
    training_logger.info("=" * 60)
    training_logger.info("  PhishGuard AI — Training Pipeline")
    training_logger.info("=" * 60)

    # ── Step 1: Load Data ──────────────────────────────────────────────────────
    training_logger.info("[1/9] Loading datasets...")
    df = load_datasets(dataset_dir=dataset_dir)
    stats = get_dataset_stats(df)
    training_logger.info(f"Dataset stats: {stats}")

    # ── Step 2: Train/Val/Test Split ──────────────────────────────────────────
    training_logger.info("[2/9] Splitting data (70/10/20)...")
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.1)

    # ── Step 3: Text Preprocessing ────────────────────────────────────────────
    training_logger.info("[3/9] Preprocessing text (train)...")
    preprocessor = TextPreprocessor(use_lemmatizer=True)

    train_clean = preprocessor.preprocess_batch(train_df["text_combined"].tolist())
    val_clean = preprocessor.preprocess_batch(val_df["text_combined"].tolist())
    test_clean = preprocessor.preprocess_batch(test_df["text_combined"].tolist())

    # ── Step 4: Feature Engineering ───────────────────────────────────────────
    training_logger.info("[4/9] Building feature matrix (TF-IDF + hand-crafted)...")
    fe = FeatureEngineer(
        max_features=max_features,
        use_char_ngrams=use_char_ngrams,
    )

    X_train = fe.fit_transform(
        train_df["text_combined"].tolist(),
        clean_texts=train_clean,
    )
    X_val = fe.transform(
        val_df["text_combined"].tolist(),
        clean_texts=val_clean,
    )
    X_test = fe.transform(
        test_df["text_combined"].tolist(),
        clean_texts=test_clean,
    )

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    training_logger.info(
        f"Feature matrix — Train: {X_train.shape} | "
        f"Val: {X_val.shape} | Test: {X_test.shape}"
    )

    # ── Step 5: Train All Models ──────────────────────────────────────────────
    training_logger.info("[5/9] Training all models...")
    trainer = ModelTrainer(cv_folds=5)
    results = trainer.train_all(X_train, y_train, X_val, y_val, run_cv=run_cv)

    # ── Step 6: Print Comparison Table ────────────────────────────────────────
    training_logger.info("[6/9] Model comparison:")
    comparison = trainer.get_comparison_table()
    print_comparison_table(comparison)

    # ── Step 7: Get Best Model ────────────────────────────────────────────────
    training_logger.info("[7/9] Selecting best model...")
    best_name, best_model = trainer.get_best_model()

    # ── Step 8: Evaluate on Test Set + Find Optimal Threshold ────────────────
    training_logger.info("[8/9] Final evaluation on test set...")
    optimal_threshold, test_metrics = find_optimal_threshold(
        best_model, X_test, y_test, min_recall=min_recall
    )

    training_logger.info("Test set results:")
    training_logger.info(
        f"  Precision: {test_metrics['precision']:.4f} | "
        f"  Recall:    {test_metrics['recall']:.4f} | "
        f"  F1:        {test_metrics['f1']:.4f} | "
        f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}"
    )
    training_logger.info(
        f"  Optimal threshold: {optimal_threshold:.4f}"
    )

    # ── Step 9: SHAP Explanation ──────────────────────────────────────────────
    training_logger.info("[9/9] Generating SHAP explanations...")
    feature_names = fe.get_feature_names()
    shap_results = generate_shap_explanation(
        best_model, X_test, feature_names, max_samples=200
    )
    if shap_results:
        training_logger.info(
            f"Top SHAP features: "
            + ", ".join(
                f['feature'] for f in shap_results["top_features"][:5]
            )
        )

    # ── Save Artefacts ────────────────────────────────────────────────────────
    save_dir = save_model_artefacts(
        model=best_model,
        feature_engineer=fe,
        model_name=best_name,
        metrics=test_metrics,
        threshold=optimal_threshold,
        save_dir=model_save_dir,
    )

    training_logger.info(f"✅ Training complete. Artefacts saved to: {save_dir}")

    return {
        "best_model": best_name,
        "threshold": optimal_threshold,
        "test_metrics": test_metrics,
        "all_results": results,
        "comparison": comparison,
        "shap": shap_results,
        "dataset_stats": stats,
    }


# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PhishGuard AI — Model Training Pipeline"
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="./datasets",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./models/saved",
        help="Path to save trained models"
    )
    parser.add_argument(
        "--max-features", type=int, default=75_000,
        help="TF-IDF max features"
    )
    parser.add_argument(
        "--run-cv", action="store_true",
        help="Run cross-validation (slower)"
    )
    parser.add_argument(
        "--no-char-ngrams", action="store_true",
        help="Disable character n-gram features"
    )
    parser.add_argument(
        "--min-recall", type=float, default=0.95,
        help="Minimum recall for threshold optimisation"
    )

    args = parser.parse_args()

    results = run_training_pipeline(
        dataset_dir=args.dataset_dir,
        model_save_dir=args.model_dir,
        run_cv=args.run_cv,
        max_features=args.max_features,
        use_char_ngrams=not args.no_char_ngrams,
        min_recall=args.min_recall,
    )

    logger.info(f"Pipeline complete. Best model: {results['best_model']}")
