#!/usr/bin/env python3
"""
PhishGuard AI - Training Script
=================================
Standalone script to train the phishing detection model.

Usage:
    python scripts/train.py
    python scripts/train.py --max-features 50000
    python scripts/train.py --run-cv --min-recall 0.97
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.pipeline import run_training_pipeline
from utils.logger import setup_logging
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train PhishGuard AI model")
    parser.add_argument("--dataset-dir", default="./datasets")
    parser.add_argument("--model-dir", default="./models/saved")
    parser.add_argument("--max-features", type=int, default=75_000)
    parser.add_argument("--run-cv", action="store_true")
    parser.add_argument("--no-char-ngrams", action="store_true")
    parser.add_argument("--min-recall", type=float, default=0.95)
    args = parser.parse_args()

    setup_logging()

    print("\n" + "=" * 60)
    print("  PhishGuard AI — Model Training")
    print("=" * 60 + "\n")

    results = run_training_pipeline(
        dataset_dir=args.dataset_dir,
        model_save_dir=args.model_dir,
        run_cv=args.run_cv,
        max_features=args.max_features,
        use_char_ngrams=not args.no_char_ngrams,
        min_recall=args.min_recall,
    )

    print("\n" + "=" * 60)
    print(f"  ✅ Training Complete!")
    print(f"  Best Model:  {results['best_model']}")
    print(f"  Threshold:   {results['threshold']:.4f}")
    m = results['test_metrics']
    print(f"  Precision:   {m['precision']:.4f}")
    print(f"  Recall:      {m['recall']:.4f}")
    print(f"  F1 Score:    {m['f1']:.4f}")
    print(f"  ROC-AUC:     {m['roc_auc']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
