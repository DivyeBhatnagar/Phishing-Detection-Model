"""
PhishGuard AI - Data Loader
=============================
Loads and merges multiple phishing / spam datasets into a
unified DataFrame with a standard schema:

    text_combined | label

Supported datasets:
- phishing_email.csv  (text_combined, label)
- CEAS_08.csv         (sender, receiver, date, subject, body, label, urls)
- Enron.csv           (subject, body, label)
- Ling.csv            (subject, body, label)
- Nazario.csv         (subject, body, label)
- Nigerian_Fraud.csv  (subject, body, label)
- SpamAssasin.csv     (subject, body, label)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

# ── Schema mapping per dataset ─────────────────────────────────────────────────
DATASET_CONFIGS: Dict[str, Dict] = {
    "phishing_email.csv": {
        "text_col": "text_combined",
        "label_col": "label",
        "combine": False,
    },
    "CEAS_08.csv": {
        "subject_col": "subject",
        "body_col": "body",
        "label_col": "label",
        "combine": True,
    },
    "Enron.csv": {
        "subject_col": "subject",
        "body_col": "body",
        "label_col": "label",
        "combine": True,
    },
    "Ling.csv": {
        "subject_col": "subject",
        "body_col": "body",
        "label_col": "label",
        "combine": True,
    },
    "Nazario.csv": {
        "subject_col": "subject",
        "body_col": "body",
        "label_col": "label",
        "combine": True,
    },
    "Nigerian_Fraud.csv": {
        "subject_col": "subject",
        "body_col": "body",
        "label_col": "label",
        "combine": True,
    },
    "SpamAssasin.csv": {
        "subject_col": "subject",
        "body_col": "body",
        "label_col": "label",
        "combine": True,
    },
}


def _load_single_dataset(filepath: Path, config: Dict) -> Optional[pd.DataFrame]:
    """Load a single CSV and normalise to {text_combined, label} schema."""
    if not filepath.exists():
        logger.warning(f"Dataset not found, skipping: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        logger.info(f"Loaded {len(df):,} rows from {filepath.name}")
    except Exception as exc:
        logger.error(f"Failed to read {filepath.name}: {exc}")
        return None

    label_col = config.get("label_col", "label")
    if label_col not in df.columns:
        logger.error(f"{filepath.name}: Missing label column '{label_col}'")
        return None

    if config.get("combine", False):
        # Combine subject + body into text_combined
        subject_col = config.get("subject_col", "subject")
        body_col = config.get("body_col", "body")

        subj = df.get(subject_col, pd.Series([""] * len(df))).fillna("")
        body = df.get(body_col, pd.Series([""] * len(df))).fillna("")
        df["text_combined"] = (subj + " " + body).str.strip()
    else:
        text_col = config.get("text_col", "text_combined")
        if text_col not in df.columns:
            logger.error(f"{filepath.name}: Missing text column '{text_col}'")
            return None
        df["text_combined"] = df[text_col].fillna("")

    # Normalise label to 0 / 1 integers
    df["label"] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df[["text_combined", "label"]].copy()


def load_datasets(
    dataset_dir: str | Path = "./datasets",
    datasets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load and merge all available phishing datasets.

    Args:
        dataset_dir: Directory containing dataset CSV files.
        datasets: Optional list of specific filenames to load.
                  Loads all known datasets if None.

    Returns:
        Unified DataFrame with columns [text_combined, label].
    """
    dataset_dir = Path(dataset_dir)
    targets = datasets or list(DATASET_CONFIGS.keys())

    frames: List[pd.DataFrame] = []
    for filename in targets:
        config = DATASET_CONFIGS.get(filename)
        if not config:
            logger.warning(f"No config found for {filename}, skipping.")
            continue
        df = _load_single_dataset(dataset_dir / filename, config)
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        raise RuntimeError(
            "No datasets could be loaded. "
            f"Check that CSV files exist in: {dataset_dir}"
        )

    combined = pd.concat(frames, ignore_index=True)

    # Remove exact duplicates
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["text_combined"])
    after_dedup = len(combined)
    logger.info(
        f"Removed {before_dedup - after_dedup:,} duplicate rows. "
        f"Final dataset: {after_dedup:,} rows."
    )

    # Summary
    spam_count = (combined["label"] == 1).sum()
    legit_count = (combined["label"] == 0).sum()
    logger.info(
        f"Dataset summary — "
        f"Spam/Phishing: {spam_count:,} | "
        f"Legitimate: {legit_count:,} | "
        f"Total: {len(combined):,}"
    )

    return combined


def get_dataset_stats(df: pd.DataFrame) -> Dict:
    """Return a dictionary of dataset statistics."""
    stats = {
        "total": len(df),
        "spam": int((df["label"] == 1).sum()),
        "legitimate": int((df["label"] == 0).sum()),
        "spam_ratio": round((df["label"] == 1).mean() * 100, 2),
        "avg_text_length": round(df["text_combined"].str.len().mean(), 1),
        "min_text_length": int(df["text_combined"].str.len().min()),
        "max_text_length": int(df["text_combined"].str.len().max()),
        "empty_texts": int((df["text_combined"].str.strip() == "").sum()),
    }
    return stats


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/validation/test split.

    Returns:
        (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    # First split off test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    # Then split validation from the train_val set
    adjusted_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        stratify=train_val["label"],
        random_state=random_state,
    )

    logger.info(
        f"Split — Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}"
    )
    return train, val, test
