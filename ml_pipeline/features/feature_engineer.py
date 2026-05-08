"""
PhishGuard AI - Feature Engineering
=====================================
Builds the final feature matrix by combining:
1. TF-IDF features from clean text
2. Hand-crafted phishing-specific features

The combined feature set improves recall on subtle phishing attempts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from ml_pipeline.preprocessing.text_preprocessor import TextPreprocessor


class FeatureEngineer:
    """
    Builds the complete feature matrix for the phishing classifier.

    Components:
    - TF-IDF (word n-grams + char n-grams)
    - Phishing-specific hand-crafted features
    """

    def __init__(
        self,
        max_features: int = 75_000,
        ngram_range_word: Tuple[int, int] = (1, 2),
        ngram_range_char: Tuple[int, int] = (3, 5),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        use_char_ngrams: bool = True,
    ) -> None:
        self.max_features = max_features
        self.use_char_ngrams = use_char_ngrams

        # Word-level TF-IDF
        self.word_tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range_word,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            analyzer="word",
            token_pattern=r"\b[a-zA-Z]{2,}\b",
            strip_accents="unicode",
        )

        # Character-level TF-IDF (catches obfuscated phishing text)
        if use_char_ngrams:
            self.char_tfidf = TfidfVectorizer(
                max_features=max_features // 3,
                ngram_range=ngram_range_char,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=sublinear_tf,
                analyzer="char_wb",
            )

        self.scaler = MinMaxScaler()
        self._preprocessor = TextPreprocessor(use_lemmatizer=True)
        self._is_fitted: bool = False

    # ── Feature Extraction ─────────────────────────────────────────────────────

    def _extract_handcrafted(self, texts: List[str]) -> np.ndarray:
        """
        Extract hand-crafted phishing features from raw texts.

        Returns:
            Dense numpy array of shape (n_samples, n_features).
        """
        rows = []
        for text in texts:
            feats = self._preprocessor.extract_phishing_features(text)
            rows.append(list(feats.values()))
        return np.array(rows, dtype=np.float32)

    def fit_transform(
        self,
        raw_texts: List[str],
        clean_texts: Optional[List[str]] = None,
    ) -> csr_matrix:
        """
        Fit all transformers on training data and transform.

        Args:
            raw_texts: Original email texts (for hand-crafted features).
            clean_texts: Pre-processed texts. If None, preprocessing
                         is applied here.

        Returns:
            Sparse feature matrix.
        """
        if clean_texts is None:
            logger.info("Preprocessing texts inside FeatureEngineer...")
            clean_texts = self._preprocessor.preprocess_batch(raw_texts)

        logger.info("Fitting TF-IDF word vectorizer...")
        word_matrix = self.word_tfidf.fit_transform(clean_texts)

        parts = [word_matrix]

        if self.use_char_ngrams:
            logger.info("Fitting TF-IDF char vectorizer...")
            char_matrix = self.char_tfidf.fit_transform(clean_texts)
            parts.append(char_matrix)

        logger.info("Extracting hand-crafted phishing features...")
        hc_features = self._extract_handcrafted(raw_texts)
        hc_scaled = self.scaler.fit_transform(hc_features)
        # Convert to sparse for hstack
        hc_sparse = csr_matrix(hc_scaled)
        parts.append(hc_sparse)

        combined = hstack(parts, format="csr")
        self._is_fitted = True

        logger.info(
            f"Feature matrix shape: {combined.shape} "
            f"({combined.shape[1]:,} features)"
        )
        return combined

    def transform(
        self,
        raw_texts: List[str],
        clean_texts: Optional[List[str]] = None,
    ) -> csr_matrix:
        """
        Transform new texts using the fitted vectorizers.

        Args:
            raw_texts: Original email texts.
            clean_texts: Pre-processed texts (optional).

        Returns:
            Sparse feature matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before calling transform().")

        if clean_texts is None:
            clean_texts = self._preprocessor.preprocess_batch(raw_texts, verbose=False)

        word_matrix = self.word_tfidf.transform(clean_texts)
        parts = [word_matrix]

        if self.use_char_ngrams:
            char_matrix = self.char_tfidf.transform(clean_texts)
            parts.append(char_matrix)

        hc_features = self._extract_handcrafted(raw_texts)
        hc_scaled = self.scaler.transform(hc_features)
        hc_sparse = csr_matrix(hc_scaled)
        parts.append(hc_sparse)

        return hstack(parts, format="csr")

    def transform_single(self, raw_text: str) -> csr_matrix:
        """Convenience wrapper for transforming a single email."""
        return self.transform([raw_text])

    def get_feature_names(self) -> List[str]:
        """Return combined list of feature names."""
        names = list(self.word_tfidf.get_feature_names_out())
        if self.use_char_ngrams:
            names += [f"char_{n}" for n in self.char_tfidf.get_feature_names_out()]
        # Hand-crafted feature names
        sample_feats = self._preprocessor.extract_phishing_features("")
        names += list(sample_feats.keys())
        return names
