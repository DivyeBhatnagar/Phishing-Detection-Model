"""
PhishGuard AI - Advanced NLP Text Preprocessor
================================================
Implements a comprehensive text preprocessing pipeline for email content:

1. HTML / XML tag removal
2. URL normalisation
3. Email address removal
4. Lowercase conversion
5. Punctuation / special character removal
6. Whitespace normalisation
7. Stopword removal
8. Stemming (PorterStemmer) OR Lemmatization (WordNetLemmatizer)

Designed for high-throughput batch processing with nltk.
"""

from __future__ import annotations

import re
import string
from typing import List, Optional

import nltk
from bs4 import BeautifulSoup
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ── NLTK Resource Bootstrap ────────────────────────────────────────────────────
def download_nltk_resources() -> None:
    """Download required NLTK corpora if not already present.

    Handles macOS SSL certificate verification issue automatically.
    """
    import ssl

    # Fix macOS SSL certificate verification issue
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context
    except AttributeError:
        pass

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name, quiet=True)


# ── Compiled Regex Patterns ────────────────────────────────────────────────────
URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|"
    r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(
    r"(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)\d{3}[\s\-]?\d{4}"
)
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
REPEATED_CHARS = re.compile(r"(.)\1{3,}")  # e.g. "freeeee" → "free"

# ── Phishing Keyword Vocabulary (used for feature engineering) ─────────────────
PHISHING_KEYWORDS = {
    "urgent", "click", "verify", "account", "suspended", "prize",
    "winner", "congratulations", "free", "offer", "limited", "expire",
    "password", "confirm", "bank", "credit", "debit", "login",
    "update", "immediately", "warning", "alert", "security",
    "paypal", "amazon", "microsoft", "bitcoin", "cryptocurrency",
    "claim", "reward", "gift", "billion", "million", "transfer",
    "inheritance", "prince", "nigeria", "help", "wire",
}


class TextPreprocessor:
    """
    Full NLP preprocessing pipeline for email text.

    Usage::

        preprocessor = TextPreprocessor(use_lemmatizer=True)
        clean = preprocessor.preprocess("Click here to WIN $1000!")
    """

    def __init__(
        self,
        use_lemmatizer: bool = True,
        remove_stopwords: bool = True,
        min_token_length: int = 2,
        max_tokens: int = 1_000,
    ) -> None:
        download_nltk_resources()

        self.use_lemmatizer = use_lemmatizer
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.max_tokens = max_tokens

        self._stop_words: set[str] = set(stopwords.words("english"))
        self._stemmer = PorterStemmer()
        self._lemmatizer = WordNetLemmatizer()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _remove_html(self, text: str) -> str:
        """Strip HTML/XML tags using BeautifulSoup."""
        try:
            soup = BeautifulSoup(text, "lxml")
            return soup.get_text(separator=" ")
        except Exception:
            # Fallback: crude regex
            return re.sub(r"<[^>]+>", " ", text)

    def _normalise_urls(self, text: str) -> str:
        """Replace URLs with the token 'URL'."""
        return URL_PATTERN.sub(" URL ", text)

    def _remove_emails(self, text: str) -> str:
        """Replace email addresses with 'EMAIL'."""
        return EMAIL_PATTERN.sub(" EMAIL ", text)

    def _remove_phones(self, text: str) -> str:
        """Replace phone numbers with 'PHONE'."""
        return PHONE_PATTERN.sub(" PHONE ", text)

    def _remove_special_chars(self, text: str) -> str:
        """Remove non-alphanumeric characters."""
        return SPECIAL_CHAR_PATTERN.sub(" ", text)

    def _normalise_repeated(self, text: str) -> str:
        """Collapse repeated characters: freeeeee → free."""
        return REPEATED_CHARS.sub(r"\1\1", text)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and apply stemming / lemmatization."""
        tokens = word_tokenize(text.lower())

        processed: List[str] = []
        for token in tokens[: self.max_tokens]:
            # Filter short / numeric tokens
            if len(token) < self.min_token_length:
                continue
            if token.isdigit():
                continue

            # Remove stopwords
            if self.remove_stopwords and token in self._stop_words:
                continue

            # Normalise token
            if self.use_lemmatizer:
                token = self._lemmatizer.lemmatize(token, pos="v")
            else:
                token = self._stemmer.stem(token)

            processed.append(token)

        return processed

    # ── Public API ─────────────────────────────────────────────────────────────

    def preprocess(self, text: str) -> str:
        """
        Run the full preprocessing pipeline.

        Args:
            text: Raw email content (subject + body).

        Returns:
            Clean, normalised text string ready for vectorisation.
        """
        if not text or not isinstance(text, str):
            return ""

        # Pipeline steps
        text = self._remove_html(text)
        text = self._normalise_urls(text)
        text = self._remove_emails(text)
        text = self._remove_phones(text)
        text = self._normalise_repeated(text)
        text = self._remove_special_chars(text)
        text = WHITESPACE_PATTERN.sub(" ", text).strip()

        # Tokenize + normalise
        tokens = self._tokenize(text)
        return " ".join(tokens)

    def preprocess_batch(self, texts: List[str], verbose: bool = True) -> List[str]:
        """
        Preprocess a list of texts efficiently.

        Args:
            texts: List of raw email strings.
            verbose: Log progress every 10k items.

        Returns:
            List of cleaned text strings.
        """
        results: List[str] = []
        total = len(texts)
        for i, text in enumerate(texts):
            results.append(self.preprocess(text))
            if verbose and (i + 1) % 10_000 == 0:
                logger.info(f"Preprocessed {i + 1:,} / {total:,} texts")
        return results

    def extract_phishing_features(self, text: str) -> dict:
        """
        Extract phishing-specific boolean/numeric features.

        Returns a dict of hand-crafted features for feature engineering.
        """
        text_lower = text.lower()
        tokens = set(text_lower.split())

        return {
            "has_url": bool(URL_PATTERN.search(text)),
            "url_count": len(URL_PATTERN.findall(text)),
            "has_email": bool(EMAIL_PATTERN.search(text)),
            "has_phone": bool(PHONE_PATTERN.search(text)),
            "phishing_keyword_count": len(tokens & PHISHING_KEYWORDS),
            "has_html": bool(re.search(r"<[a-zA-Z]", text)),
            "exclamation_count": text.count("!"),
            "dollar_sign_count": text.count("$"),
            "question_count": text.count("?"),
            "all_caps_word_count": sum(1 for w in text.split() if w.isupper() and len(w) > 2),
            "text_length": len(text),
            "word_count": len(text.split()),
            "digit_ratio": sum(c.isdigit() for c in text) / max(len(text), 1),
        }
