"""
PhishGuard AI - ML Pipeline Tests
====================================
Unit tests for text preprocessing and feature engineering.
"""

import pytest
from ml_pipeline.preprocessing.text_preprocessor import TextPreprocessor, PHISHING_KEYWORDS


class TestTextPreprocessor:
    """Tests for the NLP preprocessing pipeline."""

    def setup_method(self):
        """Create a preprocessor instance for each test."""
        self.preprocessor = TextPreprocessor(use_lemmatizer=True)

    def test_basic_preprocessing(self):
        """Test that basic text is cleaned."""
        result = self.preprocessor.preprocess("Hello World!")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_html_removal(self):
        """Test HTML tag removal."""
        html_text = "<h1>Congratulations!</h1><p>You won a prize.</p>"
        result = self.preprocessor.preprocess(html_text)
        assert "<h1>" not in result
        assert "<p>" not in result

    def test_url_replacement(self):
        """Test URL replacement with token."""
        text = "Click here: http://phishing-site.com/claim?id=123"
        result = self.preprocessor.preprocess(text)
        assert "http" not in result.lower() or "url" in result.lower()

    def test_email_removal(self):
        """Test email address removal."""
        text = "Contact us at winner@fake-bank.com for your prize"
        result = self.preprocessor.preprocess(text)
        assert "@" not in result

    def test_lowercase_conversion(self):
        """Test that output is lowercased."""
        text = "URGENT SECURITY ALERT"
        result = self.preprocessor.preprocess(text)
        assert result == result.lower()

    def test_empty_input(self):
        """Test handling of empty input."""
        result = self.preprocessor.preprocess("")
        assert result == ""

    def test_none_like_input(self):
        """Test handling of None-like falsy input."""
        result = self.preprocessor.preprocess("")
        assert result == ""

    def test_stopword_removal(self):
        """Test that common stopwords are removed."""
        text = "this is a very important message for you"
        result = self.preprocessor.preprocess(text)
        # Common stopwords like 'this', 'is', 'a', 'for' should be removed
        tokens = result.split()
        stopwords_in_result = {"this", "is", "a", "for"} & set(tokens)
        assert len(stopwords_in_result) == 0

    def test_batch_processing(self):
        """Test batch preprocessing."""
        texts = [
            "Congratulations! You won!",
            "Dear customer, your invoice is attached.",
            "",
        ]
        results = self.preprocessor.preprocess_batch(texts, verbose=False)
        assert len(results) == 3
        assert results[2] == ""  # Empty input → empty output

    def test_phishing_feature_extraction(self):
        """Test phishing feature extraction."""
        phishing_text = (
            "URGENT! Click http://win.fake.com to claim your FREE prize! "
            "Use your CREDIT card details to verify. $$$"
        )
        features = self.preprocessor.extract_phishing_features(phishing_text)

        assert features["has_url"] is True
        assert features["url_count"] >= 1
        assert features["exclamation_count"] >= 1
        assert features["dollar_sign_count"] >= 1
        assert features["phishing_keyword_count"] > 0
        assert features["all_caps_word_count"] > 0

    def test_legitimate_email_features(self):
        """Legitimate emails should have low phishing feature scores."""
        legit_text = (
            "Hi team, please review the attached quarterly report. "
            "Meeting is scheduled for tomorrow at 3 PM. Best regards."
        )
        features = self.preprocessor.extract_phishing_features(legit_text)
        assert features["has_url"] is False
        assert features["dollar_sign_count"] == 0
        assert features["phishing_keyword_count"] <= 2


class TestPhishingKeywords:
    """Tests for the phishing keyword vocabulary."""

    def test_keywords_not_empty(self):
        """Phishing keyword set should not be empty."""
        assert len(PHISHING_KEYWORDS) > 10

    def test_common_phishing_words_present(self):
        """Common phishing words should be in the vocabulary."""
        expected = {"urgent", "click", "prize", "verify", "account"}
        assert expected.issubset(PHISHING_KEYWORDS)
