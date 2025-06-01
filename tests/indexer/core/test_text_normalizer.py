"""Tests for the TextNormalizer component."""

import pytest

from oboyu.indexer.core.text_normalizer import TextNormalizer


class TestTextNormalizer:
    """Test cases for TextNormalizer."""

    def test_normalize_removes_excessive_whitespace(self) -> None:
        """Test that normalize removes excessive whitespace."""
        normalizer = TextNormalizer()
        
        # Test with multiple spaces
        text = "This   is    a   test"
        result = normalizer.normalize(text)
        assert result == "This is a test"
        
        # Test with tabs and newlines
        text = "This\t\tis\n\na\r\ntest"
        result = normalizer.normalize(text)
        assert result == "This is a test"
        
        # Test with mixed whitespace
        text = "  Leading   and   trailing   spaces  "
        result = normalizer.normalize(text)
        assert result == "Leading and trailing spaces"

    def test_normalize_preserves_single_spaces(self) -> None:
        """Test that normalize preserves single spaces."""
        normalizer = TextNormalizer()
        
        text = "This is a properly spaced sentence."
        result = normalizer.normalize(text)
        assert result == text

    def test_normalize_handles_empty_string(self) -> None:
        """Test that normalize handles empty strings."""
        normalizer = TextNormalizer()
        
        result = normalizer.normalize("")
        assert result == ""
        
        result = normalizer.normalize("   ")
        assert result == ""

    def test_normalize_with_different_languages(self) -> None:
        """Test that normalize works with different language codes."""
        normalizer = TextNormalizer()
        
        # English
        text = "This   is   English"
        result = normalizer.normalize(text, "en")
        assert result == "This is English"
        
        # Japanese (language param currently unused but should not fail)
        text = "これは   日本語   です"
        result = normalizer.normalize(text, "ja")
        assert result == "これは 日本語 です"