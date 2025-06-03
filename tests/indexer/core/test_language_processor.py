"""Tests for the LanguageProcessor component."""

import pytest
from unittest.mock import patch, MagicMock

from oboyu.indexer.core.language_processor import LanguageProcessor


class TestLanguageProcessor:
    """Test cases for LanguageProcessor."""

    def test_prepare_text_english(self) -> None:
        """Test that English text is returned as-is."""
        processor = LanguageProcessor()
        
        text = "This is English text."
        result = processor.prepare_text(text, "en")
        assert result == text

    def test_prepare_text_unknown_language(self) -> None:
        """Test that unknown language text is returned as-is."""
        processor = LanguageProcessor()
        
        text = "Some text in unknown language"
        result = processor.prepare_text(text, "unknown")
        assert result == text

    @patch("oboyu.crawler.services.encoding_detector.EncodingDetector.process_japanese_text")
    def test_prepare_text_japanese(self, mock_process_japanese: MagicMock) -> None:
        """Test that Japanese text is processed correctly."""
        processor = LanguageProcessor()
        
        # Set up mock
        japanese_text = "これは日本語のテキストです。"
        processed_text = "これは日本語のテキストです。"  # Processed version
        mock_process_japanese.return_value = processed_text
        
        # Test Japanese processing
        result = processor.prepare_text(japanese_text, "ja")
        
        # Verify the Japanese processor was called
        mock_process_japanese.assert_called_once_with(japanese_text, "utf-8")
        assert result == processed_text

    def test_prepare_text_empty_string(self) -> None:
        """Test handling of empty strings."""
        processor = LanguageProcessor()
        
        result = processor.prepare_text("", "en")
        assert result == ""
        
        result = processor.prepare_text("", "ja")
        assert result == ""

    def test_prepare_text_various_languages(self) -> None:
        """Test that various non-Japanese languages are handled."""
        processor = LanguageProcessor()
        
        languages = ["fr", "de", "es", "zh", "ko", "ru"]
        text = "Sample text"
        
        for lang in languages:
            result = processor.prepare_text(text, lang)
            # Non-Japanese languages should return text as-is
            assert result == text