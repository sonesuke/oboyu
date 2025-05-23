"""Tests for the Japanese text processing functionality."""

import pytest

from oboyu.crawler.japanese import (
    detect_encoding,
    process_japanese_text,
    _normalize_japanese,
    _standardize_line_endings,
)


class TestJapaneseProcessing:
    """Test cases for Japanese text processing."""

    def test_detect_encoding(self) -> None:
        """Test encoding detection for Japanese text."""
        # UTF-8 text with Japanese characters
        utf8_text = "これは日本語のテキストです。"
        assert detect_encoding(utf8_text, ["utf-8", "shift-jis", "euc-jp"]) == "utf-8"

        # Text with replacement characters (indicating encoding issues)
        text_with_issues = "This text has replacement characters: \ufffd\ufffd"
        # Since we're using various encoding detection libraries, normalize the result for comparison
        result = detect_encoding(text_with_issues, ["utf-8", "shift-jis", "euc-jp"])
        # Normalize result by removing underscores and dashes, and converting to lowercase
        normalized_result = result.lower().replace('_', '-')
        assert normalized_result in ["utf-8", "shift-jis", "euc-jp", "ascii", "windows-1252", "latin-1"]

    def test_process_japanese_text(self) -> None:
        """Test processing of Japanese text."""
        # Text with full-width characters and other issues
        text = "１２３４５ ＡＢＣ　テスト\r\nテスト"
        processed = process_japanese_text(text, "utf-8")

        # Should normalize numbers and alphabet, and fix line endings
        assert "12345" in processed  # Half-width numbers
        assert "ABC" in processed    # Half-width alphabet
        assert "　テスト" not in processed  # Full-width space replaced
        assert "\r\n" not in processed  # Windows line endings replaced

    def test_normalize_japanese(self) -> None:
        """Test normalization of Japanese text."""
        # Test with full-width characters
        text = "ＡＢＣ　１２３"
        normalized = _normalize_japanese(text)

        # Should convert full-width to half-width
        assert normalized == "ABC 123"

        # Test with Japanese symbols
        text = "テスト～テスト"
        normalized = _normalize_japanese(text)

        # Should normalize Japanese symbols
        assert normalized == "テスト〜テスト"

    def test_standardize_line_endings(self) -> None:
        """Test standardization of line endings."""
        # Test with various line endings
        text = "Line1\r\nLine2\rLine3\nLine4\r\n\r\n\r\nLine5"
        standardized = _standardize_line_endings(text)

        # Should convert all line endings to Unix style
        assert "\r\n" not in standardized
        assert "\r" not in standardized

        # Should normalize multiple consecutive newlines
        assert "\n\n\n" not in standardized

        # Final result should have consistent line endings
        expected = "Line1\nLine2\nLine3\nLine4\n\nLine5"
        assert standardized == expected