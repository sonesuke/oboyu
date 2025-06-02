"""Tests for the Japanese text processing functionality."""

import pytest

from oboyu.crawler.japanese import standardize_line_endings
from oboyu.crawler.services.encoding_detector import EncodingDetector


class TestJapaneseProcessing:
    """Test cases for Japanese text processing."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.encoding_detector = EncodingDetector()

    def test_detect_encoding(self) -> None:
        """Test encoding detection for Japanese text."""
        detector = EncodingDetector()
        
        # UTF-8 text with Japanese characters
        utf8_text = "これは日本語のテキストです。"
        assert detector.detect_encoding(utf8_text, ["utf-8", "shift-jis", "euc-jp"]) == "utf-8"

        # Text with replacement characters (indicating encoding issues)
        text_with_issues = "This text has replacement characters: \ufffd\ufffd"
        # Since we're using various encoding detection libraries, normalize the result for comparison
        result = detector.detect_encoding(text_with_issues, ["utf-8", "shift-jis", "euc-jp"])
        # Normalize result by removing underscores and dashes, and converting to lowercase
        normalized_result = result.lower().replace('_', '-')
        assert normalized_result in ["utf-8", "shift-jis", "euc-jp", "ascii", "windows-1252", "latin-1"]

    def test_process_japanese_text(self) -> None:
        """Test processing of Japanese text."""
        detector = EncodingDetector()
        
        # Text with full-width characters and other issues
        text = "１２３４５ ＡＢＣ　テスト\r\nテスト"
        processed = detector.process_japanese_text(text, "utf-8")

        # Should normalize numbers and alphabet, and fix line endings
        assert "12345" in processed  # Half-width numbers
        assert "ABC" in processed    # Half-width alphabet
        assert "　テスト" not in processed  # Full-width space replaced
        assert "\r\n" not in processed  # Windows line endings replaced

    def test_normalize_in_process_japanese_text(self) -> None:
        """Test normalization within process_japanese_text."""
        detector = EncodingDetector()
        
        # Test with full-width characters
        text = "ＡＢＣ　１２３"
        processed = detector.process_japanese_text(text, "utf-8")

        # Should convert full-width to half-width
        assert "ABC 123" in processed

        # Test with Japanese symbols
        # Note: neologdn removes wave dash/tilde characters
        text = "テスト～テスト"
        processed = detector.process_japanese_text(text, "utf-8")

        # neologdn removes wave dash characters entirely
        assert "テストテスト" in processed

    def test_standardize_line_endings(self) -> None:
        """Test standardization of line endings."""
        # Test with various line endings
        text = "Line1\r\nLine2\rLine3\nLine4\r\n\r\n\r\nLine5"
        standardized = standardize_line_endings(text)

        # Should convert all line endings to Unix style
        assert "\r\n" not in standardized
        assert "\r" not in standardized

        # Should normalize multiple consecutive newlines
        assert "\n\n\n" not in standardized

        # Final result should have consistent line endings
        expected = "Line1\nLine2\nLine3\nLine4\n\nLine5"
        assert standardized == expected

    def test_ftfy_mojibake_fixing(self) -> None:
        """Test that ftfy correctly fixes mojibake issues."""
        detector = EncodingDetector()
        
        # Common mojibake pattern
        mojibake_text = "ãƒ†ã‚¹ãƒˆ"  # This is "テスト" mis-encoded
        processed = detector.process_japanese_text(mojibake_text, "utf-8")
        assert "テスト" in processed

    def test_neologdn_normalization(self) -> None:
        """Test comprehensive neologdn normalization features."""
        detector = EncodingDetector()
        
        # Test repeated characters normalization
        text = "すごーーーい"
        processed = detector.process_japanese_text(text, "utf-8")
        # neologdn reduces repeated characters but keeps at least one
        assert "すごーい" in processed
        assert "すごーーーい" not in processed  # But reduces multiple repeats

        # Test space normalization
        text = "これ　　は　テスト"  # Multiple spaces
        processed = detector.process_japanese_text(text, "utf-8")
        # neologdn removes extra spaces
        assert "  " not in processed

    def test_mojimoji_width_conversion(self) -> None:
        """Test mojimoji width conversion for mixed content."""
        detector = EncodingDetector()
        
        # Mixed full-width and half-width
        text = "ＰＹＴＨＯＮ３．１３とひらがな"
        processed = detector.process_japanese_text(text, "utf-8")
        assert "PYTHON3.13" in processed
        assert "ひらがな" in processed  # Japanese should remain unchanged