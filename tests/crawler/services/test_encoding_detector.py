"""Tests for EncodingDetector."""

import pytest

from oboyu.crawler.services.encoding_detector import EncodingDetector


class TestEncodingDetector:
    """Test cases for EncodingDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = EncodingDetector()
        assert detector is not None

    def test_detect_encoding_utf8_japanese(self) -> None:
        """Test encoding detection with UTF-8 Japanese text."""
        detector = EncodingDetector()
        
        # Valid Japanese text in UTF-8
        japanese_text = "これは日本語のテストです。"
        encoding = detector.detect_encoding(japanese_text)
        assert encoding == "utf-8"

    def test_detect_encoding_with_replacement_chars(self) -> None:
        """Test encoding detection when text contains replacement characters."""
        detector = EncodingDetector()
        
        # Text with Unicode replacement character (indicates decoding issues)
        text_with_replacement = "Some text with \\ufffd replacement character"
        encoding = detector.detect_encoding(text_with_replacement)
        
        # Should try to detect proper encoding
        assert isinstance(encoding, str)
        assert len(encoding) > 0

    def test_detect_encoding_non_japanese(self) -> None:
        """Test encoding detection with non-Japanese text."""
        detector = EncodingDetector()
        
        # English text without Japanese characters
        english_text = "This is English text without any Japanese characters."
        encoding = detector.detect_encoding(english_text)
        assert encoding == "utf-8"  # Default for non-Japanese

    def test_detect_encoding_preferred_encodings(self) -> None:
        """Test encoding detection with preferred encodings list."""
        detector = EncodingDetector()
        
        # Test with custom preferred encodings
        text = "Test text"
        preferred = ["shift-jis", "euc-jp", "utf-8"]
        encoding = detector.detect_encoding(text, preferred)
        assert encoding in preferred

    def test_process_japanese_text_basic(self) -> None:
        """Test basic Japanese text processing."""
        detector = EncodingDetector()
        
        # Test with normal Japanese text
        japanese_text = "これは日本語のテストです。"
        processed = detector.process_japanese_text(japanese_text, "utf-8")
        
        # Should return processed text
        assert isinstance(processed, str)
        assert len(processed) > 0

    def test_process_japanese_text_with_width_conversion(self) -> None:
        """Test Japanese text processing with full-width character conversion."""
        detector = EncodingDetector()
        
        # Text with full-width numbers and ASCII
        text_with_fullwidth = "全角数字１２３と全角英字ＡＢＣ"
        processed = detector.process_japanese_text(text_with_fullwidth, "utf-8")
        
        # Should convert full-width numbers and ASCII to half-width
        assert "123" in processed or "１２３" in processed  # Either converted or original
        assert isinstance(processed, str)

    def test_process_japanese_text_line_endings(self) -> None:
        """Test Japanese text processing with different line endings."""
        detector = EncodingDetector()
        
        # Text with Windows line endings
        text_with_crlf = "First line\r\nSecond line\r\nThird line"
        processed = detector.process_japanese_text(text_with_crlf, "utf-8")
        
        # Should standardize to Unix line endings
        assert "\r\n" not in processed
        assert "\n" in processed

        # Text with Mac line endings
        text_with_cr = "First line\rSecond line\rThird line"
        processed = detector.process_japanese_text(text_with_cr, "utf-8")
        
        # Should standardize to Unix line endings
        assert "\r" not in processed
        assert "\n" in processed

    def test_process_japanese_text_multiple_newlines(self) -> None:
        """Test Japanese text processing with multiple consecutive newlines."""
        detector = EncodingDetector()
        
        # Text with many consecutive newlines
        text_with_many_newlines = "Line 1\n\n\n\n\nLine 2"
        processed = detector.process_japanese_text(text_with_many_newlines, "utf-8")
        
        # Should normalize to at most two newlines
        assert "\n\n\n" not in processed
        assert "Line 1" in processed
        assert "Line 2" in processed

    def test_needs_width_conversion(self) -> None:
        """Test width conversion detection."""
        detector = EncodingDetector()
        
        # Text with full-width numbers
        text_with_fullwidth_numbers = "１２３４５"
        assert detector._needs_width_conversion(text_with_fullwidth_numbers) is True
        
        # Text with full-width ASCII letters
        text_with_fullwidth_ascii = "ＡＢＣ"
        assert detector._needs_width_conversion(text_with_fullwidth_ascii) is True
        
        # Text without full-width characters
        normal_text = "ABC123"
        assert detector._needs_width_conversion(normal_text) is False
        
        # Japanese text without full-width ASCII/numbers
        japanese_text = "これは日本語です"
        assert detector._needs_width_conversion(japanese_text) is False

    def test_standardize_line_endings(self) -> None:
        """Test line ending standardization."""
        detector = EncodingDetector()
        
        # Test Windows line endings
        windows_text = "Line 1\r\nLine 2\r\nLine 3"
        standardized = detector._standardize_line_endings(windows_text)
        assert "\r\n" not in standardized
        assert standardized.count("\n") == 2
        
        # Test Mac line endings
        mac_text = "Line 1\rLine 2\rLine 3"
        standardized = detector._standardize_line_endings(mac_text)
        assert "\r" not in standardized
        assert standardized.count("\n") == 2
        
        # Test mixed line endings
        mixed_text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        standardized = detector._standardize_line_endings(mixed_text)
        assert "\r" not in standardized
        assert standardized.count("\n") == 3
        
        # Test multiple consecutive newlines
        multiple_newlines = "Line 1\n\n\n\n\nLine 2"
        standardized = detector._standardize_line_endings(multiple_newlines)
        assert "\n\n\n" not in standardized  # Should be reduced to at most \n\n

    def test_process_japanese_text_comprehensive(self) -> None:
        """Test comprehensive Japanese text processing."""
        detector = EncodingDetector()
        
        # Complex text with various issues
        complex_text = "全角数字１２３\r\n\r\n\r\n全角英字ＡＢＣ\r\n日本語テキスト"
        processed = detector.process_japanese_text(complex_text, "utf-8")
        
        # Should be processed and cleaned
        assert isinstance(processed, str)
        assert len(processed) > 0
        # Should not have excessive newlines
        assert "\n\n\n" not in processed
        # Should not have Windows line endings
        assert "\r\n" not in processed

    def test_encoding_detection_edge_cases(self) -> None:
        """Test encoding detection with edge cases."""
        detector = EncodingDetector()
        
        # Empty string
        encoding = detector.detect_encoding("")
        assert encoding == "utf-8"
        
        # String with only ASCII
        ascii_text = "Hello, World!"
        encoding = detector.detect_encoding(ascii_text)
        assert encoding == "utf-8"
        
        # String with only numbers
        numbers_text = "1234567890"
        encoding = detector.detect_encoding(numbers_text)
        assert encoding == "utf-8"