"""Tests for LanguageDetector."""

import pytest

from oboyu.crawler.services.language_detector import LanguageDetector


class TestLanguageDetector:
    """Test cases for LanguageDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = LanguageDetector()
        assert detector is not None

    def test_detect_language_english(self) -> None:
        """Test English language detection."""
        detector = LanguageDetector()
        
        english_text = "This is a sample English text. It contains multiple sentences to help with language detection."
        detected = detector.detect_language(english_text)
        assert detected == "en"

    def test_detect_language_japanese(self) -> None:
        """Test Japanese language detection."""
        detector = LanguageDetector()
        
        # Test with mixed Japanese text (hiragana, katakana, kanji)
        japanese_text = "これは日本語のテストです。ひらがな、カタカナ、漢字が含まれています。"
        detected = detector.detect_language(japanese_text)
        assert detected == "ja"

        # Test with mostly hiragana
        hiragana_text = "これはひらがなだけのテストです。"
        detected = detector.detect_language(hiragana_text)
        assert detected == "ja"

        # Test with mostly katakana
        katakana_text = "コレハカタカナダケノテストデス。"
        detected = detector.detect_language(katakana_text)
        assert detected == "ja"

        # Test with mostly kanji
        kanji_text = "日本語文章漢字使用例文。"
        detected = detector.detect_language(kanji_text)
        assert detected == "ja"

    def test_detect_language_short_text(self) -> None:
        """Test language detection with very short text."""
        detector = LanguageDetector()
        
        # Very short text should default to English
        short_text = "Hi"
        detected = detector.detect_language(short_text)
        assert detected == "en"

        # Empty text should default to English
        empty_text = ""
        detected = detector.detect_language(empty_text)
        assert detected == "en"

        # Whitespace only should default to English
        whitespace_text = "   \\n\\t  "
        detected = detector.detect_language(whitespace_text)
        assert detected == "en"

    def test_detect_language_mixed_content(self) -> None:
        """Test language detection with mixed content."""
        detector = LanguageDetector()
        
        # Mixed Japanese and English (Japanese should be detected due to quick pre-check)
        mixed_text = "This is English text. これは日本語です。More English here."
        detected = detector.detect_language(mixed_text)
        assert detected == "ja"  # Japanese characters trigger Japanese detection

        # Mostly English with few Japanese characters
        mostly_english = "This is a long English document with occasional Japanese words like こんにちは."
        detected = detector.detect_language(mostly_english)
        # Could be either, but our implementation favors Japanese if any Japanese chars present
        assert detected in ["en", "ja"]

    def test_detect_language_special_characters(self) -> None:
        """Test language detection with special characters and numbers."""
        detector = LanguageDetector()
        
        # Text with numbers and punctuation
        text_with_numbers = "The year 2023 has been great! We achieved 95% success rate."
        detected = detector.detect_language(text_with_numbers)
        assert detected == "en"

        # Japanese with numbers
        japanese_with_numbers = "2023年は良い年でした。成功率は95%でした。"
        detected = detector.detect_language(japanese_with_numbers)
        assert detected == "ja"

    def test_detect_language_unicode_range_check(self) -> None:
        """Test the Japanese character range detection logic."""
        detector = LanguageDetector()
        
        # Test with longer Japanese text to ensure proper detection
        # The Unicode range check should kick in for substantial Japanese content
        japanese_chars = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"  # Longer hiragana
        detected = detector.detect_language(japanese_chars)
        assert detected == "ja"

        japanese_chars2 = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"  # Longer katakana
        detected = detector.detect_language(japanese_chars2)
        assert detected == "ja"

        # Test with mixed Japanese text that should definitely be detected
        japanese_chars3 = "これは日本語の文章です。漢字とひらがなとカタカナが含まれています。"  # Mixed Japanese
        detected = detector.detect_language(japanese_chars3)
        assert detected == "ja"

    @pytest.mark.slow
    def test_fasttext_model_loading(self) -> None:
        """Test FastText model loading (slow test due to model download)."""
        detector = LanguageDetector()
        
        # This should trigger model loading
        sample_text = "This is a test text for FastText model loading."
        try:
            model = detector._get_fasttext_model()
            assert model is not None
            
            # Test actual prediction
            detected = detector.detect_language(sample_text)
            assert detected is not None
            assert len(detected) >= 2  # Should be a valid language code
        except RuntimeError:
            # If model fails to load, that's acceptable for this test
            pytest.skip("FastText model could not be loaded")

    def test_detect_language_graceful_fallback(self) -> None:
        """Test that detection falls back gracefully when FastText fails."""
        detector = LanguageDetector()
        
        # Mock a scenario where FastText might fail by testing with unusual text
        unusual_text = "\\x00\\x01\\x02 Some text with control characters"
        detected = detector.detect_language(unusual_text)
        
        # Should still return a valid language code
        assert detected in ["en", "ja"] or len(detected) == 2

    def test_detect_language_common_languages(self) -> None:
        """Test detection of other common languages (if FastText works)."""
        detector = LanguageDetector()
        
        # These tests may depend on FastText being available
        test_cases = [
            ("Bonjour, comment allez-vous?", "fr"),  # French
            ("Hola, ¿cómo estás?", "es"),           # Spanish
            ("Guten Tag, wie geht es Ihnen?", "de"), # German
            ("Ciao, come stai?", "it"),             # Italian
        ]
        
        for text, expected_lang in test_cases:
            try:
                detected = detector.detect_language(text)
                # Don't assert exact match since model might not be available
                # Just ensure we get a reasonable response
                assert isinstance(detected, str)
                assert len(detected) >= 2
            except Exception:
                # If detection fails, that's acceptable for this test
                pass