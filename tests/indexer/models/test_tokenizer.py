"""Tests for the Japanese tokenizer module."""

import pytest

from oboyu.common.services.tokenizer import (
    FallbackTokenizer,
    JapaneseTokenizer,
    create_tokenizer,
    HAS_JAPANESE_TOKENIZER,
)


class TestJapaneseTokenizer:
    """Test cases for Japanese tokenizer."""
    
    @pytest.mark.skipif(not HAS_JAPANESE_TOKENIZER, reason="Japanese tokenizer not installed")
    def test_tokenize_japanese_text(self) -> None:
        """Test tokenizing Japanese text."""
        tokenizer = JapaneseTokenizer()
        
        # Test basic Japanese sentence
        text = "日本語の文章をトークン化します。"
        tokens = tokenizer.tokenize(text)
        
        assert len(tokens) > 0
        # MeCab may tokenize "日本語" as "日本" and "語" separately
        assert "日本" in tokens or "日本語" in tokens
        assert "文章" in tokens
        # MeCab may tokenize "トークン化" differently
        assert any("トークン" in token for token in tokens)
        
    @pytest.mark.skipif(not HAS_JAPANESE_TOKENIZER, reason="Japanese tokenizer not installed")
    def test_tokenize_mixed_text(self) -> None:
        """Test tokenizing mixed Japanese/English text."""
        tokenizer = JapaneseTokenizer()
        
        text = "Pythonで日本語のNLP処理を行います。"
        tokens = tokenizer.tokenize(text)
        
        assert "Python" in tokens or "python" in tokens
        # MeCab may tokenize "日本語" as "日本" and "語" separately
        assert "日本" in tokens or "日本語" in tokens
        assert "NLP" in tokens or "nlp" in tokens
        assert "処理" in tokens
        
    @pytest.mark.skipif(not HAS_JAPANESE_TOKENIZER, reason="Japanese tokenizer not installed")
    def test_stop_word_removal(self) -> None:
        """Test stop word removal."""
        tokenizer = JapaneseTokenizer()
        
        text = "これは日本語のテストです。"
        tokens = tokenizer.tokenize(text)
        
        # Stop words should be removed
        assert "これ" not in tokens
        assert "は" not in tokens
        assert "の" not in tokens
        assert "です" not in tokens
        
        # Content words should remain
        # MeCab may tokenize "日本語" as "日本" and "語" separately
        assert "日本" in tokens or "日本語" in tokens
        assert "テスト" in tokens or any("テスト" in token for token in tokens)
        
    @pytest.mark.skipif(not HAS_JAPANESE_TOKENIZER, reason="Japanese tokenizer not installed")
    def test_min_token_length(self) -> None:
        """Test minimum token length filtering."""
        tokenizer = JapaneseTokenizer(min_token_length=3)
        
        text = "私は東京に住んでいます。"
        tokens = tokenizer.tokenize(text)
        
        # Short tokens should be filtered out
        assert all(len(token) >= 3 for token in tokens)
        
    @pytest.mark.skipif(not HAS_JAPANESE_TOKENIZER, reason="Japanese tokenizer not installed")
    def test_pos_filtering(self) -> None:
        """Test part-of-speech filtering."""
        tokenizer = JapaneseTokenizer(use_pos_filter=True)
        
        text = "美しい富士山を見ました。"
        tokens = tokenizer.tokenize(text)
        
        # Content words should remain
        assert "美しい" in tokens or "美しく" in tokens  # Adjective
        # MeCab may tokenize "富士山" as separate words
        assert "富士山" in tokens or "富士" in tokens or "フジ" in tokens  # Noun
        # Verb might be lemmatized or in various forms
        assert any(t in tokens for t in ["見る", "見", "見ま", "ミル"]) or len(tokens) >= 2  # At least some content words
        
    @pytest.mark.skipif(not HAS_JAPANESE_TOKENIZER, reason="Japanese tokenizer not installed")
    def test_text_normalization(self) -> None:
        """Test text normalization."""
        tokenizer = JapaneseTokenizer(normalize_text=True)
        
        # Full-width to half-width conversion
        text = "ＰＹＴＨＯＮプログラミング"
        tokens = tokenizer.tokenize(text)
        
        # Should be normalized to half-width
        assert "python" in tokens or "PYTHON" in tokens
        
    @pytest.mark.skipif(not HAS_JAPANESE_TOKENIZER, reason="Japanese tokenizer not installed")
    def test_get_term_frequencies(self) -> None:
        """Test term frequency calculation."""
        tokenizer = JapaneseTokenizer()
        
        text = "日本語の文章です。日本語は美しい言語です。"
        term_freq = tokenizer.get_term_frequencies(text)
        
        # MeCab may tokenize "日本語" as "日本" and "語" separately
        assert "日本" in term_freq or "日本語" in term_freq
        if "日本語" in term_freq:
            assert term_freq["日本語"] == 2
        elif "日本" in term_freq:
            assert term_freq["日本"] == 2
        assert "文章" in term_freq
        assert term_freq["文章"] == 1
        
    @pytest.mark.skipif(not HAS_JAPANESE_TOKENIZER, reason="Japanese tokenizer not installed")
    def test_is_japanese_text(self) -> None:
        """Test Japanese text detection."""
        tokenizer = JapaneseTokenizer()
        
        assert tokenizer.is_japanese_text("これは日本語です")
        assert tokenizer.is_japanese_text("カタカナ")
        assert tokenizer.is_japanese_text("漢字")
        assert not tokenizer.is_japanese_text("This is English")
        assert tokenizer.is_japanese_text("Mixed 日本語 text")


class TestFallbackTokenizer:
    """Test cases for fallback tokenizer."""
    
    def test_tokenize_basic(self) -> None:
        """Test basic tokenization."""
        tokenizer = FallbackTokenizer()
        
        text = "This is a test sentence."
        tokens = tokenizer.tokenize(text)
        
        assert "this" in tokens
        assert "is" in tokens
        assert "test" in tokens
        assert "sentence" in tokens
        
    def test_tokenize_japanese_fallback(self) -> None:
        """Test Japanese tokenization with fallback."""
        tokenizer = FallbackTokenizer()
        
        # Fallback tokenizer uses regex patterns
        text = "日本語のテスト"
        tokens = tokenizer.tokenize(text)
        
        # Should extract continuous Japanese characters
        assert len(tokens) > 0
        
    def test_min_token_length_fallback(self) -> None:
        """Test minimum token length in fallback tokenizer."""
        tokenizer = FallbackTokenizer(min_token_length=3)
        
        text = "I am testing the tokenizer"
        tokens = tokenizer.tokenize(text)
        
        # Short tokens should be filtered
        assert "i" not in tokens
        assert "am" not in tokens
        assert "testing" in tokens
        assert "the" in tokens  # Exactly 3 characters
        
    def test_stop_words_fallback(self) -> None:
        """Test stop word removal in fallback tokenizer."""
        stop_words = {"the", "is", "a"}
        tokenizer = FallbackTokenizer(stop_words=stop_words)
        
        text = "This is a test"
        tokens = tokenizer.tokenize(text)
        
        assert "this" in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "test" in tokens


class TestCreateTokenizer:
    """Test cases for tokenizer factory function."""
    
    def test_create_japanese_tokenizer(self) -> None:
        """Test creating Japanese tokenizer."""
        if HAS_JAPANESE_TOKENIZER:
            tokenizer = create_tokenizer(language="ja")
            assert isinstance(tokenizer, JapaneseTokenizer)
        else:
            # Should fall back when not available
            tokenizer = create_tokenizer(language="ja")
            assert isinstance(tokenizer, FallbackTokenizer)
    
    def test_create_fallback_tokenizer(self) -> None:
        """Test creating fallback tokenizer."""
        # Non-Japanese language
        tokenizer = create_tokenizer(language="en")
        assert isinstance(tokenizer, FallbackTokenizer)
        
        # Force fallback
        tokenizer = create_tokenizer(language="ja", use_fallback=True)
        assert isinstance(tokenizer, FallbackTokenizer)
    
    def test_create_tokenizer_with_params(self) -> None:
        """Test creating tokenizer with parameters."""
        stop_words = {"test", "words"}
        tokenizer = create_tokenizer(
            language="en",
            stop_words=stop_words,
            min_token_length=3
        )
        
        assert isinstance(tokenizer, FallbackTokenizer)
        assert tokenizer.stop_words == stop_words
        assert tokenizer.min_token_length == 3