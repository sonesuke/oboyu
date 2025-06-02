"""Tests for Japanese snippet processor functionality."""

import pytest

from oboyu.retriever.search.japanese_snippet_processor import JapaneseSnippetProcessor


class TestJapaneseSnippetProcessor:
    """Test Japanese snippet processing functionality."""

    def test_find_sentence_boundaries(self):
        """Test finding Japanese sentence boundaries."""
        processor = JapaneseSnippetProcessor()
        text = "最初の文章です。二番目の文章です。三番目の文章です。"
        
        boundaries = processor.find_sentence_boundaries(text)
        
        assert len(boundaries) == 3
        # Check that boundaries are at the end of each sentence (after 。)
        for boundary in boundaries:
            assert text[boundary-1] == "。"

    def test_find_paragraph_boundaries(self):
        """Test finding Japanese paragraph boundaries."""
        processor = JapaneseSnippetProcessor()
        text = "最初の段落です。\n\n二番目の段落です。\n\n三番目の段落です。"
        
        boundaries = processor.find_paragraph_boundaries(text)
        
        assert len(boundaries) == 2
        # Should find positions of paragraph breaks

    def test_is_japanese_text_hiragana(self):
        """Test Japanese text detection with hiragana."""
        processor = JapaneseSnippetProcessor()
        
        assert processor.is_japanese_text("これは日本語です") is True
        assert processor.is_japanese_text("This is English") is False

    def test_is_japanese_text_katakana(self):
        """Test Japanese text detection with katakana."""
        processor = JapaneseSnippetProcessor()
        
        assert processor.is_japanese_text("コンピュータ") is True
        assert processor.is_japanese_text("COMPUTER") is False

    def test_is_japanese_text_kanji(self):
        """Test Japanese text detection with kanji."""
        processor = JapaneseSnippetProcessor()
        
        assert processor.is_japanese_text("機械学習") is True
        assert processor.is_japanese_text("machine learning") is False

    def test_is_japanese_text_mixed(self):
        """Test Japanese text detection with mixed content."""
        processor = JapaneseSnippetProcessor()
        
        assert processor.is_japanese_text("This is 日本語 text") is True
        assert processor.is_japanese_text("123 456 789") is False

    def test_normalize_japanese_text(self):
        """Test Japanese text normalization."""
        processor = JapaneseSnippetProcessor()
        
        # Test punctuation normalization
        text = "これは文章です 。次の文章です。 "
        normalized = processor.normalize_japanese_text(text)
        
        # The normalization should remove spaces before punctuation and add space after
        assert "です。" in normalized
        assert "です。次" in normalized

    def test_adjust_to_sentence_boundaries(self):
        """Test adjusting text to Japanese sentence boundaries."""
        processor = JapaneseSnippetProcessor()
        text = "最初の文章です。二番目の文章です。三番目の"
        
        adjusted = processor.adjust_to_sentence_boundaries(text, prefer_complete=True)
        
        assert adjusted == "最初の文章です。二番目の文章です。"

    def test_adjust_to_sentence_boundaries_no_complete(self):
        """Test adjusting text when not preferring complete sentences."""
        processor = JapaneseSnippetProcessor()
        text = "最初の文章です。二番目の文章です。三番目の"
        
        adjusted = processor.adjust_to_sentence_boundaries(text, prefer_complete=False)
        
        assert adjusted == text  # Should return unchanged

    def test_adjust_to_paragraph_boundaries(self):
        """Test adjusting text to Japanese paragraph boundaries."""
        processor = JapaneseSnippetProcessor()
        text = "最初の段落です。\n\n二番目の段落です。\n\n三番目の"
        
        adjusted = processor.adjust_to_paragraph_boundaries(text, prefer_complete=True)
        
        # Should stop at the last complete paragraph
        assert "三番目の" not in adjusted
        assert "二番目の段落です。" in adjusted

    def test_get_character_density(self):
        """Test getting character density analysis."""
        processor = JapaneseSnippetProcessor()
        
        # Test pure hiragana
        text = "これはひらがなです"
        density = processor.get_character_density(text)
        assert density["hiragana"] > 0.8
        assert density["katakana"] == 0.0
        assert density["kanji"] == 0.0
        
        # Test mixed content
        text = "これはTest文章です"  # Mixed hiragana, kanji, and English
        density = processor.get_character_density(text)
        assert density["hiragana"] > 0
        assert density["kanji"] > 0
        assert density["other"] > 0

    def test_avoid_word_breaks_japanese(self):
        """Test avoiding word breaks in Japanese text."""
        processor = JapaneseSnippetProcessor()
        
        # Test with trailing punctuation
        text = "これは文章です、。"
        adjusted = processor.avoid_word_breaks_japanese(text)
        
        assert adjusted == "これは文章です"  # Should remove incomplete punctuation

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        processor = JapaneseSnippetProcessor()
        
        assert processor.find_sentence_boundaries("") == []
        assert processor.find_paragraph_boundaries("") == []
        assert processor.is_japanese_text("") is False
        assert processor.normalize_japanese_text("") == ""
        assert processor.adjust_to_sentence_boundaries("") == ""
        assert processor.avoid_word_breaks_japanese("") == ""

    def test_character_density_empty(self):
        """Test character density with empty text."""
        processor = JapaneseSnippetProcessor()
        
        density = processor.get_character_density("")
        assert density["hiragana"] == 0.0
        assert density["katakana"] == 0.0
        assert density["kanji"] == 0.0
        assert density["other"] == 1.0

    def test_sentence_endings_variations(self):
        """Test various Japanese sentence ending patterns."""
        processor = JapaneseSnippetProcessor()
        
        # Test different ending punctuation
        text = "質問です？感嘆です！普通です。"
        boundaries = processor.find_sentence_boundaries(text)
        
        assert len(boundaries) == 3
        # Should find all three different endings

    def test_fallback_to_sentence_boundaries(self):
        """Test paragraph boundary adjustment fallback."""
        processor = JapaneseSnippetProcessor()
        # Text with no paragraph breaks, should fallback to sentence boundaries
        text = "最初の文章です。二番目の文章です。三番目の"
        
        adjusted = processor.adjust_to_paragraph_boundaries(text, prefer_complete=True)
        
        # Should fallback to sentence boundary adjustment
        assert adjusted == "最初の文章です。二番目の文章です。"