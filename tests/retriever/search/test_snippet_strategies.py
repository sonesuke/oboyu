"""Tests for snippet strategies functionality."""

import pytest

from oboyu.retriever.search.context_provider import ContextProvider
from oboyu.retriever.search.japanese_snippet_processor import JapaneseSnippetProcessor
from oboyu.retriever.search.snippet_types import SnippetConfig, SnippetMatch, SnippetStrategy
from oboyu.retriever.search.snippet_strategies import (
    FixedLengthStrategy,
    ParagraphBoundaryStrategy,
    SentenceBoundaryStrategy,
)


class TestSnippetStrategies:
    """Test snippet strategy implementations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context_provider = ContextProvider()
        self.japanese_processor = JapaneseSnippetProcessor()
        self.config = SnippetConfig(length=100, prefer_complete_sentences=True)

    def test_fixed_length_strategy_with_matches(self):
        """Test fixed length strategy with matches."""
        strategy = FixedLengthStrategy(self.context_provider, self.japanese_processor)
        content = "This is a long document about machine learning and artificial intelligence."
        matches = [SnippetMatch(start=30, end=46, text="machine learning", score=1.0)]
        
        result = strategy.process(content, matches, self.config)
        
        assert len(result) <= self.config.length
        assert "machine learning" in result

    def test_fixed_length_strategy_no_matches(self):
        """Test fixed length strategy without matches."""
        strategy = FixedLengthStrategy(self.context_provider, self.japanese_processor)
        content = "This is a long document that doesn't contain the search terms."
        matches = []
        
        result = strategy.process(content, matches, self.config)
        
        assert len(result) <= self.config.length
        assert result.startswith("This is a long document")

    def test_sentence_boundary_strategy(self):
        """Test sentence boundary strategy."""
        strategy = SentenceBoundaryStrategy(self.context_provider, self.japanese_processor)
        content = "First sentence about something. Second sentence about machine learning. Third sentence."
        matches = [SnippetMatch(start=47, end=63, text="machine learning", score=1.0)]
        
        result = strategy.process(content, matches, self.config)
        
        # Should end with a complete sentence
        assert result.endswith('.') or result.endswith('。')
        assert "machine learning" in result

    def test_sentence_boundary_strategy_japanese(self):
        """Test sentence boundary strategy with Japanese text."""
        strategy = SentenceBoundaryStrategy(self.context_provider, self.japanese_processor)
        content = "最初の文章です。機械学習について説明します。三番目の文章です。"
        matches = [SnippetMatch(start=8, end=12, text="機械学習", score=1.0)]
        config = SnippetConfig(length=100, japanese_aware=True, prefer_complete_sentences=True)
        
        result = strategy.process(content, matches, config)
        
        assert result.endswith('。')
        assert "機械学習" in result

    def test_paragraph_boundary_strategy(self):
        """Test paragraph boundary strategy."""
        strategy = ParagraphBoundaryStrategy(self.context_provider, self.japanese_processor)
        content = "First paragraph with machine learning content.\n\nSecond paragraph continues.\n\nThird paragraph."
        matches = [SnippetMatch(start=20, end=36, text="machine learning", score=1.0)]
        
        result = strategy.process(content, matches, self.config)
        
        assert "machine learning" in result
        # Should respect paragraph boundaries

    def test_paragraph_boundary_fallback_to_sentence(self):
        """Test paragraph boundary strategy fallback to sentence boundaries."""
        strategy = ParagraphBoundaryStrategy(self.context_provider, self.japanese_processor)
        # Content without paragraph breaks
        content = "First sentence with machine learning. Second sentence continues. Third sentence."
        matches = [SnippetMatch(start=20, end=36, text="machine learning", score=1.0)]
        
        result = strategy.process(content, matches, self.config)
        
        assert "machine learning" in result
        assert result.endswith('.')  # Should end with sentence boundary

    def test_sentence_boundary_without_complete_sentences(self):
        """Test sentence boundary strategy without preferring complete sentences."""
        strategy = SentenceBoundaryStrategy(self.context_provider, self.japanese_processor)
        content = "First sentence about machine learning algorithms and their applications."
        matches = [SnippetMatch(start=21, end=37, text="machine learning", score=1.0)]
        config = SnippetConfig(length=50, prefer_complete_sentences=False)
        
        result = strategy.process(content, matches, config)
        
        assert len(result) <= 50
        assert "machine learning" in result

    def test_sentence_boundary_short_adjustment(self):
        """Test sentence boundary strategy when adjustment makes snippet too short."""
        strategy = SentenceBoundaryStrategy(self.context_provider, self.japanese_processor)
        content = "Machine learning. Very short."
        matches = [SnippetMatch(start=0, end=16, text="Machine learning", score=1.0)]
        config = SnippetConfig(length=100, prefer_complete_sentences=True)
        
        result = strategy.process(content, matches, config)
        
        # Should fall back to fixed length if adjustment makes it too short
        assert "Machine learning" in result

    def test_paragraph_boundary_short_adjustment(self):
        """Test paragraph boundary strategy when adjustment makes snippet too short."""
        strategy = ParagraphBoundaryStrategy(self.context_provider, self.japanese_processor)
        content = "Machine learning.\n\nShort."
        matches = [SnippetMatch(start=0, end=16, text="Machine learning", score=1.0)]
        config = SnippetConfig(length=100, prefer_complete_sentences=True)
        
        result = strategy.process(content, matches, config)
        
        # Should fall back to sentence strategy if adjustment makes it too short
        assert "Machine learning" in result

    def test_avoid_word_breaks(self):
        """Test avoiding word breaks in sentence boundary strategy."""
        strategy = SentenceBoundaryStrategy(self.context_provider, self.japanese_processor)
        content = "This is a document about machine learning algorithms and other topics"
        matches = [SnippetMatch(start=25, end=41, text="machine learning", score=1.0)]
        config = SnippetConfig(length=45, prefer_complete_sentences=True)
        
        result = strategy.process(content, matches, config)
        
        # Should avoid cutting words in the middle
        assert not result.endswith("algorith")  # Should not cut "algorithms"

    def test_empty_content(self):
        """Test strategies with empty content."""
        strategies = [
            FixedLengthStrategy(self.context_provider, self.japanese_processor),
            SentenceBoundaryStrategy(self.context_provider, self.japanese_processor),
            ParagraphBoundaryStrategy(self.context_provider, self.japanese_processor),
        ]
        
        for strategy in strategies:
            result = strategy.process("", [], self.config)
            assert result == ""

    def test_multiple_matches(self):
        """Test strategies with multiple matches."""
        strategy = FixedLengthStrategy(self.context_provider, self.japanese_processor)
        content = "This document discusses machine learning and artificial intelligence topics."
        matches = [
            SnippetMatch(start=24, end=40, text="machine learning", score=1.0),
            SnippetMatch(start=45, end=67, text="artificial intelligence", score=0.8),
        ]
        
        result = strategy.process(content, matches, self.config)
        
        # Should center around the highest scoring match
        assert "machine learning" in result
        assert len(result) <= self.config.length