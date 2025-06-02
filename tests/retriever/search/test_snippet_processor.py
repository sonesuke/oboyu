"""Tests for snippet processor functionality."""

import pytest

from oboyu.retriever.search.snippet_processor import SnippetProcessor
from oboyu.retriever.search.snippet_types import SnippetConfig, SnippetStrategy


class TestSnippetProcessor:
    """Test snippet processing functionality."""

    def test_basic_snippet_generation(self):
        """Test basic snippet generation."""
        config = SnippetConfig(length=100, highlight_matches=False)
        processor = SnippetProcessor(config)
        
        content = "This is a long piece of text that contains important information about machine learning and AI systems."
        query = "machine learning"
        
        snippet = processor.generate_snippet(content, query)
        
        assert len(snippet) <= 100
        assert "machine learning" in snippet.lower()

    def test_japanese_text_processing(self):
        """Test snippet processing with Japanese text."""
        config = SnippetConfig(
            length=200,
            japanese_aware=True,
            strategy=SnippetStrategy.SENTENCE_BOUNDARY,
            highlight_matches=False
        )
        processor = SnippetProcessor(config)
        
        content = "これは機械学習についての重要な文書です。人工知能技術は急速に発展しています。この分野では多くの研究が行われています。"
        query = "機械学習"
        
        snippet = processor.generate_snippet(content, query)
        
        assert "機械学習" in snippet
        assert len(snippet) <= 200

    def test_highlight_matches(self):
        """Test query match highlighting."""
        config = SnippetConfig(length=100, highlight_matches=True)
        processor = SnippetProcessor(config)
        
        content = "This text contains machine learning concepts and artificial intelligence topics."
        query = "machine learning"
        
        snippet = processor.generate_snippet(content, query)
        
        # Should contain highlighted matches
        assert "**machine**" in snippet or "**learning**" in snippet

    def test_sentence_boundary_strategy(self):
        """Test sentence boundary strategy."""
        config = SnippetConfig(
            length=150,
            strategy=SnippetStrategy.SENTENCE_BOUNDARY,
            prefer_complete_sentences=True,
            highlight_matches=False
        )
        processor = SnippetProcessor(config)
        
        content = "First sentence about machine learning. Second sentence continues the topic. Third sentence adds more detail."
        query = "machine learning"
        
        snippet = processor.generate_snippet(content, query)
        
        # Should end with a complete sentence
        assert snippet.endswith('.') or snippet.endswith('。')

    def test_context_window(self):
        """Test context window around matches."""
        config = SnippetConfig(
            length=200,
            context_window=30,
            highlight_matches=False
        )
        processor = SnippetProcessor(config)
        
        content = "A" * 50 + " machine learning " + "B" * 50
        query = "machine learning"
        
        snippet = processor.generate_snippet(content, query)
        
        assert "machine learning" in snippet
        # Should contain context around the match
        assert "A" in snippet
        # Note: B might not be included if snippet is truncated to respect sentence boundaries

    def test_no_matches_fixed_length(self):
        """Test behavior when no matches found with fixed length strategy."""
        config = SnippetConfig(
            length=50,
            strategy=SnippetStrategy.FIXED_LENGTH,
            highlight_matches=False
        )
        processor = SnippetProcessor(config)
        
        content = "This is content that does not contain the search term."
        query = "nonexistent"
        
        snippet = processor.generate_snippet(content, query)
        
        assert len(snippet) <= 50
        assert snippet.startswith("This is content")

    def test_empty_content(self):
        """Test handling of empty content."""
        config = SnippetConfig(length=100)
        processor = SnippetProcessor(config)
        
        snippet = processor.generate_snippet("", "query")
        assert snippet == ""
        
        snippet = processor.generate_snippet("   ", "query")
        assert snippet == ""

    def test_snippet_config_validation(self):
        """Test snippet configuration validation."""
        # Valid config
        config = SnippetConfig(length=100)
        assert config.length == 100
        
        # Invalid length
        with pytest.raises(ValueError):
            SnippetConfig(length=0)
        
        with pytest.raises(ValueError):
            SnippetConfig(length=-1)

    def test_multi_level_snippets(self):
        """Test multi-level snippet configuration."""
        from oboyu.retriever.search.snippet_types import SnippetLevel
        
        levels = [
            SnippetLevel(type="summary", length=50),
            SnippetLevel(type="detailed", length=150),
        ]
        
        config = SnippetConfig(
            length=300,
            levels=levels,
            highlight_matches=False
        )
        processor = SnippetProcessor(config)
        
        content = "This is a comprehensive document about machine learning algorithms and their applications in various domains."
        query = "machine learning"
        
        snippet = processor.generate_snippet(content, query)
        
        # Should use the first level configuration (50 chars)
        assert len(snippet) <= 50
        # Check that it contains the query terms (may be truncated)
        content_lower = snippet.lower()
        assert "machine" in content_lower or "learning" in content_lower

    def test_japanese_sentence_boundaries(self):
        """Test Japanese sentence boundary detection."""
        config = SnippetConfig(
            length=200,
            strategy=SnippetStrategy.SENTENCE_BOUNDARY,
            japanese_aware=True,
            prefer_complete_sentences=True,
            highlight_matches=False
        )
        processor = SnippetProcessor(config)
        
        content = "最初の文章です。二番目の文章には機械学習について書かれています。三番目の文章は詳細を追加します。"
        query = "機械学習"
        
        snippet = processor.generate_snippet(content, query)
        
        # Should end with Japanese sentence ending
        assert snippet.endswith('。') or snippet.endswith('！') or snippet.endswith('？')
        assert "機械学習" in snippet