"""Tests for text highlighter functionality."""

import pytest

from oboyu.retriever.search.text_highlighter import TextHighlighter


class TestTextHighlighter:
    """Test text highlighting functionality."""

    def test_highlight_single_match(self):
        """Test highlighting a single match."""
        highlighter = TextHighlighter()
        text = "This is a test document about machine learning."
        matches = ["machine"]
        
        result = highlighter.highlight_matches(text, matches)
        
        assert "**machine**" in result
        assert "learning" in result  # Should not be highlighted

    def test_highlight_multiple_matches(self):
        """Test highlighting multiple matches."""
        highlighter = TextHighlighter()
        text = "Machine learning and artificial intelligence are important topics."
        matches = ["machine", "artificial"]
        
        result = highlighter.highlight_matches(text, matches)
        
        assert "**Machine**" in result
        assert "**artificial**" in result

    def test_highlight_query(self):
        """Test highlighting all terms from a query."""
        highlighter = TextHighlighter()
        text = "This document discusses machine learning algorithms."
        query = "machine learning"
        
        result = highlighter.highlight_query(text, query)
        
        assert "**machine**" in result
        assert "**learning**" in result

    def test_case_insensitive_highlighting(self):
        """Test case-insensitive highlighting."""
        highlighter = TextHighlighter()
        text = "Machine Learning and machine learning are the same."
        matches = ["machine"]
        
        result = highlighter.highlight_matches(text, matches, case_sensitive=False)
        
        # Both instances should be highlighted
        assert result.count("**Machine**") == 1
        assert result.count("**machine**") == 1

    def test_case_sensitive_highlighting(self):
        """Test case-sensitive highlighting."""
        highlighter = TextHighlighter()
        text = "Machine Learning and machine learning are the same."
        matches = ["machine"]
        
        result = highlighter.highlight_matches(text, matches, case_sensitive=True)
        
        # Only lowercase instance should be highlighted
        assert "Machine Learning" in result  # Not highlighted
        assert "**machine** learning" in result  # Highlighted

    def test_word_boundaries(self):
        """Test that highlighting respects word boundaries."""
        highlighter = TextHighlighter()
        text = "The programmer programmed a program."
        matches = ["program"]
        
        result = highlighter.highlight_matches(text, matches)
        
        # Only the standalone "program" should be highlighted
        assert "programmer" in result  # Not highlighted
        assert "programmed" in result  # Not highlighted
        assert "**program**." in result  # Highlighted

    def test_custom_highlight_format(self):
        """Test custom highlight format."""
        highlighter = TextHighlighter("<mark>{}</mark>")
        text = "This is a test document."
        matches = ["test"]
        
        result = highlighter.highlight_matches(text, matches)
        
        assert "<mark>test</mark>" in result

    def test_remove_highlights(self):
        """Test removing highlights from text."""
        highlighter = TextHighlighter()
        text = "This is a **highlighted** word in **bold**."
        
        result = highlighter.remove_highlights(text)
        
        assert result == "This is a highlighted word in bold."

    def test_set_highlight_format(self):
        """Test setting highlight format after initialization."""
        highlighter = TextHighlighter()
        highlighter.set_highlight_format("<em>{}</em>")
        
        text = "This is a test."
        matches = ["test"]
        
        result = highlighter.highlight_matches(text, matches)
        
        assert "<em>test</em>" in result

    def test_empty_matches(self):
        """Test behavior with empty matches list."""
        highlighter = TextHighlighter()
        text = "This is a test document."
        
        result = highlighter.highlight_matches(text, [])
        
        assert result == text  # Should return unchanged

    def test_empty_query(self):
        """Test behavior with empty query."""
        highlighter = TextHighlighter()
        text = "This is a test document."
        
        result = highlighter.highlight_query(text, "")
        
        assert result == text  # Should return unchanged

    def test_short_matches_ignored(self):
        """Test that very short matches are ignored."""
        highlighter = TextHighlighter()
        text = "This is a test document."
        matches = ["a", "test"]  # "a" should be ignored, "is" is 2 chars so it's kept
        
        result = highlighter.highlight_matches(text, matches)
        
        assert "**test**" in result
        assert "**a**" not in result