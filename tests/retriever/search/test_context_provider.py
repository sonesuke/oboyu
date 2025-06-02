"""Tests for context provider functionality."""

import pytest

from oboyu.retriever.search.context_provider import ContextProvider


class TestContextProvider:
    """Test context provider functionality."""

    def test_get_surrounding_context(self):
        """Test getting context around a match position."""
        provider = ContextProvider()
        content = "This is a long document with machine learning content."
        match_position = 30  # Position of "machine"
        context_window = 10
        
        before, after = provider.get_surrounding_context(content, match_position, context_window)
        
        assert len(before) <= context_window
        assert len(after) <= context_window
        # Check actual position - let's find where "machine" actually is
        machine_pos = content.find("machine")
        before, after = provider.get_surrounding_context(content, machine_pos, context_window)
        assert "with " in before
        assert "machine" in after

    def test_get_context_around_range(self):
        """Test getting context around a text range."""
        provider = ContextProvider()
        content = "This is a document about machine learning algorithms."
        start_pos = 25  # Start of "machine"
        end_pos = 41   # End of "learning"
        context_window = 10
        
        before, after = provider.get_context_around_range(content, start_pos, end_pos, context_window)
        
        assert "about " in before
        assert " algorit" in after

    def test_expand_context_to_boundaries(self):
        """Test expanding context to natural boundaries."""
        provider = ContextProvider()
        content = "First sentence. Second sentence with machine learning. Third sentence."
        # Find actual position of "machine"
        machine_pos = content.find("machine")
        initial_window = 10
        
        start, end = provider.expand_context_to_boundaries(content, machine_pos, initial_window)
        
        # Should expand to include complete sentences
        expanded_text = content[start:end]
        # Check that it includes sentence boundaries
        assert "machine learning" in expanded_text
        assert expanded_text.endswith(".")

    def test_expand_context_with_japanese_boundaries(self):
        """Test expanding context with Japanese boundaries."""
        provider = ContextProvider()
        content = "最初の文章。機械学習について。三番目の文章。"
        center_pos = 8  # Around "機械学習"
        initial_window = 5
        boundary_chars = "。！？"
        
        start, end = provider.expand_context_to_boundaries(
            content, center_pos, initial_window, boundary_chars
        )
        
        expanded_text = content[start:end]
        assert "機械学習について。" in expanded_text

    def test_get_optimal_context_window(self):
        """Test calculating optimal context window size."""
        provider = ContextProvider()
        
        # Test with short content
        window = provider.get_optimal_context_window(100, 50, 1)
        assert window == 25  # Half the target length for single match
        
        # Test with long content and multiple matches
        window = provider.get_optimal_context_window(1000, 200, 3)
        assert window == 33  # 200 / (2 * 3) = 33.33, rounded down
        
        # Test minimum window enforcement
        window = provider.get_optimal_context_window(1000, 30, 5)
        assert window == 20  # Should enforce minimum of 20

    def test_context_at_boundaries(self):
        """Test context extraction at document boundaries."""
        provider = ContextProvider()
        content = "Short document."
        
        # Test at beginning
        before, after = provider.get_surrounding_context(content, 0, 10)
        assert before == ""
        assert len(after) <= 10
        
        # Test at end
        before, after = provider.get_surrounding_context(content, len(content), 10)
        assert after == ""
        assert len(before) <= 10

    def test_empty_content(self):
        """Test behavior with empty content."""
        provider = ContextProvider()
        
        before, after = provider.get_surrounding_context("", 0, 10)
        assert before == ""
        assert after == ""
        
        start, end = provider.expand_context_to_boundaries("", 0, 10)
        assert start == 0
        assert end == 0

    def test_invalid_range(self):
        """Test behavior with invalid range."""
        provider = ContextProvider()
        content = "Test document."
        
        # Test with start >= end
        before, after = provider.get_context_around_range(content, 10, 5, 10)
        assert before == ""
        assert after == ""

    def test_large_context_window(self):
        """Test with context window larger than content."""
        provider = ContextProvider()
        content = "Short."
        
        before, after = provider.get_surrounding_context(content, 3, 100)
        assert before == "Sho"
        assert after == "rt."