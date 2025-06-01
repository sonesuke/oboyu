"""Tests for the EmbeddingPrefixHandler component."""

import pytest

from oboyu.indexer.core.embedding_prefix_handler import EmbeddingPrefixHandler


class TestEmbeddingPrefixHandler:
    """Test cases for EmbeddingPrefixHandler."""

    def test_add_document_prefix_default(self) -> None:
        """Test default prefix handling for Ruri model."""
        handler = EmbeddingPrefixHandler()
        
        text = "This is a test document."
        result = handler.add_document_prefix(text, "ruri")
        
        assert result == "検索文書: This is a test document."

    def test_add_document_prefix_custom(self) -> None:
        """Test custom prefix handling."""
        custom_prefix = "Document: "
        handler = EmbeddingPrefixHandler(document_prefix=custom_prefix)
        
        text = "This is a test document."
        result = handler.add_document_prefix(text, "ruri")
        
        assert result == "Document: This is a test document."

    def test_add_document_prefix_ruri_v3(self) -> None:
        """Test prefix handling for Ruri v3 model."""
        handler = EmbeddingPrefixHandler()
        
        text = "Test content"
        result = handler.add_document_prefix(text, "ruri-v3")
        
        assert result == "検索文書: Test content"

    def test_add_document_prefix_other_models(self) -> None:
        """Test that other models return text without prefix."""
        handler = EmbeddingPrefixHandler()
        
        text = "Test content"
        
        # Test various non-Ruri models
        for model_type in ["bert", "openai", "sentence-transformers", "unknown"]:
            result = handler.add_document_prefix(text, model_type)
            assert result == text

    def test_add_document_prefix_empty_text(self) -> None:
        """Test handling of empty text."""
        handler = EmbeddingPrefixHandler()
        
        result = handler.add_document_prefix("", "ruri")
        assert result == "検索文書: "
        
        result = handler.add_document_prefix("", "other")
        assert result == ""

    def test_add_document_prefix_with_spaces(self) -> None:
        """Test prefix handling with text that has leading/trailing spaces."""
        handler = EmbeddingPrefixHandler()
        
        text = "  Text with spaces  "
        result = handler.add_document_prefix(text, "ruri")
        
        assert result == "検索文書:   Text with spaces  "

    def test_add_document_prefix_multiline(self) -> None:
        """Test prefix handling with multiline text."""
        handler = EmbeddingPrefixHandler()
        
        text = "First line\nSecond line\nThird line"
        result = handler.add_document_prefix(text, "ruri")
        
        assert result.startswith("検索文書: ")
        assert "First line\nSecond line\nThird line" in result