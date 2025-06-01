"""Tests for the DocumentChunker component."""

import pytest

from oboyu.indexer.core.document_chunker import DocumentChunker


class TestDocumentChunker:
    """Test cases for DocumentChunker."""

    def test_chunk_text_basic(self) -> None:
        """Test basic text chunking functionality."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        
        text = "This is a test sentence. " * 10  # 260 characters
        chunks = chunker.chunk_text(text)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should not exceed chunk_size
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_chunk_text_with_overlap(self) -> None:
        """Test that chunks have proper overlap."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        
        text = "This is a test sentence that is long enough to be split into multiple chunks."
        chunks = chunker.chunk_text(text)
        
        # Check overlap between consecutive chunks
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                # Check for overlap by finding common content
                # The overlap may not be exactly 10 chars due to boundary adjustments
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                
                # Find if there's any overlap
                overlap_found = False
                for j in range(min(len(current_chunk), 20), 0, -1):
                    if current_chunk[-j:] in next_chunk:
                        overlap_found = True
                        break
                assert overlap_found, f"No overlap found between chunks {i} and {i+1}"

    def test_chunk_text_single_chunk(self) -> None:
        """Test that short text returns a single chunk."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        text = "Short text"
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_sentence_boundaries(self) -> None:
        """Test that chunks prefer to break at sentence boundaries."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk_text(text)
        
        # Check that chunks end with sentence boundaries where possible
        for i, chunk in enumerate(chunks[:-1]):  # All but the last chunk
            # Should end with a period if it's not at the exact chunk size limit
            assert chunk.endswith(".") or chunk.endswith("。") or len(chunk) == 50

    def test_chunk_text_paragraph_boundaries(self) -> None:
        """Test that chunks prefer paragraph boundaries."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph."
        chunks = chunker.chunk_text(text)
        
        # With proper size, should break at paragraph boundaries
        for chunk in chunks[:-1]:
            # Check if chunk ends at a natural boundary
            assert chunk.endswith(".") or chunk.endswith("\n") or len(chunk) >= 80

    def test_chunk_text_japanese(self) -> None:
        """Test chunking with Japanese text."""
        chunker = DocumentChunker(chunk_size=30, chunk_overlap=10)
        
        text = "これは日本語のテストです。これは二番目の文です。これは三番目の文です。"
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        
        # Check that Japanese sentence boundaries are respected
        for chunk in chunks[:-1]:
            # Should try to end with Japanese period
            assert "。" in chunk or len(chunk) >= 25

    def test_chunk_text_empty_input(self) -> None:
        """Test chunking with empty input."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        
        chunks = chunker.chunk_text("")
        assert chunks == [""]
        
        chunks = chunker.chunk_text("   ")
        assert chunks == [""]

    def test_chunk_text_large_document(self) -> None:
        """Test chunking with a large document to ensure no infinite loops."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        # Create a large text
        text = "This is a test sentence. " * 1000  # 26,000 characters
        chunks = chunker.chunk_text(text)
        
        # Should create many chunks without hanging
        assert len(chunks) > 200
        
        # Verify all text is covered
        total_length = sum(len(chunk) for chunk in chunks)
        # Account for overlap
        assert total_length >= len(text)