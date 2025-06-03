"""Tests for the document processor functionality."""

import uuid
from datetime import datetime
from pathlib import Path

from oboyu.common.types import Chunk
from oboyu.indexer.core.document_processor import DocumentProcessor
from oboyu.indexer.core.document_processor import chunk_documents


class TestDocumentProcessor:
    """Test cases for the document processor."""

    def test_chunk_creation(self) -> None:
        """Test that chunks are created correctly."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20, document_prefix="検索文書: ")
        
        # Test data
        path = Path("/test/document.txt")
        content = "This is a test document. " * 10  # 260 characters
        title = "Test Document"
        language = "en"
        metadata = {"source": "test", "author": "tester"}
        
        # Process document
        chunks = processor.process_document(path, content, title, language, metadata)
        
        # Verify chunks
        assert len(chunks) == 3  # Should create 3 chunks with size 100 and overlap 20
        
        # Check first chunk
        assert chunks[0].path == path
        assert chunks[0].title == "Test Document - Part 1"
        assert len(chunks[0].content) <= 100  # Should not exceed chunk size
        assert chunks[0].chunk_index == 0
        assert chunks[0].language == "en"
        assert chunks[0].metadata == metadata
        assert chunks[0].prefix_content == f"検索文書: {chunks[0].content}"
        
        # Check overlapping content
        end_of_first = chunks[0].content[-20:]
        start_of_second = chunks[1].content[:20]
        assert end_of_first in chunks[1].content  # Verify overlap between chunks

    def test_chunking_with_japanese(self) -> None:
        """Test chunking with Japanese content."""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10, document_prefix="検索文書: ")
        
        # Japanese test data
        path = Path("/test/japanese.txt")
        content = "これは日本語のテストドキュメントです。 " * 5  # Longer than chunk size
        title = "日本語テスト"
        language = "ja"
        
        # Process document
        chunks = processor.process_document(path, content, title, language)
        
        # Verify chunks
        assert len(chunks) > 1  # Should create multiple chunks
        assert chunks[0].language == "ja"
        assert chunks[0].prefix_content.startswith("検索文書: ")
        
        # Check that each chunk has the correct prefix
        for chunk in chunks:
            assert chunk.prefix_content.startswith("検索文書: ")

    def test_chunk_boundary_handling(self) -> None:
        """Test that chunk boundaries are handled properly."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        # Text with clear sentence boundaries
        path = Path("/test/boundaries.txt")
        content = "First sentence. Second sentence. " * 10
        title = "Boundary Test"
        language = "en"
        
        chunks = processor.process_document(path, content, title, language)
        
        # Verify that chunks try to break at sentence boundaries
        for chunk in chunks:
            # Check if chunks end with a sentence
            if chunk.chunk_index < len(chunks) - 1:  # Not the last chunk
                assert chunk.content.rstrip().endswith(".") or chunk.content.rstrip().endswith("。")

    def test_chunk_documents_helper(self) -> None:
        """Test the chunk_documents helper function."""
        # Create test documents
        docs = [
            {
                "path": "/test/doc1.txt",
                "content": "Content of document 1. " * 5,
                "title": "Doc 1",
                "language": "en",
                "metadata": {"source": "test1"}
            },
            {
                "path": "/test/doc2.txt", 
                "content": "Content of document 2. " * 5,
                "title": "Doc 2",
                "language": "en",
                "metadata": {"source": "test2"}
            },
        ]
        
        # Process documents
        chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10, document_prefix="検索文書: ")
        
        # Verify results
        assert len(chunks) > 2  # Should have multiple chunks from the two documents
        
        # Check that chunks have correct metadata
        doc1_chunks = [c for c in chunks if str(c.path) == "/test/doc1.txt"]
        doc2_chunks = [c for c in chunks if str(c.path) == "/test/doc2.txt"]
        
        assert len(doc1_chunks) > 0
        assert len(doc2_chunks) > 0
        
        assert doc1_chunks[0].metadata == {"source": "test1"}
        assert doc2_chunks[0].metadata == {"source": "test2"}
        
        # Verify prefix is applied
        for chunk in chunks:
            assert chunk.prefix_content.startswith("検索文書: ")