"""Tests for the main indexer functionality."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oboyu.crawler.crawler import CrawlerResult
from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.indexer import Indexer, SearchResult
from oboyu.indexer.processor import Chunk


class TestIndexer:
    """Test cases for the main Indexer class."""

    def test_indexer_initialization(self) -> None:
        """Test indexer initialization with default configuration."""
        # Use mocks for all components to avoid actual initialization
        with patch("oboyu.indexer.indexer.DocumentProcessor") as mock_processor, \
             patch("oboyu.indexer.indexer.EmbeddingGenerator") as mock_generator, \
             patch("oboyu.indexer.indexer.Database") as mock_db:
            
            # Initialize the indexer
            indexer = Indexer(config=IndexerConfig(config_dict={"indexer": {"db_path": "test.db"}}))
            
            # Verify components were initialized
            assert indexer.config is not None
            assert mock_processor.called
            assert mock_generator.called
            assert mock_db.called
            
            # Verify database was set up
            assert mock_db.return_value.setup.called

    def test_index_documents(self) -> None:
        """Test indexing documents from crawler results."""
        # Create mock components
        mock_processor = MagicMock()
        mock_generator = MagicMock()
        mock_db = MagicMock()
        
        # Set up mock return values
        mock_processor.process_document.return_value = [
            Chunk(
                id="test-chunk-1",
                path=Path("/test/doc1.txt"),
                title="Test Document 1",
                content="This is test document one.",
                chunk_index=0,
                language="en",
                created_at=datetime.now(),
                modified_at=datetime.now(),
                metadata={"source": "test"},
                prefix_content="検索文書: This is test document one.",
            )
        ]
        
        mock_generator.generate_embeddings.return_value = [
            ("test-embedding-1", "test-chunk-1", np.array([0.1] * 256), datetime.now())
        ]
        
        # Initialize indexer with mocks
        indexer = Indexer(
            config=IndexerConfig(config_dict={"indexer": {"db_path": "test.db"}}),
            processor=mock_processor,
            embedding_generator=mock_generator,
            database=mock_db
        )
        
        # Create test crawler results
        crawler_results = [
            CrawlerResult(
                path=Path("/test/doc1.txt"),
                title="Test Document 1",
                content="This is test document one.",
                language="en",
                metadata={"source": "test"},
            )
        ]
        
        # Index the documents
        chunk_count = indexer.index_documents(crawler_results)
        
        # Verify results
        assert chunk_count == 1
        mock_processor.process_document.assert_called_once()
        mock_generator.generate_embeddings.assert_called_once()
        mock_db.store_chunks.assert_called_once()
        mock_db.store_embeddings.assert_called_once()

    def test_search(self) -> None:
        """Test searching for documents."""
        # Create mock components
        mock_processor = MagicMock()
        mock_generator = MagicMock()
        mock_db = MagicMock()
        
        # Set up mock return values
        mock_generator.generate_query_embedding.return_value = np.array([0.1] * 256)
        mock_db.search.return_value = [
            {
                "chunk_id": "test-chunk-1",
                "path": "/test/doc1.txt",
                "title": "Test Document 1",
                "content": "This is test document one.",
                "chunk_index": 0,
                "language": "en",
                "metadata": {"source": "test"},
                "score": 0.1,
            }
        ]
        
        # Initialize indexer with mocks
        indexer = Indexer(
            config=IndexerConfig(config_dict={"indexer": {"db_path": "test.db"}}),
            processor=mock_processor,
            embedding_generator=mock_generator,
            database=mock_db
        )
        
        # Search for documents
        results = indexer.search("test query", limit=5)
        
        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk_id == "test-chunk-1"
        assert results[0].title == "Test Document 1"
        assert results[0].content == "This is test document one."
        assert results[0].score == 0.1
        
        # Verify method calls
        mock_generator.generate_query_embedding.assert_called_once_with("test query")
        mock_db.search.assert_called_once_with(mock_generator.generate_query_embedding.return_value, 5, None)

    def test_delete_document(self) -> None:
        """Test deleting a document from the index."""
        # Create mock components
        mock_processor = MagicMock()
        mock_generator = MagicMock()
        mock_db = MagicMock()
        
        # Set up mock return values
        mock_db.delete_chunks_by_path.return_value = 2  # 2 chunks deleted
        
        # Initialize indexer with mocks
        indexer = Indexer(
            config=IndexerConfig(config_dict={"indexer": {"db_path": "test.db"}}),
            processor=mock_processor,
            embedding_generator=mock_generator,
            database=mock_db
        )
        
        # Add to processed files set (for testing removal)
        test_path = Path("/test/doc1.txt")
        indexer._processed_files.add(test_path)
        
        # Delete the document
        deleted_count = indexer.delete_document(test_path)
        
        # Verify results
        assert deleted_count == 2
        mock_db.delete_chunks_by_path.assert_called_once_with(test_path)
        assert test_path not in indexer._processed_files
        
    def test_clear_index(self) -> None:
        """Test clearing the entire index."""
        # Create mock components
        mock_processor = MagicMock()
        mock_generator = MagicMock()
        mock_db = MagicMock()
        
        # Initialize indexer with mocks, providing a test db_path
        indexer = Indexer(
            config=IndexerConfig(config_dict={"indexer": {"db_path": "test.db"}}),
            processor=mock_processor,
            embedding_generator=mock_generator,
            database=mock_db
        )
        
        # Add to processed files set (for testing that it gets cleared)
        indexer._processed_files.add(Path("/test/doc1.txt"))
        indexer._processed_files.add(Path("/test/doc2.txt"))
        
        # Clear the index
        indexer.clear_index()
        
        # Verify results
        mock_db.clear.assert_called_once()
        assert len(indexer._processed_files) == 0  # Processed files should be cleared

