"""Tests for IndexingPipeline."""

import pytest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

from oboyu.crawler.crawler import CrawlerResult
from oboyu.indexer.orchestrators.indexing_pipeline import IndexingPipeline
from oboyu.indexer.orchestrators.service_registry import ServiceRegistry


@pytest.fixture
def mock_services() -> ServiceRegistry:
    """Create mock service registry."""
    services = MagicMock(spec=ServiceRegistry)
    
    # Mock individual services
    services.get_document_processor.return_value = MagicMock()
    services.get_database_service.return_value = MagicMock()
    services.get_embedding_service.return_value = MagicMock()
    services.get_bm25_indexer.return_value = MagicMock()
    services.get_change_detector.return_value = MagicMock()
    
    return services


@pytest.fixture
def indexing_pipeline(mock_services: ServiceRegistry) -> IndexingPipeline:
    """Create indexing pipeline for testing."""
    return IndexingPipeline(mock_services)


@pytest.fixture
def sample_crawler_results() -> List[CrawlerResult]:
    """Create sample crawler results."""
    # Create temporary test files
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    test_file1 = temp_dir / "test1.txt"
    test_file2 = temp_dir / "test2.txt"
    
    test_file1.write_text("Test content 1")
    test_file2.write_text("Test content 2")
    
    return [
        CrawlerResult(
            path=test_file1,
            title="Test Title 1",
            content="Test content 1",
            language="en",
            metadata={},
        ),
        CrawlerResult(
            path=test_file2,
            title="Test Title 2",
            content="Test content 2",
            language="en",
            metadata={},
        ),
    ]


def test_index_documents_empty_list(indexing_pipeline: IndexingPipeline) -> None:
    """Test indexing with empty crawler results."""
    result = indexing_pipeline.index_documents([])
    
    assert result == {"indexed_chunks": 0, "total_documents": 0}


def test_index_documents_success(
    indexing_pipeline: IndexingPipeline,
    sample_crawler_results: List[CrawlerResult],
) -> None:
    """Test successful document indexing."""
    # Mock document processor to return chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.id = "chunk1"
    mock_chunk2 = MagicMock()
    mock_chunk2.id = "chunk2"
    
    indexing_pipeline.document_processor.process_document.side_effect = [
        [mock_chunk1],
        [mock_chunk2],
    ]
    
    # Mock embedding preparation and generation
    indexing_pipeline.document_processor.prepare_for_embedding.return_value = ["text1", "text2"]
    indexing_pipeline.embedding_service.generate_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
    
    result = indexing_pipeline.index_documents(sample_crawler_results)
    
    assert result["indexed_chunks"] == 2
    assert result["total_documents"] == 2
    assert "error" not in result
    
    # Verify service calls
    assert indexing_pipeline.document_processor.process_document.call_count == 2
    indexing_pipeline.database_service.store_chunks.assert_called_once()
    indexing_pipeline.embedding_service.generate_embeddings.assert_called_once()
    indexing_pipeline.database_service.store_embeddings.assert_called_once()
    indexing_pipeline.bm25_indexer.index_chunks.assert_called_once()


def test_index_documents_no_chunks(
    indexing_pipeline: IndexingPipeline,
    sample_crawler_results: List[CrawlerResult],
) -> None:
    """Test indexing when no chunks are produced."""
    # Mock document processor to return empty chunks
    indexing_pipeline.document_processor.process_document.return_value = []
    
    result = indexing_pipeline.index_documents(sample_crawler_results)
    
    assert result["indexed_chunks"] == 0
    assert result["total_documents"] == 2
    
    # Verify minimal service calls
    assert indexing_pipeline.document_processor.process_document.call_count == 2
    indexing_pipeline.database_service.store_chunks.assert_not_called()


def test_index_documents_with_progress_callback(
    indexing_pipeline: IndexingPipeline,
    sample_crawler_results: List[CrawlerResult],
) -> None:
    """Test indexing with progress callback."""
    mock_chunk = MagicMock()
    mock_chunk.id = "chunk1"
    
    indexing_pipeline.document_processor.process_document.return_value = [mock_chunk]
    indexing_pipeline.document_processor.prepare_for_embedding.return_value = ["text1"]
    indexing_pipeline.embedding_service.generate_embeddings.return_value = [[0.1, 0.2]]
    
    progress_callback = MagicMock()
    result = indexing_pipeline.index_documents(sample_crawler_results, progress_callback)
    
    assert result["indexed_chunks"] == 2  # One chunk per document
    assert result["total_documents"] == 2
    
    # Verify progress callback was called
    assert progress_callback.call_count > 0


def test_index_documents_error_handling(
    indexing_pipeline: IndexingPipeline,
    sample_crawler_results: List[CrawlerResult],
) -> None:
    """Test error handling during indexing."""
    # Mock document processor to raise an exception
    indexing_pipeline.document_processor.process_document.side_effect = Exception("Processing error")
    
    result = indexing_pipeline.index_documents(sample_crawler_results)
    
    assert result["indexed_chunks"] == 0
    assert result["total_documents"] == 2
    assert "error" in result
    assert "Processing error" in result["error"]


def test_rebuild_indexes(indexing_pipeline: IndexingPipeline) -> None:
    """Test index rebuilding."""
    # Mock BM25 indexer with clear method
    indexing_pipeline.bm25_indexer.clear = MagicMock()
    
    indexing_pipeline.rebuild_indexes()
    
    # Verify services were called
    indexing_pipeline.database_service.clear_database.assert_called_once()
    indexing_pipeline.bm25_indexer.clear.assert_called_once()


def test_rebuild_indexes_no_clear_method(indexing_pipeline: IndexingPipeline) -> None:
    """Test index rebuilding when BM25 indexer has no clear method."""
    # Remove clear method from BM25 indexer
    del indexing_pipeline.bm25_indexer.clear
    
    # Should not raise an exception
    indexing_pipeline.rebuild_indexes()
    
    # Verify database was still cleared
    indexing_pipeline.database_service.clear_database.assert_called_once()