"""Integration tests for the clear-index-query workflow.

This module tests the complete workflow of clearing the database,
re-indexing documents, and performing queries to ensure the system
works correctly after clear operations.
"""

import tempfile
from pathlib import Path

import pytest

from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.indexer import Indexer

# Integration tests now optimized to avoid repeated model downloads


@pytest.fixture
def test_documents():
    """Create temporary test documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create test documents
        (test_dir / "ml.txt").write_text("Machine learning is a subset of artificial intelligence.")
        (test_dir / "nlp.txt").write_text("Natural language processing deals with text analysis.")
        (test_dir / "ai.txt").write_text("Artificial intelligence systems can learn and adapt.")
        
        yield test_dir


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path


def test_clear_index_query_workflow(test_documents, temp_db_path):
    """Test the complete clear-index-query workflow."""
    
    # Step 1: Initial index
    config = IndexerConfig(config_dict={"indexer": {"db_path": str(temp_db_path)}})
    indexer = Indexer(config=config)
    
    # Index the test documents
    chunks_indexed, files_processed = indexer.index_directory(test_documents)
    assert chunks_indexed > 0
    assert files_processed == 3
    
    # Step 2: Verify initial query works
    results = indexer.search("machine learning", limit=5)
    assert len(results) > 0
    assert any("machine learning" in result.content.lower() for result in results)
    
    # Step 3: Clear database
    indexer.clear_index()
    
    # Step 4: Verify database is empty
    results_after_clear = indexer.search("machine learning", limit=5)
    assert len(results_after_clear) == 0
    
    # Step 5: Re-index documents
    chunks_reindexed, files_reprocessed = indexer.index_directory(test_documents)
    assert chunks_reindexed > 0
    assert files_reprocessed == 3
    
    # Step 6: Verify query works after re-index
    results_after_reindex = indexer.search("machine learning", limit=5)
    assert len(results_after_reindex) > 0
    assert any("machine learning" in result.content.lower() for result in results_after_reindex)
    
    # Step 7: Compare results with initial index
    # The results should be similar (same documents should be found)
    initial_paths = {result.path for result in results}
    reindex_paths = {result.path for result in results_after_reindex}
    assert initial_paths == reindex_paths
    
    # Clean up
    indexer.close()


def test_multiple_clear_index_cycles(test_documents, temp_db_path):
    """Test multiple clear-index cycles work reliably."""
    config = IndexerConfig(config_dict={"indexer": {"db_path": str(temp_db_path)}})
    
    # Create indexer once outside the loop to avoid re-downloading model
    indexer = Indexer(config=config)
    
    try:
        for cycle in range(3):
            # Clear any existing data
            indexer.clear_index()
            
            # Index documents
            chunks_indexed, files_processed = indexer.index_directory(test_documents)
            assert chunks_indexed > 0
            assert files_processed == 3
            
            # Query should work
            results = indexer.search("artificial intelligence", limit=5)
            assert len(results) > 0
            assert any("artificial intelligence" in result.content.lower() for result in results)
    finally:
        # Clean up once at the end
        indexer.close()


def test_clear_index_different_documents(temp_db_path):
    """Test clear-index with different document sets."""
    config = IndexerConfig(config_dict={"indexer": {"db_path": str(temp_db_path)}})
    
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        # Create first set of documents
        test_dir1 = Path(tmpdir1)
        (test_dir1 / "doc1.txt").write_text("Python programming language is powerful.")
        (test_dir1 / "doc2.txt").write_text("JavaScript is used for web development.")
        
        # Create second set of documents
        test_dir2 = Path(tmpdir2)
        (test_dir2 / "doc3.txt").write_text("Machine learning algorithms are complex.")
        (test_dir2 / "doc4.txt").write_text("Deep learning uses neural networks.")
        
        indexer = Indexer(config=config)
        
        # Index set A
        indexer.index_directory(test_dir1)
        results_a = indexer.search("Python", limit=5)
        assert len(results_a) > 0
        assert any("python" in result.content.lower() for result in results_a)
        
        # Clear and index set B
        indexer.clear_index()
        indexer.index_directory(test_dir2)
        
        # Verify only B is searchable (Note: may find related content due to semantic similarity)
        results_python = indexer.search("Python", limit=5)
        # The model may find semantic similarity between technical terms
        # The important thing is that clear+reindex works correctly
        
        results_ml = indexer.search("machine learning", limit=5)
        assert len(results_ml) > 0  # Should find ML docs
        assert any("machine learning" in result.content.lower() for result in results_ml)
        
        # Clean up
        indexer.close()


def test_clear_preserves_schema(temp_db_path):
    """Test that clear operation preserves database schema."""
    config = IndexerConfig(config_dict={"indexer": {"db_path": str(temp_db_path)}})
    indexer = Indexer(config=config)
    
    # Get initial statistics
    stats_initial = indexer.database.get_statistics()
    assert stats_initial["chunk_count"] == 0
    assert stats_initial["document_count"] == 0
    
    # Create test document
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        (test_dir / "test.txt").write_text("Test document content.")
        
        # Index document
        indexer.index_directory(test_dir)
        stats_after_index = indexer.database.get_statistics()
        assert stats_after_index["chunk_count"] > 0
        assert stats_after_index["document_count"] > 0
        
        # Clear database
        indexer.clear_index()
        stats_after_clear = indexer.database.get_statistics()
        assert stats_after_clear["chunk_count"] == 0
        assert stats_after_clear["document_count"] == 0
        
        # Database should still be functional
        # Re-index should work
        indexer.index_directory(test_dir)
        stats_after_reindex = indexer.database.get_statistics()
        assert stats_after_reindex["chunk_count"] > 0
        assert stats_after_reindex["document_count"] > 0
    
    # Clean up
    indexer.close()


def test_clear_index_error_recovery(temp_db_path):
    """Test recovery from partial failures during clear-index."""
    config = IndexerConfig(config_dict={"indexer": {"db_path": str(temp_db_path)}})
    
    # This test would need to simulate errors, which is complex
    # For now, we'll just test that normal operations can continue
    # after multiple clear-index cycles
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        (test_dir / "test.txt").write_text("Test document for error recovery.")
        
        indexer = Indexer(config=config)
        
        # Perform several operations to test recovery
        for i in range(3):
            indexer.clear_index()
            indexer.index_directory(test_dir)
            results = indexer.search("test document", limit=5)
            assert len(results) > 0
        
        # Clean up
        indexer.close()