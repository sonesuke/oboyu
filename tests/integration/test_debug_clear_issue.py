"""Debug test to isolate the clear-index-query issue."""

import tempfile
from pathlib import Path

import pytest

from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.indexer import Indexer


def test_debug_clear_issue():
    """Debug the clear-index-query issue step by step."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Create test documents with very different content
        test_dir1 = Path(tmpdir) / "docs1"
        test_dir1.mkdir()
        (test_dir1 / "python.txt").write_text("Python programming language is powerful and versatile for coding.")
        
        test_dir2 = Path(tmpdir) / "docs2"  
        test_dir2.mkdir()
        (test_dir2 / "ml.txt").write_text("Machine learning algorithms use complex mathematical models and statistics.")
        
        # Test configuration
        config = IndexerConfig(config_dict={"indexer": {"db_path": str(db_path)}})
        indexer = Indexer(config=config)
        
        # Phase 1: Index Python docs
        print("\n=== Phase 1: Index Python docs ===")
        indexer.index_directory(test_dir1)
        
        results = indexer.search("Python programming", limit=5)
        print(f"Search for 'Python programming': {len(results)} results")
        for result in results:
            print(f"  - {result.title}: {result.content[:50]}... (score: {result.score:.3f})")
        
        assert len(results) > 0, "Should find Python docs"
        assert any("python" in result.content.lower() for result in results), "Should contain Python content"
        
        # Phase 2: Clear database
        print("\n=== Phase 2: Clear database ===")
        indexer.clear_index()
        
        results_after_clear = indexer.search("Python programming", limit=5)
        print(f"Search after clear: {len(results_after_clear)} results")
        assert len(results_after_clear) == 0, "Should have no results after clear"
        
        # Phase 3: Index ML docs (completely different content)
        print("\n=== Phase 3: Index ML docs ===")
        indexer.index_directory(test_dir2)
        
        results_ml = indexer.search("machine learning", limit=5)
        print(f"Search for 'machine learning': {len(results_ml)} results")
        for result in results_ml:
            print(f"  - {result.title}: {result.content[:50]}... (score: {result.score:.3f})")
        
        assert len(results_ml) > 0, "Should find ML docs"
        assert any("machine learning" in result.content.lower() for result in results_ml), "Should contain ML content"
        
        # Phase 4: Search for Python in ML docs (should be 0 results or very low scores)
        print("\n=== Phase 4: Search for Python in ML docs (should be 0 or very low score results) ===")
        results_python_in_ml = indexer.search("Python programming", limit=5)
        print(f"Search for 'Python programming' in ML docs: {len(results_python_in_ml)} results")
        for result in results_python_in_ml:
            print(f"  - {result.title}: {result.content[:50]}... (score: {result.score:.3f})")
        
        # This is the critical test - searching for "Python programming" in docs about ML
        # should either return no results or results with very low similarity scores
        if len(results_python_in_ml) > 0:
            # If we get results, they should have very low scores (< 0.5 for cosine similarity)
            max_score = max(result.score for result in results_python_in_ml)
            print(f"Maximum score for 'Python programming' in ML docs: {max_score:.3f}")
            
            # This might be the bug - if we get high similarity scores for unrelated content,
            # it suggests the vector search is not working correctly after clear+reindex
            assert max_score < 0.5, f"Scores should be low for unrelated content, got {max_score:.3f}"
        
        # Check database statistics
        stats = indexer.database.get_statistics()
        print(f"\nDatabase stats: {stats['chunk_count']} chunks, {stats['document_count']} documents")
        
        indexer.close()


def test_simple_clear_workflow():
    """Test the simple clear workflow to ensure basic functionality."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        test_dir = Path(tmpdir) / "docs"
        test_dir.mkdir()
        
        # Create a simple test document
        (test_dir / "test.txt").write_text("This is a simple test document about testing.")
        
        config = IndexerConfig(config_dict={"indexer": {"db_path": str(db_path)}})
        indexer = Indexer(config=config)
        
        # Index
        indexer.index_directory(test_dir)
        results_before = indexer.search("test document", limit=5)
        assert len(results_before) > 0
        
        # Clear
        indexer.clear_index()
        results_after_clear = indexer.search("test document", limit=5)
        assert len(results_after_clear) == 0
        
        # Re-index
        indexer.index_directory(test_dir)
        results_after_reindex = indexer.search("test document", limit=5)
        assert len(results_after_reindex) > 0
        
        indexer.close()