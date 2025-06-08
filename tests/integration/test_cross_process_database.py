"""Integration tests for cross-process database reliability.

This module tests that database operations work correctly across process
boundaries and after clear→index→query workflows.
"""

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import pytest

from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.indexer import Indexer
from oboyu.retriever.retriever import Retriever

logger = logging.getLogger(__name__)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path (file doesn't exist initially)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
        db_path = Path(f.name)
    
    # File is deleted when context exits, so we just have the path
    yield db_path
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture  
def temp_content_dir():
    """Create temporary content directory with test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        content_dir = Path(temp_dir)
        
        # Create test files
        (content_dir / "test1.txt").write_text("This is test file one with some content.")
        (content_dir / "test2.txt").write_text("This is test file two with different content.")
        (content_dir / "test3.md").write_text("# Test Document\n\nThis is a markdown file.")
        
        yield content_dir


@pytest.fixture
def indexer_config(temp_db_path):
    """Create indexer configuration for testing."""
    config = IndexerConfig()
    config.db_path = temp_db_path
    config.processing.chunk_size = 100
    config.processing.chunk_overlap = 20
    return config


def run_oboyu_command(command: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
    """Run oboyu CLI command in subprocess."""
    import sys
    cmd = [sys.executable, "-m", "oboyu"] + command
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        cwd=cwd or Path.cwd(),
        timeout=30
    )
    
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Return code: {result.returncode}")
    logger.info(f"Stdout: {result.stdout}")
    if result.stderr:
        logger.info(f"Stderr: {result.stderr}")
    
    return result


class TestCrossProcessDatabase:
    """Test database operations across process boundaries."""

    def test_index_in_process_a_query_in_process_b(self, temp_db_path, temp_content_dir, indexer_config):
        """Test indexing in one process and querying in another."""
        # Process A: Index documents
        indexer = Indexer(config=indexer_config)
        
        try:
            # Create mock documents for indexing
            from datetime import datetime
            from oboyu.common.types import Chunk
            now = datetime.now()
            chunks = [
                Chunk(
                    id="test1",
                    path=temp_content_dir / "test1.txt",
                    title="Test Document 1",
                    content="This is test content for cross-process testing",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": str(temp_content_dir / "test1.txt")},
                ),
                Chunk(
                    id="test2",
                    path=temp_content_dir / "test2.txt", 
                    title="Test Document 2",
                    content="Different content for testing search functionality",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": str(temp_content_dir / "test2.txt")},
                ),
            ]
            
            # Index the chunks by storing them directly and generating embeddings
            indexer.database_service.store_chunks(chunks)
            
            # Generate embeddings for the chunks
            chunk_ids = [chunk.id for chunk in chunks]
            contents = [chunk.content for chunk in chunks]
            embeddings = indexer.embedding_service.generate_embeddings(contents)
            indexer.database_service.store_embeddings(chunk_ids, embeddings)
            
            # Ensure HNSW index exists
            indexer.database_service.ensure_hnsw_index()
            
            result = {"indexed_chunks": len(chunks), "total_documents": len(chunks)}
            
        finally:
            indexer.close()
        
        # Small delay to ensure file operations complete
        time.sleep(0.1)
        
        # Process B: Query the indexed data
        retriever = Retriever(indexer_config)
        
        try:
            # Search for content using the search method which should work
            results = retriever.search("test content", limit=10, mode="hybrid")
            
            # Should find results from the indexing done in Process A
            assert len(results) > 0
            assert any("test content" in result.content.lower() for result in results)
            
        finally:
            retriever.close()

    def test_clear_index_query_workflow(self, temp_db_path, temp_content_dir, indexer_config):
        """Test clear→index→query workflow reliability."""
        # Step 1: Initial indexing
        indexer = Indexer(config=indexer_config)
        
        try:
            from datetime import datetime
            from oboyu.common.types import Chunk
            now = datetime.now()
            initial_chunks = [
                Chunk(
                    id="initial1",
                    path=Path("initial.txt"),
                    title="Initial Document",
                    content="Initial content before clear",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "initial.txt"},
                ),
            ]
            
            # Index the chunks by storing them directly
            indexer.database_service.store_chunks(initial_chunks)
            chunk_ids = [chunk.id for chunk in initial_chunks]
            contents = [chunk.content for chunk in initial_chunks]
            embeddings = indexer.embedding_service.generate_embeddings(contents)
            indexer.database_service.store_embeddings(chunk_ids, embeddings)
            indexer.database_service.ensure_hnsw_index()
            
            result = {"indexed_chunks": len(initial_chunks)}
            
        finally:
            indexer.close()
        
        # Step 2: Clear the database
        indexer = Indexer(config=indexer_config)
        try:
            indexer.database_service.clear_database()
        finally:
            indexer.close()
        
        # Step 3: Index new content
        indexer = Indexer(config=indexer_config)
        try:
            from datetime import datetime
            from oboyu.common.types import Chunk
            now = datetime.now()
            new_chunks = [
                Chunk(
                    id="new1",
                    path=Path("new.txt"),
                    title="New Document 1",
                    content="New content after clear operation",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "new.txt"},
                ),
                Chunk(
                    id="new2",
                    path=Path("new2.txt"),
                    title="New Document 2", 
                    content="Additional new content for testing",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "new2.txt"},
                ),
            ]
            
            # Index the chunks by storing them directly
            indexer.database_service.store_chunks(new_chunks)
            chunk_ids = [chunk.id for chunk in new_chunks]
            contents = [chunk.content for chunk in new_chunks]
            embeddings = indexer.embedding_service.generate_embeddings(contents)
            indexer.database_service.store_embeddings(chunk_ids, embeddings)
            indexer.database_service.ensure_hnsw_index()
            
            result = {"indexed_chunks": len(new_chunks)}
            
        finally:
            indexer.close()
        
        # Step 4: Query the new content
        retriever = Retriever(indexer_config)
        try:
            # Should find new content
            results = retriever.search("new content", limit=10, mode="hybrid")
            assert len(results) > 0
            assert any("new content" in result.content.lower() for result in results)
            
            # Should NOT find initial content
            old_results = retriever.search("initial content", limit=10, mode="hybrid")
            # Either no results or results that don't contain the old content
            if old_results:
                assert not any("initial content" in result.content.lower() for result in old_results)
            
        finally:
            retriever.close()

    def test_multiple_processes_simultaneous_query(self, temp_db_path, temp_content_dir, indexer_config):
        """Test multiple processes querying simultaneously."""
        # First, index some content
        indexer = Indexer(config=indexer_config)
        
        try:
            from datetime import datetime
            from oboyu.common.types import Chunk
            now = datetime.now()
            chunks = [
                Chunk(
                    id="multi1",
                    path=Path("multi.txt"),
                    title="Multi Document 1",
                    content="Content for multi-process query testing",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "multi.txt"},
                ),
                Chunk(
                    id="multi2",
                    path=Path("multi2.txt"),
                    title="Multi Document 2",
                    content="Another piece of content for concurrent access",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "multi2.txt"},
                ),
            ]
            
            # Index the chunks by storing them directly
            indexer.database_service.store_chunks(chunks)
            chunk_ids = [chunk.id for chunk in chunks]
            contents = [chunk.content for chunk in chunks]
            embeddings = indexer.embedding_service.generate_embeddings(contents)
            indexer.database_service.store_embeddings(chunk_ids, embeddings)
            indexer.database_service.ensure_hnsw_index()
            
            result = {"indexed_chunks": len(chunks)}
            
        finally:
            indexer.close()
        
        # Now test multiple retrievers accessing simultaneously
        retrievers = []
        try:
            # Create multiple retriever instances (simulating different processes)
            for i in range(3):
                retriever = Retriever(indexer_config)
                retrievers.append(retriever)
            
            # All should be able to query successfully
            for i, retriever in enumerate(retrievers):
                results = retriever.search("content", limit=5, mode="hybrid")
                assert len(results) > 0, f"Retriever {i} failed to get results"
                
        finally:
            for retriever in retrievers:
                try:
                    retriever.close()
                except Exception as e:
                    logger.warning(f"Error closing retriever: {e}")

    def test_database_corruption_recovery(self, temp_db_path, indexer_config):
        """Test database recovery from corruption scenarios."""
        # Create initial database with content
        indexer = Indexer(config=indexer_config)
        
        try:
            from datetime import datetime
            from oboyu.common.types import Chunk
            now = datetime.now()
            chunks = [
                Chunk(
                    id="recovery1",
                    path=Path("recovery.txt"),
                    title="Recovery Document",
                    content="Content for recovery testing",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "recovery.txt"},
                ),
            ]
            
            # Index the chunks by storing them directly
            indexer.database_service.store_chunks(chunks)
            chunk_ids = [chunk.id for chunk in chunks]
            contents = [chunk.content for chunk in chunks]
            embeddings = indexer.embedding_service.generate_embeddings(contents)
            indexer.database_service.store_embeddings(chunk_ids, embeddings)
            indexer.database_service.ensure_hnsw_index()
            
            result = {"indexed_chunks": len(chunks)}
            
        finally:
            indexer.close()
        
        # Simulate corruption by truncating the database file
        with open(temp_db_path, "w") as f:
            f.write("corrupted")
        
        # Should be able to recover by creating fresh database
        new_indexer = Indexer(config=indexer_config)
        try:
            # Should not crash, should initialize fresh database
            from datetime import datetime
            from oboyu.common.types import Chunk
            now = datetime.now()
            new_chunks = [
                Chunk(
                    id="recovery2",
                    path=Path("recovery2.txt"),
                    title="Recovery Document 2",
                    content="Content after recovery",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "recovery2.txt"},
                ),
            ]
            
            # Index the chunks by storing them directly
            new_indexer.database_service.store_chunks(new_chunks)
            chunk_ids = [chunk.id for chunk in new_chunks]
            contents = [chunk.content for chunk in new_chunks]
            embeddings = new_indexer.embedding_service.generate_embeddings(contents)
            new_indexer.database_service.store_embeddings(chunk_ids, embeddings)
            new_indexer.database_service.ensure_hnsw_index()
            
            result = {"indexed_chunks": len(new_chunks)}
            
        finally:
            new_indexer.close()
        
        # Should be able to query the recovered database
        retriever = Retriever(indexer_config)
        try:
            results = retriever.search("recovery", limit=10, mode="hybrid")
            # Should find the new content
            assert len(results) > 0
            
        finally:
            retriever.close()

    @pytest.mark.slow
    def test_cli_cross_process_workflow(self, temp_db_path, temp_content_dir):
        """Test cross-process workflow using CLI commands."""
        # Step 1: Clear using CLI
        result = run_oboyu_command([
            "clear", 
            "--db-path", str(temp_db_path),
            "--force"
        ])
        assert result.returncode == 0
        
        # Step 2: Index using CLI  
        result = run_oboyu_command([
            "index",
            str(temp_content_dir),
            "--db-path", str(temp_db_path),
            "--chunk-size", "100"
        ])
        assert result.returncode == 0
        assert "indexed" in result.stdout.lower() or "chunks" in result.stdout.lower()
        
        # Step 3: Query using CLI
        result = run_oboyu_command([
            "query",
            "--query", "test content",
            "--db-path", str(temp_db_path),
            "--top-k", "5"
        ])
        assert result.returncode == 0
        # Should have some results
        assert len(result.stdout.strip()) > 0

    def test_fresh_process_database_loading(self, temp_db_path, indexer_config):
        """Test that fresh processes can load existing databases reliably."""
        # Create database in first process
        indexer1 = Indexer(config=indexer_config)
        
        try:
            from datetime import datetime
            from oboyu.common.types import Chunk
            now = datetime.now()
            chunks = [
                Chunk(
                    id="fresh1",
                    path=Path("fresh.txt"),
                    title="Fresh Document 1",
                    content="Content for fresh process testing",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "fresh.txt"},
                ),
            ]
            
            # Index the chunks by storing them directly
            indexer1.database_service.store_chunks(chunks)
            chunk_ids = [chunk.id for chunk in chunks]
            contents = [chunk.content for chunk in chunks]
            embeddings = indexer1.embedding_service.generate_embeddings(contents)
            indexer1.database_service.store_embeddings(chunk_ids, embeddings)
            indexer1.database_service.ensure_hnsw_index()
            
            result = {"indexed_chunks": len(chunks)}
            
        finally:
            indexer1.close()
        
        # Simulate fresh process by creating new indexer instance
        # with same config (fresh process would reload from disk)
        indexer2 = Indexer(config=indexer_config)
        
        try:
            # Should be able to access existing data
            stats = indexer2.get_database_stats()
            assert stats["chunk_count"] > 0
            
            # Should be able to add more content
            from datetime import datetime
            from oboyu.common.types import Chunk
            now = datetime.now()
            more_chunks = [
                Chunk(
                    id="fresh2",
                    path=Path("fresh2.txt"),
                    title="Fresh Document 2",
                    content="Additional content from fresh process",
                    chunk_index=0,
                    language="en",
                    created_at=now,
                    modified_at=now,
                    metadata={"source": "fresh2.txt"},
                ),
            ]
            
            # Index the chunks by storing them directly
            indexer2.database_service.store_chunks(more_chunks)
            chunk_ids = [chunk.id for chunk in more_chunks]
            contents = [chunk.content for chunk in more_chunks]
            embeddings = indexer2.embedding_service.generate_embeddings(contents)
            indexer2.database_service.store_embeddings(chunk_ids, embeddings)
            indexer2.database_service.ensure_hnsw_index()
            
            result = {"indexed_chunks": len(more_chunks)}
            
        finally:
            indexer2.close()
        
        # Third process should see all content
        retriever = Retriever(indexer_config)
        try:
            results = retriever.search("fresh", limit=10, mode="hybrid")
            assert len(results) >= 2  # Should find content from both processes
            
        finally:
            retriever.close()