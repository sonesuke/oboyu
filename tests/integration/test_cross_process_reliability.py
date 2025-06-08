"""Integration tests for cross-process reliability.

This module tests database state management across process boundaries
to ensure that the cross-process reliability issues identified in issue #164
are properly resolved.
"""

import logging
import multiprocessing
import os
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from oboyu.indexer import Indexer
from oboyu.indexer.config.indexer_config import IndexerConfig, ModelConfig, ProcessingConfig
from oboyu.indexer.storage.database_service import DatabaseService
from oboyu.retriever import Retriever

logger = logging.getLogger(__name__)


def create_test_indexer_config(db_path: Path) -> IndexerConfig:
    """Create a minimal indexer configuration for testing."""
    return IndexerConfig(
        model=ModelConfig(
            embedding_model="cl-nagoya/ruri-v3-30m",
            batch_size=32
        ),
        processing=ProcessingConfig(
            chunk_size=500,
            chunk_overlap=50,
            max_workers=2,
            db_path=db_path
        )
    )


def create_test_files(test_dir: Path) -> List[Path]:
    """Create test files for indexing."""
    test_files = []
    
    # Create some test documents
    for i in range(5):
        test_file = test_dir / f"test_document_{i}.txt"
        test_file.write_text(f"This is test document {i}. " * 20)
        test_files.append(test_file)
    
    return test_files


def index_documents_in_process(db_path: str, test_dir: str) -> Dict[str, int]:
    """Index documents in a separate process."""
    try:
        config = create_test_indexer_config(Path(db_path))
        indexer = Indexer(config=config)
        
        # Create crawler results
        from oboyu.crawler.crawler import CrawlerResult
        crawler_results = []
        
        for test_file in Path(test_dir).glob("*.txt"):
            content = test_file.read_text()
            crawler_results.append(CrawlerResult(
                path=test_file,
                title=f"Test Document - {test_file.name}",
                content=content,
                language="en",
                metadata={}
            ))
        
        # Index the documents
        result = indexer.index_documents(crawler_results)
        indexer.close()
        
        return {
            "success": True,
            "indexed_chunks": result.get("indexed_chunks", 0),
            "total_documents": result.get("total_documents", 0),
            "process_id": os.getpid()
        }
    except Exception as e:
        logger.error(f"Indexing failed in process {os.getpid()}: {e}")
        return {
            "success": False,
            "error": str(e),
            "process_id": os.getpid()
        }


def query_database_in_process(db_path: str, query: str) -> Dict[str, int]:
    """Query the database in a separate process."""
    try:
        config = create_test_indexer_config(Path(db_path))
        retriever = Retriever(config=config)
        
        # Perform a search using the search method
        results = retriever.search(query, limit=5)
        
        retriever.close()
        
        return {
            "success": True,
            "result_count": len(results),
            "process_id": os.getpid()
        }
    except Exception as e:
        logger.error(f"Query failed in process {os.getpid()}: {e}")
        return {
            "success": False,
            "error": str(e),
            "process_id": os.getpid()
        }


def clear_database_in_process(db_path: str) -> Dict[str, int]:
    """Clear the database in a separate process."""
    try:
        config = create_test_indexer_config(Path(db_path))
        indexer = Indexer(config=config)
        
        # Clear the index
        indexer.clear_index()
        indexer.close()
        
        return {
            "success": True,
            "process_id": os.getpid()
        }
    except Exception as e:
        logger.error(f"Clear failed in process {os.getpid()}: {e}")
        return {
            "success": False,
            "error": str(e),
            "process_id": os.getpid()
        }


def concurrent_database_access(db_path: str, operation: str, thread_id: int) -> Dict[str, int]:
    """Perform concurrent database operations."""
    try:
        config = create_test_indexer_config(Path(db_path))
        database_service = DatabaseService(
            db_path=db_path
        )
        database_service.initialize()
        
        if operation == "read":
            # Perform a read operation
            count = database_service.get_chunk_count()
            result = {"operation": "read", "chunk_count": count}
        elif operation == "validate":
            # Validate database state
            validation = database_service.db_manager.validate_database_state()
            result = {"operation": "validate", "is_valid": validation["is_valid"]}
        else:
            result = {"operation": operation, "error": "Unknown operation"}
        
        database_service.close()
        
        return {
            "success": True,
            "thread_id": thread_id,
            "result": result
        }
    except Exception as e:
        logger.error(f"Concurrent access failed in thread {thread_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "thread_id": thread_id
        }


class TestCrossProcessReliability:
    """Test cross-process database reliability."""

    def test_index_in_process_a_query_in_process_b(self):
        """Test indexing in one process and querying in another."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_docs"
            test_dir.mkdir()
            db_path = Path(temp_dir) / "test.db"
            
            # Create test files
            create_test_files(test_dir)
            
            # Index documents in Process A
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(index_documents_in_process, str(db_path), str(test_dir))
                index_result = future.result()
            
            assert index_result["success"], f"Indexing failed: {index_result.get('error', 'Unknown error')}"
            assert index_result["indexed_chunks"] > 0
            assert index_result["total_documents"] == 5
            
            # Query database in Process B
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(query_database_in_process, str(db_path), "test document")
                query_result = future.result()
            
            assert query_result["success"], f"Query failed: {query_result.get('error', 'Unknown error')}"
            assert query_result["result_count"] > 0
            
            # Verify that different processes were used
            assert index_result["process_id"] != query_result["process_id"]

    def test_clear_index_query_workflow(self):
        """Test clear -> index -> query workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_docs"
            test_dir.mkdir()
            db_path = Path(temp_dir) / "test.db"
            
            # Create test files
            create_test_files(test_dir)
            
            # Step 1: Index some documents first
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(index_documents_in_process, str(db_path), str(test_dir))
                initial_result = future.result()
            
            assert initial_result["success"]
            assert initial_result["indexed_chunks"] > 0
            
            # Step 2: Clear the database
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(clear_database_in_process, str(db_path))
                clear_result = future.result()
            
            assert clear_result["success"], f"Clear failed: {clear_result.get('error', 'Unknown error')}"
            
            # Step 3: Index documents again
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(index_documents_in_process, str(db_path), str(test_dir))
                reindex_result = future.result()
            
            assert reindex_result["success"], f"Reindexing failed: {reindex_result.get('error', 'Unknown error')}"
            assert reindex_result["indexed_chunks"] > 0
            
            # Step 4: Query the database
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(query_database_in_process, str(db_path), "test document")
                final_query_result = future.result()
            
            assert final_query_result["success"], f"Final query failed: {final_query_result.get('error', 'Unknown error')}"
            assert final_query_result["result_count"] > 0

    def test_concurrent_database_access(self):
        """Test concurrent access to the database from multiple threads."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_docs"
            test_dir.mkdir()
            db_path = Path(temp_dir) / "test.db"
            
            # Create test files
            create_test_files(test_dir)
            
            # Index documents first
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(index_documents_in_process, str(db_path), str(test_dir))
                index_result = future.result()
            
            assert index_result["success"]
            
            # Test concurrent read access
            threads = []
            results = []
            
            def worker(thread_id: int):
                result = concurrent_database_access(str(db_path), "read", thread_id)
                results.append(result)
            
            # Create multiple threads for concurrent access
            for i in range(5):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify all operations succeeded
            assert len(results) == 5
            for result in results:
                assert result["success"], f"Concurrent access failed: {result.get('error', 'Unknown error')}"
                assert result["result"]["chunk_count"] > 0

    def test_database_state_validation_across_processes(self):
        """Test database state validation across processes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_docs"
            test_dir.mkdir()
            db_path = Path(temp_dir) / "test.db"
            
            # Create test files
            create_test_files(test_dir)
            
            # Index documents
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(index_documents_in_process, str(db_path), str(test_dir))
                index_result = future.result()
            
            assert index_result["success"]
            
            # Validate database state in multiple processes
            with ProcessPoolExecutor(max_workers=3) as executor:
                futures = []
                for i in range(3):
                    future = executor.submit(concurrent_database_access, str(db_path), "validate", i)
                    futures.append(future)
                
                validation_results = []
                for future in as_completed(futures):
                    result = future.result()
                    validation_results.append(result)
            
            # All validations should succeed
            assert len(validation_results) == 3
            for result in validation_results:
                assert result["success"], f"Validation failed: {result.get('error', 'Unknown error')}"
                assert result["result"]["is_valid"]

    def test_multiple_process_indexing_same_database(self):
        """Test multiple processes trying to index the same database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_docs"
            test_dir.mkdir()
            db_path = Path(temp_dir) / "test.db"
            
            # Create test files
            create_test_files(test_dir)
            
            # Try to index from multiple processes simultaneously
            with ProcessPoolExecutor(max_workers=3) as executor:
                futures = []
                for i in range(3):
                    future = executor.submit(index_documents_in_process, str(db_path), str(test_dir))
                    futures.append(future)
                
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
            
            # At least one should succeed, others might fail due to concurrent access
            successful_results = [r for r in results if r["success"]]
            assert len(successful_results) >= 1, "At least one indexing operation should succeed"
            
            # Verify the database is in a consistent state
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(query_database_in_process, str(db_path), "test document")
                query_result = future.result()
            
            assert query_result["success"]
            assert query_result["result_count"] > 0

    def test_database_corruption_recovery(self):
        """Test database recovery from corruption scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_docs"
            test_dir.mkdir()
            db_path = Path(temp_dir) / "test.db"
            
            # Create test files
            create_test_files(test_dir)
            
            # Index documents
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(index_documents_in_process, str(db_path), str(test_dir))
                index_result = future.result()
            
            assert index_result["success"]
            
            # Simulate corruption by writing invalid data to the database file
            with open(db_path, 'ab') as f:
                f.write(b"CORRUPTED_DATA" * 100)
            
            # Try to query the database - should either work or fail gracefully
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(query_database_in_process, str(db_path), "test document")
                query_result = future.result()
            
            # The system should either recover or fail gracefully (not crash)
            assert "success" in query_result
            if not query_result["success"]:
                # If it fails, the error should be informative
                assert "error" in query_result
                assert len(query_result["error"]) > 0


@pytest.mark.slow
class TestCrossProcessReliabilityExtended:
    """Extended tests for cross-process reliability (marked as slow)."""

    def test_long_running_cross_process_operations(self):
        """Test long-running operations across processes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_docs"
            test_dir.mkdir()
            db_path = Path(temp_dir) / "test.db"
            
            # Create many test files
            for i in range(20):
                test_file = test_dir / f"large_document_{i}.txt"
                test_file.write_text(f"This is a large test document {i}. " * 500)
            
            # Index documents in one process
            start_time = time.time()
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(index_documents_in_process, str(db_path), str(test_dir))
                index_result = future.result()
            
            indexing_time = time.time() - start_time
            logger.info(f"Indexing took {indexing_time:.2f} seconds")
            
            assert index_result["success"]
            assert index_result["indexed_chunks"] > 100  # Should have many chunks
            
            # Query in another process
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(query_database_in_process, str(db_path), "large test document")
                query_result = future.result()
            
            assert query_result["success"]
            assert query_result["result_count"] > 0

    def test_stress_test_concurrent_access(self):
        """Stress test with many concurrent operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_docs"
            test_dir.mkdir()
            db_path = Path(temp_dir) / "test.db"
            
            # Create test files
            create_test_files(test_dir)
            
            # Index documents first
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(index_documents_in_process, str(db_path), str(test_dir))
                index_result = future.result()
            
            assert index_result["success"]
            
            # Perform many concurrent read operations
            threads = []
            results = []
            
            def stress_worker(thread_id: int):
                for i in range(10):  # Each thread does 10 operations
                    result = concurrent_database_access(str(db_path), "read", thread_id)
                    results.append(result)
                    time.sleep(0.01)  # Small delay between operations
            
            # Create many threads
            for i in range(10):
                thread = threading.Thread(target=stress_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify most operations succeeded
            successful_results = [r for r in results if r["success"]]
            success_rate = len(successful_results) / len(results)
            
            # We expect at least 90% success rate
            assert success_rate >= 0.9, f"Success rate too low: {success_rate:.2%}"