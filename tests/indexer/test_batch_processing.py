"""Test batch processing in BM25 index storage."""

import logging
import os
import tempfile
from datetime import datetime
from typing import Generator, List

import pytest

from oboyu.indexer.database import Database
from oboyu.indexer.processor import Chunk

# Enable debug logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def temp_db() -> Generator[Database, None, None]:
    """Create a temporary database for testing."""
    # Create a temporary directory for the database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    
    db = Database(db_path=db_path)
    db.setup()
    
    yield db
    
    db.close()
    # Clean up
    if os.path.exists(db_path):
        os.unlink(db_path)
    # Remove any WAL files
    wal_path = db_path + ".wal"
    if os.path.exists(wal_path):
        os.unlink(wal_path)
    os.rmdir(temp_dir)


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Create sample chunks for testing."""
    chunks = []
    for i in range(100):
        chunk = Chunk(
            id=f"chunk_{i}",
            path=f"/test/file_{i // 10}.txt",
            title=f"Test File {i // 10}",
            content=f"This is test content for chunk {i}. " * 10,
            chunk_index=i % 10,
            language="en",
            created_at=datetime.now(),
            modified_at=datetime.now(),
            metadata={"test": True}
        )
        chunks.append(chunk)
    return chunks


def test_batch_processing_small_dataset(temp_db: Database, sample_chunks: List[Chunk]) -> None:
    """Test batch processing with small dataset."""
    # Store chunks first (required for foreign key constraints)
    temp_db.store_chunks(sample_chunks[:10])
    
    # Create test data
    vocab = {f"term_{i}": (i % 5 + 1, i * 2) for i in range(100)}
    inverted_index = {
        f"term_{i}": [(f"chunk_{j % 10}", j+1, [j*10+k for k in range(5)]) for j in range(3)]
        for i in range(100)
    }
    doc_stats = {f"chunk_{i}": (100+i, 50+i, 2.0) for i in range(10)}
    collection_stats = {
        "total_documents": 10,
        "total_terms": 1000,
        "avg_document_length": 100.0,
    }
    
    # Store with small batch size
    temp_db.store_bm25_index(
        vocabulary=vocab,
        inverted_index=inverted_index,
        document_stats=doc_stats,
        collection_stats=collection_stats,
        batch_size=50  # Small batch size
    )
    
    # Verify data was stored correctly
    vocab_count = temp_db.conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
    assert vocab_count == len(vocab)
    
    inv_count = temp_db.conn.execute("SELECT COUNT(*) FROM inverted_index").fetchone()[0]
    assert inv_count == sum(len(postings) for postings in inverted_index.values())
    
    doc_count = temp_db.conn.execute("SELECT COUNT(*) FROM document_stats").fetchone()[0]
    assert doc_count == len(doc_stats)


@pytest.mark.slow
def test_batch_processing_large_dataset(temp_db: Database, sample_chunks: List[Chunk]) -> None:
    """Test batch processing with large dataset."""
    # Store all chunks
    temp_db.store_chunks(sample_chunks)
    
    # Create large test data
    vocab = {f"term_{i}": (i % 10 + 1, i * 2) for i in range(50000)}
    inverted_index = {}
    for i in range(5000):  # Reduced to avoid memory issues in test
        inverted_index[f"term_{i}"] = [
            (f"chunk_{j % 100}", j+1, [j*10+k for k in range(5)])
            for j in range(10)
        ]
    
    doc_stats = {f"chunk_{i}": (100+i, 50+i, 2.0) for i in range(100)}
    collection_stats = {
        "total_documents": 100,
        "total_terms": 100000,
        "avg_document_length": 100.0,
    }
    
    # Store with default batch size
    temp_db.store_bm25_index(
        vocabulary=vocab,
        inverted_index=inverted_index,
        document_stats=doc_stats,
        collection_stats=collection_stats,
    )
    
    # Verify data was stored correctly
    vocab_count = temp_db.conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
    assert vocab_count == len(vocab)
    
    inv_count = temp_db.conn.execute("SELECT COUNT(*) FROM inverted_index").fetchone()[0]
    assert inv_count == sum(len(postings) for postings in inverted_index.values())
    
    # Verify positions are stored as arrays
    result = temp_db.conn.execute("""
        SELECT positions FROM inverted_index
        WHERE positions IS NOT NULL
        LIMIT 1
    """).fetchone()
    assert result is not None
    assert isinstance(result[0], list)
    assert all(isinstance(pos, int) for pos in result[0])


@pytest.mark.slow
def test_batch_processing_memory_efficiency(temp_db: Database, sample_chunks: List[Chunk]) -> None:
    """Test that batch processing doesn't load all data into memory at once."""
    # Store chunks
    temp_db.store_chunks(sample_chunks[:50])
    
    # Create very large vocabulary (but only referencing existing chunks)
    vocab = {f"term_{i}": (1, i) for i in range(100000)}
    
    # Create inverted index with only existing chunks
    inverted_index = {}
    for i in range(10000):
        inverted_index[f"term_{i}"] = [
            (f"chunk_{j % 50}", 1, [j])
            for j in range(5)
        ]
    
    doc_stats = {f"chunk_{i}": (100, 50, 2.0) for i in range(50)}
    collection_stats = {
        "total_documents": 50,
        "total_terms": 50000,
        "avg_document_length": 100.0,
    }
    
    # This should complete without memory errors
    temp_db.store_bm25_index(
        vocabulary=vocab,
        inverted_index=inverted_index,
        document_stats=doc_stats,
        collection_stats=collection_stats,
        batch_size=5000  # Reasonable batch size
    )
    
    # Verify completion
    vocab_count = temp_db.conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
    assert vocab_count == len(vocab)


def test_batch_processing_update_mode(temp_db: Database, sample_chunks: List[Chunk]) -> None:
    """Test batch processing in update mode (non-empty tables)."""
    # Store chunks
    temp_db.store_chunks(sample_chunks[:20])
    
    # Initial data
    vocab1 = {f"term_{i}": (1, i) for i in range(100)}
    inverted_index1 = {
        f"term_{i}": [(f"chunk_{j % 20}", 1, [j]) for j in range(2)]
        for i in range(100)
    }
    doc_stats1 = {f"chunk_{i}": (100, 50, 2.0) for i in range(10)}
    
    # Store initial data
    temp_db.store_bm25_index(
        vocabulary=vocab1,
        inverted_index=inverted_index1,
        document_stats=doc_stats1,
        collection_stats={"total_documents": 10, "total_terms": 1000, "avg_document_length": 100.0},
        batch_size=50
    )
    
    # Update with new data
    vocab2 = {f"term_{i}": (2, i*2) for i in range(50, 150)}
    inverted_index2 = {
        f"term_{i}": [(f"chunk_{j % 20}", 2, [j*2]) for j in range(2)]
        for i in range(50, 150)
    }
    doc_stats2 = {f"chunk_{i}": (200, 100, 2.0) for i in range(10, 20)}
    
    # Store update data (should use UPDATE path with batching)
    temp_db.store_bm25_index(
        vocabulary=vocab2,
        inverted_index=inverted_index2,
        document_stats=doc_stats2,
        collection_stats={"total_documents": 20, "total_terms": 2000, "avg_document_length": 100.0},
        batch_size=50
    )
    
    # Verify both initial and updated data exist
    vocab_count = temp_db.conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
    assert vocab_count == 150  # 100 initial + 50 new
    
    # Verify updates worked
    result = temp_db.conn.execute(
        "SELECT document_frequency FROM vocabulary WHERE term = 'term_50'"
    ).fetchone()
    assert result[0] == 2  # Should be updated value
