#!/usr/bin/env python3
"""Test script for database performance."""

import time
import tempfile
from datetime import datetime
from pathlib import Path
import random
import string
import uuid

import numpy as np
from oboyu.indexer.storage.database_service import DatabaseService as Database
from oboyu.common.types import Chunk


def random_string(length=100):
    """Generate a random string of fixed length."""
    return ''.join(random.choice(string.ascii_letters + string.digits + ' ' * 5) for _ in range(length))


def generate_test_chunks(count=1000):
    """Generate test chunks with random data."""
    now = datetime.now()
    chunks = []
    for i in range(count):
        chunks.append(
            Chunk(
                id=f"test-chunk-{i}-{uuid.uuid4().hex[:8]}",
                path=Path(f"/test/doc{i}.txt"),
                title=f"Test Document {i}",
                content=random_string(500),  # Larger content
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={"source": "test", "index": i},
                prefix_content=f"検索文書: Test document {i}",
            )
        )
    return chunks


def generate_embeddings(chunks, dimensions=256):
    """Generate random embeddings for chunks."""
    now = datetime.now()
    embeddings = []
    for chunk in chunks:
        embeddings.append(
            (f"emb-{chunk.id}", chunk.id, np.random.rand(dimensions).astype(np.float32), now)
        )
    return embeddings


def run_performance_test():
    """Run performance test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "performance_test.db"
        
        # Initialize database
        print(f"Initializing database at {db_path}...")
        db = Database(db_path=db_path, embedding_dimensions=256)
        db.setup()
        
        # Generate chunks and embeddings
        print("Generating test data...")
        chunks = generate_test_chunks(count=1000)
        embeddings = generate_embeddings(chunks)
        
        # Store chunks
        print("Storing chunks...")
        start_time = time.time()
        db.store_chunks(chunks)
        chunk_time = time.time() - start_time
        print(f"Stored {len(chunks)} chunks in {chunk_time:.2f} seconds ({len(chunks)/chunk_time:.2f} chunks/second)")
        
        # Store embeddings
        print("Storing embeddings...")
        start_time = time.time()
        db.store_embeddings(embeddings, "test-model")
        embedding_time = time.time() - start_time
        print(f"Stored {len(embeddings)} embeddings in {embedding_time:.2f} seconds ({len(embeddings)/embedding_time:.2f} embeddings/second)")
        
        # Search with random vector
        print("Testing search...")
        query_vector = np.random.rand(256).astype(np.float32)
        start_time = time.time()
        results = db.search(query_vector, limit=10)
        search_time = time.time() - start_time
        print(f"Searched database with {len(chunks)} documents in {search_time:.4f} seconds")
        print(f"Found {len(results)} results")
        
        # Close database
        db.close()
        print("Test completed successfully!")


if __name__ == "__main__":
    run_performance_test()