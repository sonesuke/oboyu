"""Test to verify search results are ordered correctly (descending by score)."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from oboyu.indexer.database import Database
from oboyu.indexer.processor import Chunk
from datetime import datetime

def test_search_results_descending_order():
    """Test that search results are ordered in descending order by score."""
    # Create a temporary database path
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Initialize database
        db = Database(db_path=db_path, embedding_dimensions=3)
        db.setup()
        
        # Create test chunks
        chunks = []
        for i in range(5):
            chunk = Chunk(
                id=f"chunk_{i}",
                path=Path(f"/test/doc_{i}.txt"),
                title=f"Document {i}",
                content=f"Test content for document {i}",
                chunk_index=0,
                language="en",
                created_at=datetime.now(),
                modified_at=datetime.now(),
                metadata={}
            )
            chunks.append(chunk)
        
        # Store chunks
        db.store_chunks(chunks)
        
        # Create embeddings with different vectors that will have different distances
        # to our query vector
        embeddings = []
        vectors = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Should be closest
            np.array([0.8, 0.2, 0.0], dtype=np.float32),  # Second closest
            np.array([0.5, 0.5, 0.0], dtype=np.float32),  # Third
            np.array([0.2, 0.8, 0.0], dtype=np.float32),  # Fourth
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Farthest
        ]
        
        for i, vector in enumerate(vectors):
            # Normalize the vector
            vector = vector / np.linalg.norm(vector)
            embeddings.append((
                f"embedding_{i}",
                f"chunk_{i}",
                vector,
                datetime.now()
            ))
        
        # Store embeddings
        db.store_embeddings(embeddings, "test_model")
        
        # Search with a query vector close to [1, 0, 0]
        query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Search for top 3 results
        results = db.search(query_vector, limit=3)
        
        # Verify results are ordered correctly (descending by score)
        scores = [result['score'] for result in results]
        
        # Check that scores are in descending order (higher is better)
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
            f"Results should be ordered in descending order by score, but got: {scores}"
        
        # Verify scores are in expected range (0-1)
        assert all(0 <= score <= 1 for score in scores), \
            f"Scores should be between 0 and 1, but got: {scores}"
        
        # Also verify that we got the expected documents
        # With our similarity scores, higher values mean better matches
        # The search function should return results with higher scores (better matches) first
        expected_order = ["Document 0", "Document 1", "Document 2"]
        actual_order = [result['title'] for result in results]
        
        assert actual_order == expected_order, \
            f"Expected documents in order {expected_order}, but got {actual_order}"
        
        # Clean up
        db.close()