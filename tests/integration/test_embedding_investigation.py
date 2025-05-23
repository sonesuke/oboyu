"""Test to investigate if the embedding model itself has the issue."""

import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.embedding import EmbeddingGenerator
from oboyu.indexer.indexer import Indexer


def test_embedding_similarity():
    """Test direct embedding similarity to understand the scores."""
    
    # Create embedding generator
    generator = EmbeddingGenerator(model_name="cl-nagoya/ruri-v3-30m")
    
    # Test texts
    python_text = "Python programming language is powerful and versatile for coding."
    ml_text = "Machine learning algorithms use complex mathematical models and statistics."
    
    # Generate embeddings
    python_embedding = generator.generate_query_embedding("Python programming")
    ml_embedding = generator.generate_query_embedding("machine learning")
    
    # Generate document embeddings
    python_doc_embedding = generator.model.encode(python_text, normalize_embeddings=True)
    ml_doc_embedding = generator.model.encode(ml_text, normalize_embeddings=True)
    
    # Calculate similarities
    def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        """Calculate cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def cosine_distance_to_similarity(distance: float) -> float:
        """Convert cosine distance to similarity score (as done in database.py)."""
        return 1.0 - (distance / 2.0)
    
    def cosine_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        """Calculate cosine distance."""
        return 1.0 - cosine_similarity(a, b)
    
    # Test query to appropriate document
    python_query_to_python_doc = cosine_similarity(python_embedding, python_doc_embedding)
    ml_query_to_ml_doc = cosine_similarity(ml_embedding, ml_doc_embedding)
    
    # Test query to inappropriate document (cross-domain)
    python_query_to_ml_doc = cosine_similarity(python_embedding, ml_doc_embedding)
    ml_query_to_python_doc = cosine_similarity(ml_embedding, python_doc_embedding)
    
    print(f"\n=== Direct Embedding Similarities ===")
    print(f"Python query -> Python doc: {python_query_to_python_doc:.3f}")
    print(f"ML query -> ML doc: {ml_query_to_ml_doc:.3f}")
    print(f"Python query -> ML doc: {python_query_to_ml_doc:.3f}")
    print(f"ML query -> Python doc: {ml_query_to_python_doc:.3f}")
    
    # Convert to distance and then to database similarity scores
    python_query_to_ml_doc_distance = cosine_distance(python_embedding, ml_doc_embedding)
    db_similarity_score = cosine_distance_to_similarity(python_query_to_ml_doc_distance)
    
    print(f"\n=== Database-style Scoring ===")
    print(f"Python query -> ML doc distance: {python_query_to_ml_doc_distance:.3f}")
    print(f"Database similarity score: {db_similarity_score:.3f}")
    
    # The issue might be that the model actually finds some similarity between
    # "Python programming" and "machine learning" due to both being tech-related
    
    # Test with completely unrelated content
    unrelated_text = "The weather is sunny today and birds are singing loudly."
    unrelated_embedding = generator.model.encode(unrelated_text, normalize_embeddings=True)
    
    python_query_to_unrelated = cosine_similarity(python_embedding, unrelated_embedding)
    unrelated_distance = cosine_distance(python_embedding, unrelated_embedding)
    unrelated_db_score = cosine_distance_to_similarity(unrelated_distance)
    
    print(f"\n=== Completely Unrelated Content ===")
    print(f"Python query -> Weather doc similarity: {python_query_to_unrelated:.3f}")
    print(f"Python query -> Weather doc distance: {unrelated_distance:.3f}")
    print(f"Database similarity score: {unrelated_db_score:.3f}")
    
    # The test might be wrong - 0.7 similarity between "Python programming" and 
    # "machine learning" might be legitimate for this model
    
    # Let's see if the model considers them related
    assert python_query_to_python_doc > python_query_to_ml_doc, "Should prefer exact match"
    assert python_query_to_unrelated < python_query_to_ml_doc, "Tech topics should be more similar than weather"


def test_clear_preserves_embeddings_correctly():
    """Test that clear+reindex produces identical embeddings."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        test_dir = Path(tmpdir) / "docs"
        test_dir.mkdir()
        
        # Create test document
        (test_dir / "test.txt").write_text("Machine learning algorithms use complex mathematical models.")
        
        config = IndexerConfig(config_dict={"indexer": {"db_path": str(db_path)}})
        indexer = Indexer(config=config)
        
        # Index once
        indexer.index_directory(test_dir)
        results1 = indexer.search("machine learning", limit=1)
        
        # Clear and re-index
        indexer.clear_index()
        indexer.index_directory(test_dir)
        results2 = indexer.search("machine learning", limit=1)
        
        # Results should be identical
        assert len(results1) == len(results2) == 1
        assert abs(results1[0].score - results2[0].score) < 0.001, f"Scores should be identical: {results1[0].score:.3f} vs {results2[0].score:.3f}"
        
        indexer.close()