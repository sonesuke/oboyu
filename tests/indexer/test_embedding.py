"""Tests for the embedding generator functionality."""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oboyu.indexer.embedding import EmbeddingCache, EmbeddingGenerator
from oboyu.indexer.processor import Chunk


@pytest.fixture(scope="module")
def shared_embedding_generator():
    """Module-scoped fixture to share embedding generator across tests."""
    # Note: The model will be loaded lazily on first use
    return EmbeddingGenerator(
        model_name="cl-nagoya/ruri-v3-30m",
        batch_size=2,
        use_cache=True,  # Enable caching
        use_onnx=False,  # Use PyTorch backend for tests
    )


class TestEmbeddingCache:
    """Test cases for the embedding cache."""

    def test_cache_operations(self) -> None:
        """Test basic cache operations."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize cache
            cache = EmbeddingCache(cache_dir=temp_dir)
            
            # Test data
            text = "This is a test text"
            model_name = "test-model"
            embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
            
            # Initially cache should be empty
            assert cache.get(text, model_name) is None
            
            # Set cache entry
            cache.set(text, model_name, embedding)
            
            # Verify cache entry was saved
            cached_embedding = cache.get(text, model_name)
            assert cached_embedding is not None
            assert np.array_equal(cached_embedding, embedding)
            
            # Test cache key uniqueness
            text2 = "This is another text"
            assert cache.get(text2, model_name) is None
            
            # Verify changing model name creates different cache key
            assert cache.get(text, "other-model") is None

    def test_cache_resilience(self) -> None:
        """Test cache resilience against corrupted files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(cache_dir=temp_dir)
            
            # Create a corrupted cache file
            cache_key = cache._get_cache_key("corrupted", "test-model")
            cache_file = Path(temp_dir) / f"{cache_key}.pkl"
            
            with open(cache_file, "wb") as f:
                f.write(b"corrupted data")
            
            # Getting from corrupted file should return None, not raise an exception
            assert cache.get("corrupted", "test-model") is None


class TestEmbeddingGenerator:
    """Test cases for the embedding generator using the actual model.
    
    These tests will download the actual model (cl-nagoya/ruri-v3-30m) on first run.
    """

    def test_generate_embeddings(self, shared_embedding_generator) -> None:
        """Test generating embeddings for chunks."""
        # Use shared generator to avoid repeated model loading
        generator = shared_embedding_generator
        
        # Create test chunks with both English and Japanese content
        chunks = [
            Chunk(
                id="1",
                path=Path("/test/doc1.txt"),
                title="Test 1",
                content="This is test document one.",
                chunk_index=0,
                language="en",
                created_at=datetime.now(),
                modified_at=datetime.now(),
                metadata={},
                prefix_content="検索文書: This is test document one.",
            ),
            Chunk(
                id="2",
                path=Path("/test/doc2.txt"),
                title="テスト2",  # Japanese title
                content="これは日本語のテスト文書です。",  # Japanese content: "This is a Japanese test document."
                chunk_index=0,
                language="ja",
                created_at=datetime.now(),
                modified_at=datetime.now(),
                metadata={"language": "Japanese"},
                prefix_content="検索文書: これは日本語のテスト文書です。",
            ),
        ]
        
        # Generate embeddings
        embeddings = generator.generate_embeddings(chunks)
        
        # Verify results
        assert len(embeddings) == 2
        for embedding in embeddings:
            assert len(embedding) == 4  # (id, chunk_id, embedding, timestamp)
            assert isinstance(embedding[0], str)  # id
            assert embedding[1] in ["1", "2"]  # chunk_id
            assert isinstance(embedding[2], np.ndarray)  # embedding
            assert embedding[2].shape[0] == generator.dimensions  # embedding dimensions
            assert isinstance(embedding[3], datetime)  # timestamp

    def test_generate_query_embedding(self, shared_embedding_generator) -> None:
        """Test generating embedding for a search query."""
        # Use shared generator to avoid repeated model loading
        generator = shared_embedding_generator
        
        # Generate query embedding with Japanese text
        query = "意味検索のテストクエリです"  # "This is a semantic search test query" in Japanese
        embedding = generator.generate_query_embedding(query)
        
        # Verify result
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == generator.dimensions
        assert not np.isnan(embedding).any()  # No NaN values


class TestEmbeddingGeneratorMocked:
    """Test cases for the embedding generator using mocks."""

    def test_generate_embeddings_with_cache(self) -> None:
        """Test embedding generation with cache."""
        # Mock SentenceTransformer
        with patch("oboyu.indexer.embedding.SentenceTransformer") as mock_model_class:
            # Set up the mock model
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384], dtype=np.float32)
            mock_model_class.return_value = mock_model
            
            # Create temporary cache directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Initialize generator with cache and model directory
                generator = EmbeddingGenerator(
                    model_name="test-model",
                    batch_size=2,
                    use_cache=True,
                    cache_dir=temp_dir,
                    model_dir=temp_dir,  # Use same temp dir for both
                )
                
                # Create test chunks
                chunks = [
                    Chunk(
                        id="1",
                        path=Path("/test/doc1.txt"),
                        title="Test 1",
                        content="This is test document one.",
                        chunk_index=0,
                        language="en",
                        created_at=datetime.now(),
                        modified_at=datetime.now(),
                        metadata={},
                        prefix_content="検索文書: This is test document one.",
                    ),
                    Chunk(
                        id="2",
                        path=Path("/test/doc2.txt"),
                        title="Test 2",
                        content="This is test document two.",
                        chunk_index=0,
                        language="en",
                        created_at=datetime.now(),
                        modified_at=datetime.now(),
                        metadata={},
                        prefix_content="検索文書: This is test document two.",
                    ),
                ]
                
                # First call should use the model
                embeddings1 = generator.generate_embeddings(chunks)
                assert len(embeddings1) == 2
                assert mock_model.encode.call_count == 1
                
                # Second call with same chunks should use cache
                embeddings2 = generator.generate_embeddings(chunks)
                assert len(embeddings2) == 2
                assert mock_model.encode.call_count == 1  # Call count unchanged
                
                # Check that the embedding IDs are different but the vectors are the same
                assert embeddings1[0][0] != embeddings2[0][0]  # Different IDs
                assert np.array_equal(embeddings1[0][2], embeddings2[0][2])  # Same vectors