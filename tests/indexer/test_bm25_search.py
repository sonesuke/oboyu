"""Tests for BM25 search functionality."""

import pytest
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

from oboyu.indexer.database import Database
from oboyu.indexer.indexer import Indexer
from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.processor import Chunk


class TestBM25Search:
    """Test cases for BM25 search functionality."""

    @pytest.fixture
    def test_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            yield db_path

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        from datetime import datetime
        now = datetime.now()
        return [
            Chunk(
                id="chunk1",
                path=Path("/docs/python.md"),
                title="Python Programming",
                content="Python is a high-level programming language known for its simplicity.",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={"title": "Python Programming"},
            ),
            Chunk(
                id="chunk2",
                path=Path("/docs/java.md"),
                title="Java Programming",
                content="Java is a popular programming language used for enterprise applications.",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={"title": "Java Programming"},
            ),
            Chunk(
                id="chunk3",
                path=Path("/docs/ml.md"),
                title="Machine Learning",
                content="Machine learning with Python involves libraries like TensorFlow and PyTorch.",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={"title": "Machine Learning"},
            ),
            Chunk(
                id="chunk4",
                path=Path("/docs/web.md"),
                title="Web Development",
                content="Web development can be done with Python using frameworks like Django.",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={"title": "Web Development"},
            ),
        ]

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        mock = Mock()
        mock.encode.return_value = [[0.5] * 384]  # Return mock embedding
        return mock

    def test_database_bm25_operations(self, test_db_path):
        """Test BM25 database operations."""
        db = Database(test_db_path)
        db.setup()
        
        # First create some chunks in the database
        from datetime import datetime
        now = datetime.now()
        test_chunks = [
            Chunk(
                id="chunk1",
                path=Path("/test1.md"),
                title="Test 1",
                content="Python programming language",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
            Chunk(
                id="chunk2",
                path=Path("/test2.md"),
                title="Test 2",
                content="Java programming language",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
            Chunk(
                id="chunk3",
                path=Path("/test3.md"),
                title="Test 3",
                content="Python data science",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
        ]
        db.store_chunks(test_chunks)
        
        # Store BM25 index data
        vocabulary = {
            "python": (2, 3),  # (doc_freq, collection_freq)
            "programming": (2, 2),
            "language": (2, 2),
            "java": (1, 1),
            "data": (1, 1),
            "science": (1, 1),
        }
        inverted_index = {
            "python": [
                ("chunk1", 1, []),
                ("chunk3", 1, []),
            ],
            "programming": [
                ("chunk1", 1, []),
                ("chunk2", 1, []),
            ],
            "language": [
                ("chunk1", 1, []),
                ("chunk2", 1, []),
            ],
            "java": [
                ("chunk2", 1, []),
            ],
            "data": [
                ("chunk3", 1, []),
            ],
            "science": [
                ("chunk3", 1, []),
            ],
        }
        document_stats = {
            "chunk1": (3, 3, 1.0),  # (total_terms, unique_terms, avg_term_freq)
            "chunk2": (3, 3, 1.0),
            "chunk3": (3, 3, 1.0),
        }
        collection_stats = {
            "total_documents": 3,
            "total_terms": 9,
            "avg_document_length": 3.0,
        }
        
        # Store the index
        db.store_bm25_index(vocabulary, inverted_index, document_stats, collection_stats)
        
        # Test BM25 search
        results = db.search_bm25(
            query_terms=["python", "programming"],
            bm25_scores={
                "chunk1": 2.5,
                "chunk2": 1.2,
                "chunk3": 1.8,
            },
            limit=2,
        )
        
        assert len(results) == 2
        assert results[0].chunk_id == "chunk1"  # Highest score
        assert results[0].score == 2.5
        assert results[1].chunk_id == "chunk3"  # Second highest
        assert results[1].score == 1.8
        
        db.close()

    def test_indexer_bm25_search_mode(self, test_db_path, sample_chunks, mock_embedding_model):
        """Test Indexer with BM25 search mode."""
        config = IndexerConfig()
        config.update({
            "use_bm25": True,
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
        })
        
        # Create indexer with mocked embedding model
        with patch("oboyu.indexer.indexer.EmbeddingModel") as mock_embed_class:
            mock_embed_class.return_value = mock_embedding_model
            indexer = Indexer(database_path=test_db_path, config=config)
            
            # Store chunks in database
            indexer.database.store_chunks(sample_chunks)
            
            # Build BM25 index
            indexer._build_bm25_index_with_progress(sample_chunks, None)
            
            # Test BM25 search
            results = indexer.search("python programming", limit=3, mode="bm25")
            
            assert len(results) > 0
            assert all(hasattr(r, "chunk_id") for r in results)
            assert all(hasattr(r, "score") for r in results)
            
            indexer.close()

    def test_indexer_hybrid_search_mode(self, test_db_path, sample_chunks, mock_embedding_model):
        """Test Indexer with hybrid search mode."""
        config = IndexerConfig()
        config.update({
            "use_bm25": True,
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
        })
        
        # Create indexer with mocked embedding model
        with patch("oboyu.indexer.indexer.EmbeddingModel") as mock_embed_class:
            mock_embed_class.return_value = mock_embedding_model
            indexer = Indexer(database_path=test_db_path, config=config)
            
            # Store chunks in database
            indexer.database.store_chunks(sample_chunks)
            
            # Build BM25 index
            indexer._build_bm25_index_with_progress(sample_chunks, None)
            
            # Test hybrid search
            results = indexer.search(
                "python programming", 
                limit=3, 
                mode="hybrid",
                vector_weight=0.6,
                bm25_weight=0.4,
            )
            
            assert len(results) > 0
            assert all(hasattr(r, "chunk_id") for r in results)
            assert all(hasattr(r, "score") for r in results)
            
            indexer.close()

    def test_search_mode_validation(self, test_db_path, mock_embedding_model):
        """Test search mode validation."""
        config = IndexerConfig()
        
        with patch("oboyu.indexer.indexer.EmbeddingModel") as mock_embed_class:
            mock_embed_class.return_value = mock_embedding_model
            indexer = Indexer(database_path=test_db_path, config=config)
            
            # Test invalid mode
            with pytest.raises(ValueError, match="Invalid search mode"):
                indexer.search("test query", mode="invalid_mode")
            
            indexer.close()

    def test_bm25_search_without_index(self, test_db_path, mock_embedding_model):
        """Test BM25 search when index is not built."""
        config = IndexerConfig()
        config.update({"use_bm25": True})
        
        with patch("oboyu.indexer.indexer.EmbeddingModel") as mock_embed_class:
            mock_embed_class.return_value = mock_embedding_model
            indexer = Indexer(database_path=test_db_path, config=config)
            
            # Try BM25 search without building index
            results = indexer.search("python", mode="bm25")
            
            # Should return empty results
            assert results == []
            
            indexer.close()

    def test_combine_search_results(self, test_db_path, mock_embedding_model):
        """Test combining vector and BM25 search results."""
        config = IndexerConfig()
        
        with patch("oboyu.indexer.indexer.EmbeddingModel") as mock_embed_class:
            mock_embed_class.return_value = mock_embedding_model
            indexer = Indexer(database_path=test_db_path, config=config)
            
            # Create mock search results
            vector_results = [
                Mock(chunk_id="chunk1", score=0.9, content="content1", file_path="/path1", metadata={}),
                Mock(chunk_id="chunk2", score=0.8, content="content2", file_path="/path2", metadata={}),
                Mock(chunk_id="chunk3", score=0.7, content="content3", file_path="/path3", metadata={}),
            ]
            
            bm25_results = [
                Mock(chunk_id="chunk2", score=2.5, content="content2", file_path="/path2", metadata={}),
                Mock(chunk_id="chunk3", score=2.0, content="content3", file_path="/path3", metadata={}),
                Mock(chunk_id="chunk4", score=1.5, content="content4", file_path="/path4", metadata={}),
            ]
            
            # Test combination with default weights
            combined = indexer._combine_search_results(
                vector_results, bm25_results, vector_weight=0.7, bm25_weight=0.3
            )
            
            # Check that all unique chunks are included
            chunk_ids = [r.chunk_id for r in combined]
            assert len(set(chunk_ids)) == 4  # 4 unique chunks
            
            # Check that results are sorted by combined score
            scores = [r.score for r in combined]
            assert scores == sorted(scores, reverse=True)
            
            indexer.close()

    def test_clear_bm25_index(self, test_db_path):
        """Test clearing BM25 index."""
        db = Database(test_db_path)
        
        # Store some BM25 data
        vocabulary = ["test", "word"]
        posting_lists = {"test": [{"chunk_id": "chunk1", "term_freq": 1}]}
        doc_stats = {"chunk1": 5}
        collection_stats = {"total_docs": 1, "avg_doc_length": 5.0}
        
        db.store_bm25_index(vocabulary, posting_lists, doc_stats, collection_stats)
        
        # Clear BM25 index
        db.clear_bm25_index()
        
        # Verify tables are empty
        conn = db.get_connection()
        
        # Check vocabulary table
        result = conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()
        assert result[0] == 0
        
        # Check other tables
        result = conn.execute("SELECT COUNT(*) FROM inverted_index").fetchone()
        assert result[0] == 0
        
        result = conn.execute("SELECT COUNT(*) FROM document_stats").fetchone()
        assert result[0] == 0
        
        result = conn.execute("SELECT COUNT(*) FROM collection_stats").fetchone()
        assert result[0] == 0
        
        conn.close()
        db.close()

    def test_japanese_bm25_search(self, test_db_path, mock_embedding_model):
        """Test BM25 search with Japanese text."""
        config = IndexerConfig()
        config.update({
            "use_bm25": True,
            "use_japanese_tokenizer": True,
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
        })
        
        # Create Japanese chunks
        from datetime import datetime
        now = datetime.now()
        japanese_chunks = [
            Chunk(
                id="chunk1",
                path=Path("/docs/python_ja.md"),
                title="Python入門",
                content="Pythonは初心者にも優しいプログラミング言語です。",
                chunk_index=0,
                language="ja",
                created_at=now,
                modified_at=now,
                metadata={"title": "Python入門"},
            ),
            Chunk(
                id="chunk2",
                path=Path("/docs/ml_ja.md"),
                title="機械学習",
                content="機械学習ではPythonがよく使われています。",
                chunk_index=0,
                language="ja",
                created_at=now,
                modified_at=now,
                metadata={"title": "機械学習"},
            ),
        ]
        
        with patch("oboyu.indexer.indexer.EmbeddingModel") as mock_embed_class:
            mock_embed_class.return_value = mock_embedding_model
            
            # Mock the tokenizer to avoid dependency on fugashi
            with patch("oboyu.indexer.tokenizer.create_tokenizer") as mock_tokenizer:
                mock_tokenizer.return_value.tokenize.side_effect = lambda text: text.split()
                
                indexer = Indexer(database_path=test_db_path, config=config)
                
                # Store Japanese chunks
                indexer.database.store_chunks(japanese_chunks)
                
                # Build BM25 index
                indexer._build_bm25_index_with_progress(japanese_chunks, None)
                
                # Search in Japanese
                results = indexer.search("Python プログラミング", mode="bm25")
                
                # Should return results
                assert isinstance(results, list)
                
                indexer.close()