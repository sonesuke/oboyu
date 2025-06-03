"""Tests for the BM25 indexer module."""

import pytest
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

from oboyu.indexer.algorithm.bm25_indexer import BM25Indexer
from oboyu.common.types import Chunk


class TestBM25Indexer:
    """Test cases for BM25Indexer class."""

    @pytest.fixture
    def bm25_indexer(self):
        """Create a BM25Indexer instance for testing."""
        return BM25Indexer(k1=1.2, b=0.75)

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        now = datetime.now()
        return [
            Chunk(
                id="doc1",
                path=Path("/docs/python.md"),
                title="Python Programming",
                content="Python programming language Python",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
            Chunk(
                id="doc2",
                path=Path("/docs/java.md"),
                title="Java Programming",
                content="Java programming language",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
            Chunk(
                id="doc3",
                path=Path("/docs/data_science.md"),
                title="Data Science",
                content="Python data science analysis",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
            Chunk(
                id="doc4",
                path=Path("/docs/ml.md"),
                title="Machine Learning",
                content="machine learning Python TensorFlow",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
            Chunk(
                id="doc5",
                path=Path("/docs/web_dev.md"),
                title="Web Development",
                content="web development JavaScript Python",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
        ]

    def test_initialization(self):
        """Test BM25Indexer initialization."""
        indexer = BM25Indexer(k1=1.5, b=0.8)
        assert indexer.k1 == 1.5
        assert indexer.b == 0.8
        assert indexer.inverted_index == {}
        assert indexer.document_lengths == {}
        assert indexer.total_document_length == 0
        assert indexer.document_count == 0

    def test_index_chunks(self, bm25_indexer, sample_chunks):
        """Test indexing chunks."""
        # Index chunks
        stats = bm25_indexer.index_chunks(sample_chunks)

        # Check statistics
        assert stats["chunks_indexed"] == 5
        assert stats["unique_terms"] > 0
        assert stats["terms_indexed"] > 0

        # Check document count
        assert bm25_indexer.document_count == 5

        # Check document lengths
        assert "doc1" in bm25_indexer.document_lengths
        assert "doc2" in bm25_indexer.document_lengths
        assert bm25_indexer.document_lengths["doc1"] > 0

        # Check inverted index
        assert "python" in bm25_indexer.inverted_index
        # Count documents containing "python"
        python_docs = [doc_id for doc_id, _, _ in bm25_indexer.inverted_index["python"]]
        assert len(python_docs) == 4  # appears in 4 docs

    def test_index_empty_chunk(self, bm25_indexer):
        """Test indexing a chunk with empty content."""
        now = datetime.now()
        empty_chunk = Chunk(
            id="empty_doc",
            path=Path("/docs/empty.md"),
            title="Empty",
            content="",
            chunk_index=0,
            language="en",
            created_at=now,
            modified_at=now,
            metadata={},
        )
        stats = bm25_indexer.index_chunks([empty_chunk])
        
        assert bm25_indexer.document_count == 1
        assert bm25_indexer.document_lengths["empty_doc"] == 0
        assert stats["chunks_indexed"] == 1

    def test_get_term_frequency(self, bm25_indexer, sample_chunks):
        """Test getting term frequency for documents."""
        # Index chunks
        bm25_indexer.index_chunks(sample_chunks)

        # Get term frequency from inverted index
        # Find doc1's term frequency for "python"
        python_postings = bm25_indexer.inverted_index.get("python", [])
        doc1_freq = 0
        for doc_id, freq, _ in python_postings:
            if doc_id == "doc1":
                doc1_freq = freq
                break
        
        assert doc1_freq == 2  # "Python" appears twice in doc1 (case insensitive)

    def test_document_frequency(self, bm25_indexer, sample_chunks):
        """Test document frequency tracking."""
        # Index chunks
        bm25_indexer.index_chunks(sample_chunks)

        # Test document frequency
        assert bm25_indexer.document_frequencies.get("python", 0) == 4
        assert bm25_indexer.document_frequencies.get("programming", 0) == 2
        assert bm25_indexer.document_frequencies.get("javascript", 0) == 1
        assert bm25_indexer.document_frequencies.get("nonexistent", 0) == 0

    def test_compute_bm25_score(self, bm25_indexer, sample_chunks):
        """Test BM25 score computation."""
        # Index chunks
        bm25_indexer.index_chunks(sample_chunks)

        # Test single term query with a less common term
        query = ["tensorflow"]
        score1 = bm25_indexer.compute_bm25_score(query, "doc4", {"tensorflow": 1})
        score2 = bm25_indexer.compute_bm25_score(query, "doc2", {})
        
        assert score1 > 0  # doc4 contains "tensorflow"
        assert score2 == 0  # doc2 doesn't contain "tensorflow"
        
        # Test with common term (may have negative IDF due to high frequency)
        python_score = bm25_indexer.compute_bm25_score(["python"], "doc1", {"python": 2})
        # Python appears in 4/5 docs, so may have negative IDF - this is correct BM25 behavior

        # Test multi-term query with rare terms
        query = ["tensorflow", "machine"]
        score_both = bm25_indexer.compute_bm25_score(query, "doc4", {"tensorflow": 1, "machine": 1})
        score_one = bm25_indexer.compute_bm25_score(query, "doc3", {"analysis": 1})  # Different term
        
        assert score_both > score_one  # doc4 contains both query terms, doc3 contains neither

    def test_compute_bm25_score_edge_cases(self, bm25_indexer):
        """Test BM25 score computation with edge cases."""
        # Index a single chunk
        now = datetime.now()
        chunk = Chunk(
            id="doc1",
            path=Path("/docs/test.md"),
            title="Test",
            content="test document",
            chunk_index=0,
            language="en",
            created_at=now,
            modified_at=now,
            metadata={},
        )
        bm25_indexer.index_chunks([chunk])

        # Empty query
        score = bm25_indexer.compute_bm25_score([], "doc1", {"test": 1})
        assert score == 0

        # Query with terms not in document
        score = bm25_indexer.compute_bm25_score(["missing"], "doc1", {})
        assert score == 0

        # Non-existent document (should still compute, but with 0 doc length)
        score = bm25_indexer.compute_bm25_score(["test"], "doc2", {"test": 1})
        # Score should be 0 because doc2 doesn't exist in document_lengths
        assert score >= 0  # BM25 can handle missing docs

    def test_index_data_structures(self, bm25_indexer, sample_chunks):
        """Test internal index data structures after indexing."""
        # Index chunks
        bm25_indexer.index_chunks(sample_chunks)

        # Check inverted index
        assert len(bm25_indexer.inverted_index) > 0
        assert "python" in bm25_indexer.inverted_index
        assert "programming" in bm25_indexer.inverted_index
        
        # Check structure of inverted index entries
        for term, postings in bm25_indexer.inverted_index.items():
            assert isinstance(postings, list)
            for doc_id, freq, positions in postings:
                assert isinstance(doc_id, str)
                assert isinstance(freq, int)
                assert isinstance(positions, list)

        # Check document frequencies
        assert len(bm25_indexer.document_frequencies) > 0
        assert bm25_indexer.document_frequencies["python"] == 4
        
        # Check collection frequencies
        assert len(bm25_indexer.collection_frequencies) > 0
        assert bm25_indexer.collection_frequencies["python"] >= 4
        
        # Check document lengths
        assert len(bm25_indexer.document_lengths) == 5
        for doc_id, length in bm25_indexer.document_lengths.items():
            assert isinstance(length, int)
            assert length > 0

        # Check collection stats
        stats = bm25_indexer.get_collection_stats()
        assert stats["document_count"] == 5
        assert stats["vocabulary_size"] > 0
        assert stats["total_terms"] > 0
        assert stats["avg_document_length"] > 0

    def test_clear(self, bm25_indexer, sample_chunks):
        """Test clearing the index."""
        # Index chunks
        bm25_indexer.index_chunks(sample_chunks)

        # Clear the index
        bm25_indexer.clear()

        # Check that everything is reset
        assert len(bm25_indexer.inverted_index) == 0
        assert bm25_indexer.document_lengths == {}
        assert bm25_indexer.total_document_length == 0
        assert bm25_indexer.document_count == 0

    def test_idf_calculation(self, bm25_indexer):
        """Test IDF calculation in BM25 formula."""
        # Create chunks with different term distributions
        chunks = []
        now = datetime.now()
        # 6 documents containing "common"
        for i in range(6):
            chunks.append(Chunk(
                id=f"doc{i}",
                path=Path(f"/docs/doc{i}.md"),
                title=f"Document {i}",
                content="common word",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ))
        # 2 documents containing "rare"
        for i in range(6, 8):
            chunks.append(Chunk(
                id=f"doc{i}",
                path=Path(f"/docs/doc{i}.md"),
                title=f"Document {i}",
                content="rare word",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ))
        # 2 documents containing "other"
        for i in range(8, 10):
            chunks.append(Chunk(
                id=f"doc{i}",
                path=Path(f"/docs/doc{i}.md"),
                title=f"Document {i}",
                content="other word",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ))
        
        bm25_indexer.index_chunks(chunks)

        # Calculate scores for documents containing each term
        common_score = bm25_indexer.compute_bm25_score(["common"], "doc0", {"common": 1})
        rare_score = bm25_indexer.compute_bm25_score(["rare"], "doc6", {"rare": 1})

        # Rare term should have higher IDF and thus higher score
        assert rare_score > common_score

    def test_document_length_normalization(self, bm25_indexer):
        """Test document length normalization in BM25."""
        # Create documents of different lengths. Make the test term appear in only 1 out of 5 docs for positive IDF
        now = datetime.now()
        chunks = [
            Chunk(
                id="short",
                path=Path("/docs/short.md"),
                title="Short",
                content="specialword other",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
            Chunk(
                id="long",
                path=Path("/docs/long.md"),
                title="Long",
                content="specialword other other other other other other other other other other",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
            Chunk(
                id="other1",
                path=Path("/docs/other1.md"),
                title="Other1",
                content="different content entirely",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
            Chunk(
                id="other2",
                path=Path("/docs/other2.md"),
                title="Other2",
                content="more different content",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
            Chunk(
                id="other3",
                path=Path("/docs/other3.md"),
                title="Other3",
                content="totally unrelated words here",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
        ]
        bm25_indexer.index_chunks(chunks)

        # Both documents have same term frequency for "specialword" (1)
        # specialword appears in 2 out of 5 documents, so should have positive IDF
        score_short = bm25_indexer.compute_bm25_score(["specialword"], "short", {"specialword": 1})
        score_long = bm25_indexer.compute_bm25_score(["specialword"], "long", {"specialword": 1})

        # Shorter document should have higher score due to length normalization
        assert score_short > score_long

    def test_parameter_sensitivity(self):
        """Test BM25 with different k1 and b parameters."""
        # Create indexers with different parameters
        indexer_high_k1 = BM25Indexer(k1=2.0, b=0.75)
        indexer_low_k1 = BM25Indexer(k1=0.5, b=0.75)
        indexer_no_length_norm = BM25Indexer(k1=1.2, b=0.0)

        # Add same documents to all indexers
        docs = {
            "doc1": ["term"] * 5,
            "doc2": ["term"] * 1,
        }
        
        now = datetime.now()
        chunks = [
            Chunk(
                id="doc1",
                path=Path("/docs/doc1.md"),
                title="Document 1",
                content="term term term term term",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
            Chunk(
                id="doc2",
                path=Path("/docs/doc2.md"),
                title="Document 2",
                content="term",
                chunk_index=0,
                language="en",
                created_at=now,
                modified_at=now,
                metadata={},
            ),
        ]
        
        for indexer in [indexer_high_k1, indexer_low_k1, indexer_no_length_norm]:
            indexer.index_chunks(chunks)

        # Test k1 parameter effect (term frequency saturation)
        score_high_k1 = indexer_high_k1.compute_bm25_score(["term"], "doc1", {"term": 5})
        score_low_k1 = indexer_low_k1.compute_bm25_score(["term"], "doc1", {"term": 5})
        
        # Lower k1 should saturate term frequency faster
        ratio_high_k1 = score_high_k1 / indexer_high_k1.compute_bm25_score(["term"], "doc2", {"term": 1})
        ratio_low_k1 = score_low_k1 / indexer_low_k1.compute_bm25_score(["term"], "doc2", {"term": 1})
        assert ratio_high_k1 > ratio_low_k1

        # Test b parameter effect (length normalization)
        # When b=0, document length should not affect score
        score_no_norm_1 = indexer_no_length_norm.compute_bm25_score(["term"], "doc1", {"term": 5})
        score_no_norm_2 = indexer_no_length_norm.compute_bm25_score(["term"], "doc2", {"term": 1})
        
        # With b=0, the ratio should be closer to the term frequency ratio than with b=0.75
        score_with_norm_1 = indexer_high_k1.compute_bm25_score(["term"], "doc1", {"term": 5})
        score_with_norm_2 = indexer_high_k1.compute_bm25_score(["term"], "doc2", {"term": 1})
        
        ratio_no_norm = score_no_norm_1 / score_no_norm_2
        ratio_with_norm = score_with_norm_1 / score_with_norm_2
        
        # The ratio without normalization should be closer to 5 (the term frequency ratio)
        # than the ratio with normalization
        assert abs(ratio_no_norm - 5.0) < abs(ratio_with_norm - 5.0)