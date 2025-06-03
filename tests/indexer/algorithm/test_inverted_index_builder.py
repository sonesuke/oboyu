"""Tests for the inverted index builder module."""

import pytest
from datetime import datetime
from pathlib import Path

from oboyu.indexer.algorithm.inverted_index_builder import InvertedIndexBuilder
from oboyu.common.types import Chunk


class TestInvertedIndexBuilder:
    """Test cases for InvertedIndexBuilder class."""

    @pytest.fixture
    def index_builder(self):
        """Create an InvertedIndexBuilder instance for testing."""
        return InvertedIndexBuilder(store_positions=True)

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
        ]

    def test_initialization(self):
        """Test InvertedIndexBuilder initialization."""
        builder = InvertedIndexBuilder(store_positions=False)
        assert not builder.store_positions
        assert builder.get_vocabulary_size() == 0
        assert len(builder.get_all_terms()) == 0

    def test_update_index(self, index_builder):
        """Test updating index with term frequencies."""
        term_frequencies = {"python": 2, "programming": 1, "language": 1}
        unique_terms = index_builder.update_index("doc1", term_frequencies)
        
        assert unique_terms == {"python", "programming", "language"}
        assert index_builder.get_vocabulary_size() == 3
        
        # Check postings
        postings = index_builder.get_term_postings("python")
        assert len(postings) == 1
        assert postings[0][0] == "doc1"  # document ID
        assert postings[0][1] == 2       # frequency
        assert postings[0][2] == []      # positions (empty when store_positions=True)

    def test_build_index(self, index_builder, sample_chunks):
        """Test building index from chunks."""
        # Simulate term frequencies for chunks
        term_frequencies = {
            "doc1": {"python": 2, "programming": 1, "language": 1},
            "doc2": {"java": 1, "programming": 1, "language": 1},
        }
        
        stats = index_builder.build_index(sample_chunks, term_frequencies)
        
        assert stats["chunks_indexed"] == 2
        assert stats["terms_indexed"] == 7  # Total term occurrences: 4 + 3
        assert index_builder.get_vocabulary_size() == 4  # python, programming, language, java

    def test_remove_from_index(self, index_builder):
        """Test removing a chunk from the index."""
        # Add some data
        index_builder.update_index("doc1", {"python": 2, "programming": 1})
        index_builder.update_index("doc2", {"python": 1, "java": 1})
        
        assert index_builder.get_vocabulary_size() == 3
        
        # Remove doc1
        index_builder.remove_from_index("doc1")
        
        # Check that only doc2 postings remain
        python_postings = index_builder.get_term_postings("python")
        assert len(python_postings) == 1
        assert python_postings[0][0] == "doc2"
        
        # Check that programming term is completely removed
        programming_postings = index_builder.get_term_postings("programming")
        assert len(programming_postings) == 0
        assert index_builder.get_vocabulary_size() == 2  # python, java

    def test_get_term_postings(self, index_builder):
        """Test getting postings for specific terms."""
        index_builder.update_index("doc1", {"python": 2})
        index_builder.update_index("doc2", {"python": 1})
        
        postings = index_builder.get_term_postings("python")
        assert len(postings) == 2
        
        # Check document IDs are present
        doc_ids = [posting[0] for posting in postings]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        
        # Non-existent term
        empty_postings = index_builder.get_term_postings("nonexistent")
        assert len(empty_postings) == 0

    def test_get_all_terms(self, index_builder):
        """Test getting all terms in vocabulary."""
        index_builder.update_index("doc1", {"python": 2, "programming": 1})
        index_builder.update_index("doc2", {"java": 1})
        
        all_terms = index_builder.get_all_terms()
        assert all_terms == {"python", "programming", "java"}

    def test_clear(self, index_builder):
        """Test clearing the index."""
        index_builder.update_index("doc1", {"python": 2, "programming": 1})
        assert index_builder.get_vocabulary_size() == 2
        
        index_builder.clear()
        assert index_builder.get_vocabulary_size() == 0
        assert len(index_builder.get_all_terms()) == 0

    def test_get_index_data(self, index_builder):
        """Test getting raw index data."""
        index_builder.update_index("doc1", {"python": 2})
        
        index_data = index_builder.get_index_data()
        assert "python" in index_data
        assert len(index_data["python"]) == 1
        assert index_data["python"][0][0] == "doc1"

    def test_store_positions_false(self):
        """Test builder with store_positions=False."""
        builder = InvertedIndexBuilder(store_positions=False)
        builder.update_index("doc1", {"python": 1})
        
        postings = builder.get_term_postings("python")
        assert postings[0][2] is None  # positions should be None