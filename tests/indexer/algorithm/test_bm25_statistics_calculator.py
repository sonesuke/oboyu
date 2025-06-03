"""Tests for the BM25 statistics calculator module."""

import pytest
import math

from oboyu.indexer.algorithm.bm25_statistics_calculator import BM25StatisticsCalculator


class TestBM25StatisticsCalculator:
    """Test cases for BM25StatisticsCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a BM25StatisticsCalculator instance for testing."""
        return BM25StatisticsCalculator()

    def test_initialization(self):
        """Test BM25StatisticsCalculator initialization."""
        calc = BM25StatisticsCalculator()
        assert calc.document_count == 0
        assert calc.total_document_length == 0
        assert len(calc.document_frequencies) == 0
        assert len(calc.collection_frequencies) == 0

    def test_calculate_idf_scores(self, calculator):
        """Test IDF score calculation."""
        term_doc_frequencies = {"python": 2, "java": 1, "programming": 3}
        total_docs = 5
        
        idf_scores = calculator.calculate_idf_scores(term_doc_frequencies, total_docs)
        
        # IDF = log((N - df + 0.5) / (df + 0.5))
        expected_python_idf = math.log((5 - 2 + 0.5) / (2 + 0.5))  # log(3.5/2.5)
        expected_java_idf = math.log((5 - 1 + 0.5) / (1 + 0.5))    # log(4.5/1.5)
        expected_programming_idf = math.log((5 - 3 + 0.5) / (3 + 0.5))  # log(2.5/3.5)
        
        assert abs(idf_scores["python"] - expected_python_idf) < 1e-10
        assert abs(idf_scores["java"] - expected_java_idf) < 1e-10
        assert abs(idf_scores["programming"] - expected_programming_idf) < 1e-10

    def test_calculate_average_document_length(self, calculator):
        """Test average document length calculation."""
        lengths = [10, 20, 30, 40]
        avg_length = calculator.calculate_average_document_length(lengths)
        assert avg_length == 25.0
        
        # Empty list
        avg_empty = calculator.calculate_average_document_length([])
        assert avg_empty == 0.0

    def test_update_collection_statistics(self, calculator):
        """Test updating collection statistics."""
        term_frequencies = {"python": 2, "programming": 1, "language": 1}
        unique_terms = {"python", "programming", "language"}
        
        calculator.update_collection_statistics("doc1", term_frequencies, unique_terms)
        
        assert calculator.document_count == 1
        assert calculator.total_document_length == 4  # 2 + 1 + 1
        assert calculator.get_document_length("doc1") == 4
        assert calculator.get_document_frequency("python") == 1
        assert calculator.get_collection_frequency("python") == 2

    def test_get_document_frequency(self, calculator):
        """Test getting document frequency for terms."""
        calculator.update_collection_statistics(
            "doc1", {"python": 2}, {"python"}
        )
        calculator.update_collection_statistics(
            "doc2", {"python": 1, "java": 1}, {"python", "java"}
        )
        
        assert calculator.get_document_frequency("python") == 2
        assert calculator.get_document_frequency("java") == 1
        assert calculator.get_document_frequency("nonexistent") == 0

    def test_get_collection_frequency(self, calculator):
        """Test getting collection frequency for terms."""
        calculator.update_collection_statistics(
            "doc1", {"python": 2}, {"python"}
        )
        calculator.update_collection_statistics(
            "doc2", {"python": 1}, {"python"}
        )
        
        assert calculator.get_collection_frequency("python") == 3  # 2 + 1
        assert calculator.get_collection_frequency("nonexistent") == 0

    def test_get_average_document_length(self, calculator):
        """Test getting average document length."""
        calculator.update_collection_statistics("doc1", {"a": 1, "b": 1}, {"a", "b"})
        calculator.update_collection_statistics("doc2", {"c": 1, "d": 1, "e": 1}, {"c", "d", "e"})
        
        assert calculator.get_average_document_length() == 2.5  # (2 + 3) / 2

    def test_get_collection_stats(self, calculator):
        """Test getting comprehensive collection statistics."""
        calculator.update_collection_statistics("doc1", {"python": 2, "code": 1}, {"python", "code"})
        calculator.update_collection_statistics("doc2", {"python": 1, "java": 1}, {"python", "java"})
        
        stats = calculator.get_collection_stats()
        
        assert stats["document_count"] == 2
        assert stats["total_document_length"] == 5  # 3 + 2
        assert stats["average_document_length"] == 2.5
        assert stats["avg_document_length"] == 2.5  # Backward compatibility
        assert stats["vocabulary_size"] == 3  # python, code, java
        assert stats["total_terms"] == 5  # Total term occurrences

    def test_calculate_bm25_term_score(self, calculator):
        """Test BM25 term score calculation."""
        # Set up collection with known statistics
        calculator.update_collection_statistics("doc1", {"python": 2, "code": 1}, {"python", "code"})
        calculator.update_collection_statistics("doc2", {"java": 1, "code": 1}, {"java", "code"})
        calculator.update_collection_statistics("doc3", {"python": 1}, {"python"})
        
        # Calculate score for "python" in doc1
        # python appears in 2/3 documents, doc1 has length 3
        score = calculator.calculate_bm25_term_score(
            term="python",
            term_frequency=2,
            document_length=3,
            k1=1.2,
            b=0.75
        )
        
        assert score != 0  # Score can be negative for common terms (correct BM25 behavior)
        
        # Test with non-existent term
        zero_score = calculator.calculate_bm25_term_score(
            term="nonexistent",
            term_frequency=1,
            document_length=5
        )
        assert zero_score == 0.0

    def test_remove_document_statistics(self, calculator):
        """Test removing document statistics."""
        # Add documents
        calculator.update_collection_statistics("doc1", {"python": 2, "code": 1}, {"python", "code"})
        calculator.update_collection_statistics("doc2", {"python": 1, "java": 1}, {"python", "java"})
        
        assert calculator.document_count == 2
        assert calculator.get_document_frequency("python") == 2
        
        # Remove doc1
        calculator.remove_document_statistics("doc1", {"python": 2, "code": 1}, {"python", "code"})
        
        assert calculator.document_count == 1
        assert calculator.get_document_frequency("python") == 1
        assert calculator.get_document_frequency("code") == 0  # Should be removed
        assert calculator.get_collection_frequency("python") == 1  # Only from doc2
        assert "doc1" not in calculator.document_lengths

    def test_clear(self, calculator):
        """Test clearing all statistics."""
        calculator.update_collection_statistics("doc1", {"python": 2}, {"python"})
        assert calculator.document_count == 1
        
        calculator.clear()
        assert calculator.document_count == 0
        assert calculator.total_document_length == 0
        assert len(calculator.document_frequencies) == 0
        assert len(calculator.collection_frequencies) == 0
        assert len(calculator.document_lengths) == 0

    def test_idf_edge_cases(self, calculator):
        """Test IDF calculation edge cases."""
        # Zero document frequency should result in 0 IDF
        idf_scores = calculator.calculate_idf_scores({"term": 0}, 5)
        assert idf_scores["term"] == 0.0
        
        # Document frequency equal to total docs should give negative IDF
        idf_scores = calculator.calculate_idf_scores({"common": 5}, 5)
        expected_idf = math.log((5 - 5 + 0.5) / (5 + 0.5))  # log(0.5/5.5) < 0
        assert idf_scores["common"] < 0
        assert abs(idf_scores["common"] - expected_idf) < 1e-10