"""Tests for the term frequency analyzer module."""

import pytest
from unittest.mock import Mock, MagicMock

from oboyu.indexer.algorithm.term_frequency_analyzer import TermFrequencyAnalyzer


class TestTermFrequencyAnalyzer:
    """Test cases for TermFrequencyAnalyzer class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.get_term_frequencies.return_value = {"python": 2, "programming": 1, "language": 1}
        tokenizer.tokenize.return_value = ["python", "is", "a", "programming", "language", "python"]
        return tokenizer

    @pytest.fixture
    def analyzer(self, mock_tokenizer):
        """Create a TermFrequencyAnalyzer instance for testing."""
        return TermFrequencyAnalyzer(mock_tokenizer)

    def test_initialization(self, mock_tokenizer):
        """Test TermFrequencyAnalyzer initialization."""
        analyzer = TermFrequencyAnalyzer(mock_tokenizer)
        assert analyzer.tokenizer == mock_tokenizer

    def test_analyze_document(self, analyzer, mock_tokenizer):
        """Test analyzing document content."""
        content = "Python is a programming language"
        
        result = analyzer.analyze_document(content)
        
        mock_tokenizer.get_term_frequencies.assert_called_once_with(content)
        assert result == {"python": 2, "programming": 1, "language": 1}

    def test_get_document_length(self, analyzer, mock_tokenizer):
        """Test getting document length in terms."""
        content = "Python programming language"
        
        length = analyzer.get_document_length(content)
        
        # Sum of term frequencies: 2 + 1 + 1 = 4
        assert length == 4
        mock_tokenizer.get_term_frequencies.assert_called_once_with(content)

    def test_extract_unique_terms(self, analyzer, mock_tokenizer):
        """Test extracting unique terms from document."""
        content = "Python programming language"
        
        unique_terms = analyzer.extract_unique_terms(content)
        
        assert unique_terms == {"python", "programming", "language"}
        mock_tokenizer.get_term_frequencies.assert_called_once_with(content)

    def test_get_term_positions(self, analyzer, mock_tokenizer):
        """Test getting term positions in document."""
        content = "Python is a programming language Python"
        
        positions = analyzer.get_term_positions(content, "python")
        
        # "python" appears at positions 0 and 5 (case-insensitive)
        assert positions == [0, 5]
        mock_tokenizer.tokenize.assert_called_once_with(content)

    def test_get_term_positions_not_found(self, analyzer, mock_tokenizer):
        """Test getting positions for term not in document."""
        content = "Java programming language"
        mock_tokenizer.tokenize.return_value = ["java", "programming", "language"]
        
        positions = analyzer.get_term_positions(content, "python")
        
        assert positions == []

    def test_analyze_batch(self, analyzer, mock_tokenizer):
        """Test analyzing multiple documents."""
        contents = ["Python programming", "Java development"]
        
        # Set up different return values for each call
        mock_tokenizer.get_term_frequencies.side_effect = [
            {"python": 1, "programming": 1},
            {"java": 1, "development": 1}
        ]
        
        results = analyzer.analyze_batch(contents)
        
        assert len(results) == 2
        assert results[0] == {"python": 1, "programming": 1}
        assert results[1] == {"java": 1, "development": 1}
        assert mock_tokenizer.get_term_frequencies.call_count == 2

    def test_get_vocabulary_from_documents(self, analyzer, mock_tokenizer):
        """Test extracting vocabulary from multiple documents."""
        contents = ["Python programming", "Java development", "Python scripting"]
        
        # Set up different return values for each call
        mock_tokenizer.get_term_frequencies.side_effect = [
            {"python": 1, "programming": 1},
            {"java": 1, "development": 1},
            {"python": 1, "scripting": 1}
        ]
        
        vocabulary = analyzer.get_vocabulary_from_documents(contents)
        
        assert vocabulary == {"python", "programming", "java", "development", "scripting"}
        assert mock_tokenizer.get_term_frequencies.call_count == 3

    def test_empty_document(self, mock_tokenizer):
        """Test analyzing empty document."""
        analyzer = TermFrequencyAnalyzer(mock_tokenizer)
        mock_tokenizer.get_term_frequencies.return_value = {}
        mock_tokenizer.tokenize.return_value = []
        
        result = analyzer.analyze_document("")
        assert result == {}
        
        length = analyzer.get_document_length("")
        assert length == 0
        
        unique_terms = analyzer.extract_unique_terms("")
        assert unique_terms == set()

    def test_single_term_document(self, mock_tokenizer):
        """Test analyzing document with single term."""
        analyzer = TermFrequencyAnalyzer(mock_tokenizer)
        mock_tokenizer.get_term_frequencies.return_value = {"python": 1}
        mock_tokenizer.tokenize.return_value = ["python"]
        
        result = analyzer.analyze_document("Python")
        assert result == {"python": 1}
        
        length = analyzer.get_document_length("Python")
        assert length == 1
        
        unique_terms = analyzer.extract_unique_terms("Python")
        assert unique_terms == {"python"}

    def test_repeated_terms(self, mock_tokenizer):
        """Test analyzing document with repeated terms."""
        analyzer = TermFrequencyAnalyzer(mock_tokenizer)
        mock_tokenizer.get_term_frequencies.return_value = {"python": 3}
        mock_tokenizer.tokenize.return_value = ["python", "python", "python"]
        
        result = analyzer.analyze_document("Python Python Python")
        assert result == {"python": 3}
        
        length = analyzer.get_document_length("Python Python Python")
        assert length == 3
        
        positions = analyzer.get_term_positions("Python Python Python", "python")
        assert positions == [0, 1, 2]