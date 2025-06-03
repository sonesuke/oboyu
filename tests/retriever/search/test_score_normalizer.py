"""Tests for ScoreNormalizer."""

import pytest
import numpy as np
from unittest.mock import patch

from oboyu.retriever.search.score_normalizer import (
    NormalizationMethod,
    ScoreNormalizer,
)
from oboyu.common.types import SearchResult


@pytest.fixture
def sample_results():
    """Sample results with varying scores."""
    return [
        SearchResult(
            chunk_id="1",
            path="/test1.txt",
            title="Test 1",
            content="Content 1",
            chunk_index=0,
            language="en",
            score=0.9,
        ),
        SearchResult(
            chunk_id="2",
            path="/test2.txt",
            title="Test 2",
            content="Content 2",
            chunk_index=0,
            language="en",
            score=0.5,
        ),
        SearchResult(
            chunk_id="3",
            path="/test3.txt",
            title="Test 3",
            content="Content 3",
            chunk_index=0,
            language="en",
            score=0.7,
        ),
    ]


def test_min_max_normalization(sample_results):
    """Test min-max normalization."""
    normalizer = ScoreNormalizer(NormalizationMethod.MIN_MAX)
    normalized = normalizer.normalize_scores(sample_results, "test")
    
    # Check that scores are normalized to [0, 1] range
    scores = [r.score for r in normalized]
    assert min(scores) == 0.0  # Min score becomes 0
    assert max(scores) == 1.0  # Max score becomes 1
    
    # Check that we have the correct number of results
    assert len(normalized) == 3
    
    # Find results by chunk_id to check normalization
    result_1 = next(r for r in normalized if r.chunk_id == "1")
    result_2 = next(r for r in normalized if r.chunk_id == "2")
    result_3 = next(r for r in normalized if r.chunk_id == "3")
    
    # Check relative ordering of normalized scores
    assert result_1.score == 1.0  # Highest original score (0.9) -> 1.0
    assert result_2.score == 0.0  # Lowest original score (0.5) -> 0.0
    assert abs(result_3.score - 0.5) < 1e-10  # Middle original score (0.7) -> ~0.5


def test_min_max_normalization_identical_scores():
    """Test min-max normalization with identical scores."""
    results = [
        SearchResult(
            chunk_id="1",
            path="/test1.txt",
            title="Test 1",
            content="Content 1",
            chunk_index=0,
            language="en",
            score=0.5,
        ),
        SearchResult(
            chunk_id="2",
            path="/test2.txt",
            title="Test 2",
            content="Content 2",
            chunk_index=0,
            language="en",
            score=0.5,
        ),
    ]
    
    normalizer = ScoreNormalizer(NormalizationMethod.MIN_MAX)
    normalized = normalizer.normalize_scores(results, "test")
    
    # Should return original results when all scores are identical
    assert len(normalized) == 2
    assert normalized[0].score == 0.5
    assert normalized[1].score == 0.5


def test_z_score_normalization(sample_results):
    """Test z-score normalization."""
    normalizer = ScoreNormalizer(NormalizationMethod.Z_SCORE)
    normalized = normalizer.normalize_scores(sample_results, "test")
    
    # Check that we get valid scores
    scores = [r.score for r in normalized]
    assert all(0 <= score <= 1 for score in scores)
    
    # Check that original structure is preserved
    assert len(normalized) == 3
    assert all(hasattr(r, 'chunk_id') for r in normalized)


def test_z_score_normalization_zero_std():
    """Test z-score normalization with zero standard deviation."""
    results = [
        SearchResult(
            chunk_id="1",
            path="/test1.txt",
            title="Test 1",
            content="Content 1",
            chunk_index=0,
            language="en",
            score=0.5,
        ),
        SearchResult(
            chunk_id="2",
            path="/test2.txt",
            title="Test 2",
            content="Content 2",
            chunk_index=0,
            language="en",
            score=0.5,
        ),
    ]
    
    normalizer = ScoreNormalizer(NormalizationMethod.Z_SCORE)
    normalized = normalizer.normalize_scores(results, "test")
    
    # Should return original results when std is 0
    assert len(normalized) == 2
    assert normalized[0].score == 0.5
    assert normalized[1].score == 0.5


def test_rank_based_normalization(sample_results):
    """Test rank-based normalization."""
    normalizer = ScoreNormalizer(NormalizationMethod.RANK_BASED)
    normalized = normalizer.normalize_scores(sample_results, "test")
    
    # Check that scores are based on rank
    scores = [r.score for r in normalized]
    assert max(scores) == 1.0  # Best rank gets 1.0
    assert abs(min(scores) - 1/3) < 1e-6   # Worst rank gets 1/n (with floating point tolerance)
    
    # Check that results are sorted by original score
    assert normalized[0].chunk_id == "1"  # Highest original score
    assert normalized[1].chunk_id == "3"  # Middle original score
    assert normalized[2].chunk_id == "2"  # Lowest original score


def test_empty_results():
    """Test normalization with empty results."""
    normalizer = ScoreNormalizer()
    normalized = normalizer.normalize_scores([], "test")
    assert normalized == []


def test_unknown_normalization_method():
    """Test unknown normalization method fallback."""
    with patch("oboyu.retriever.search.score_normalizer.logger") as mock_logger:
        # Create normalizer with invalid method by bypassing enum validation
        normalizer = ScoreNormalizer()
        normalizer.method = "invalid_method"
        
        sample_results = [
            SearchResult(
                chunk_id="1",
                path="/test.txt",
                title="Test",
                content="Content",
                chunk_index=0,
                language="en",
                score=0.5,
            )
        ]
        
        # Should fallback to min-max
        normalized = normalizer.normalize_scores(sample_results, "test")
        mock_logger.warning.assert_called_once()
        assert len(normalized) == 1


@patch("oboyu.retriever.search.score_normalizer.logger")
def test_normalization_exception_handling(mock_logger, sample_results):
    """Test error handling in normalization."""
    normalizer = ScoreNormalizer()
    
    # Mock a method that raises an exception
    with patch.object(normalizer, '_min_max_normalize', side_effect=Exception("Test error")):
        result = normalizer.normalize_scores(sample_results, "test")
        
        # Should return original results on error
        assert result == sample_results
        mock_logger.error.assert_called_once()