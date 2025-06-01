"""Tests for HybridSearchCombiner."""

import pytest
from unittest.mock import Mock

from oboyu.indexer.core.hybrid_search_combiner import HybridSearchCombiner
from oboyu.indexer.core.score_normalizer import ScoreNormalizer
from oboyu.indexer.search.search_result import SearchResult


@pytest.fixture
def vector_results():
    """Sample vector search results."""
    return [
        SearchResult(
            chunk_id="1",
            path="/test1.txt",
            title="Test 1",
            content="Vector content 1",
            chunk_index=0,
            language="en",
            score=0.9,
        ),
        SearchResult(
            chunk_id="2",
            path="/test2.txt",
            title="Test 2",
            content="Vector content 2",
            chunk_index=0,
            language="en",
            score=0.7,
        ),
    ]


@pytest.fixture
def bm25_results():
    """Sample BM25 search results."""
    return [
        SearchResult(
            chunk_id="1",  # Overlapping with vector results
            path="/test1.txt",
            title="Test 1",
            content="BM25 content 1",
            chunk_index=0,
            language="en",
            score=0.8,
        ),
        SearchResult(
            chunk_id="3",  # Unique to BM25
            path="/test3.txt",
            title="Test 3",
            content="BM25 content 3",
            chunk_index=0,
            language="en",
            score=0.6,
        ),
    ]


def test_combiner_default_weights():
    """Test combiner with default weights."""
    combiner = HybridSearchCombiner()
    assert combiner.vector_weight == 0.7
    assert combiner.bm25_weight == 0.3


def test_combiner_custom_weights():
    """Test combiner with custom weights."""
    combiner = HybridSearchCombiner(vector_weight=0.6, bm25_weight=0.4)
    assert combiner.vector_weight == 0.6
    assert combiner.bm25_weight == 0.4


def test_combiner_weight_normalization():
    """Test automatic weight normalization."""
    combiner = HybridSearchCombiner(vector_weight=0.8, bm25_weight=0.4)
    # Should normalize to sum to 1.0
    assert abs(combiner.vector_weight + combiner.bm25_weight - 1.0) < 1e-10


def test_combiner_zero_weights():
    """Test combiner with zero weights fallback."""
    combiner = HybridSearchCombiner(vector_weight=0.0, bm25_weight=0.0)
    # Should use default values
    assert combiner.vector_weight == 0.7
    assert combiner.bm25_weight == 0.3


def test_combine_overlapping_results(vector_results, bm25_results):
    """Test combining results with overlapping chunks."""
    combiner = HybridSearchCombiner(vector_weight=0.7, bm25_weight=0.3)
    combined = combiner.combine(vector_results, bm25_results, limit=10)
    
    # Should have 3 unique chunks
    assert len(combined) == 3
    
    # Check that overlapping chunk has combined score
    chunk_1 = next(r for r in combined if r.chunk_id == "1")
    expected_score = 0.9 * 0.7 + 0.8 * 0.3  # Combined weighted score
    assert abs(chunk_1.score - expected_score) < 1e-10
    
    # Check that results are sorted by combined score
    assert combined[0].score >= combined[1].score >= combined[2].score


def test_combine_with_limit(vector_results, bm25_results):
    """Test combining with result limit."""
    combiner = HybridSearchCombiner()
    combined = combiner.combine(vector_results, bm25_results, limit=2)
    
    # Should respect the limit
    assert len(combined) == 2
    
    # Should return top 2 by combined score
    assert combined[0].score >= combined[1].score


def test_combine_empty_vector_results(bm25_results):
    """Test combining with empty vector results."""
    combiner = HybridSearchCombiner()
    combined = combiner.combine([], bm25_results, limit=10)
    
    # Should return BM25 results with weighted scores
    assert len(combined) == 2
    assert all(r.score == orig.score * combiner.bm25_weight for r, orig in zip(combined, bm25_results))


def test_combine_empty_bm25_results(vector_results):
    """Test combining with empty BM25 results."""
    combiner = HybridSearchCombiner()
    combined = combiner.combine(vector_results, [], limit=10)
    
    # Should return vector results with weighted scores
    assert len(combined) == 2
    assert all(r.score == orig.score * combiner.vector_weight for r, orig in zip(combined, vector_results))


def test_combine_both_empty():
    """Test combining with both result lists empty."""
    combiner = HybridSearchCombiner()
    combined = combiner.combine([], [], limit=10)
    
    assert len(combined) == 0


def test_combine_with_score_normalizer(vector_results, bm25_results):
    """Test combining with score normalizer."""
    mock_normalizer = Mock(spec=ScoreNormalizer)
    mock_normalizer.normalize_scores.side_effect = lambda results, method: results
    
    combiner = HybridSearchCombiner(score_normalizer=mock_normalizer)
    combined = combiner.combine(vector_results, bm25_results, limit=10)
    
    # Should call normalizer for both result sets
    assert mock_normalizer.normalize_scores.call_count == 2
    mock_normalizer.normalize_scores.assert_any_call(vector_results, "vector")
    mock_normalizer.normalize_scores.assert_any_call(bm25_results, "bm25")
    
    assert len(combined) == 3


def test_combine_exception_handling(vector_results, bm25_results):
    """Test error handling in combination."""
    # Create a combiner that will cause an exception
    combiner = HybridSearchCombiner()
    
    # Pass invalid results that should cause an error but be handled gracefully
    result = combiner.combine([None], bm25_results, limit=10)
    # Should fallback to vector_results (which is [None][:10] = [None])
    assert len(result) == 1
    assert result[0] is None


def test_combine_fallback_on_exception(vector_results):
    """Test fallback behavior on exception."""
    combiner = HybridSearchCombiner()
    
    # This should trigger the exception handling and fallback to vector_results
    result = combiner.combine(vector_results, [None], limit=10)
    assert len(result) == 2
    assert result[0].chunk_id == "1"