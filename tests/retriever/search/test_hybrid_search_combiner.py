"""Tests for HybridSearchCombiner."""

import pytest
from unittest.mock import Mock

from oboyu.retriever.search.hybrid_search_combiner import HybridSearchCombiner
from oboyu.retriever.search.score_normalizer import ScoreNormalizer
from oboyu.common.types import SearchResult


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


def test_combiner_default_rrf_k():
    """Test combiner default RRF k parameter."""
    combiner = HybridSearchCombiner()
    assert combiner.rrf_k == 60  # Default value


def test_combiner_custom_rrf_k():
    """Test combiner with custom RRF k parameter."""
    combiner = HybridSearchCombiner(rrf_k=100)
    assert combiner.rrf_k == 100


def test_combiner_deprecated_weights_warning(caplog):
    """Test that deprecated weight parameters issue a warning."""
    import logging
    caplog.set_level(logging.WARNING)
    
    combiner = HybridSearchCombiner(vector_weight=0.8, bm25_weight=0.2)
    assert combiner.rrf_k == 60  # Should still use default RRF
    assert "vector_weight and bm25_weight parameters are deprecated" in caplog.text


def test_combiner_rrf_k_validation():
    """Test RRF k parameter validation."""
    # Negative k should raise error
    with pytest.raises(ValueError, match="rrf_k must be positive"):
        HybridSearchCombiner(rrf_k=-1)
    
    # Zero k should raise error
    with pytest.raises(ValueError, match="rrf_k must be positive"):
        HybridSearchCombiner(rrf_k=0)


def test_combine_overlapping_results(vector_results, bm25_results):
    """Test combining results with overlapping documents using RRF."""
    combiner = HybridSearchCombiner(rrf_k=60)
    combined = combiner.combine(vector_results, bm25_results, limit=10)
    
    # Should have 3 unique documents
    assert len(combined) == 3
    
    # Check that chunk_id "1" has the highest score (appears in both)
    # It has rank 1 in both vector and BM25, so RRF score = 1/(60+1) + 1/(60+1)
    assert combined[0].chunk_id == "1"
    expected_score = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert abs(combined[0].score - expected_score) < 1e-5
    
    # Check other scores are lower than the first (overlapping) result
    assert combined[1].score < combined[0].score
    assert combined[2].score < combined[0].score
    
    # chunk_id "2" and "3" should have the same score (both rank 2 in their respective searches)
    expected_single_score = 1.0 / (60 + 2)
    assert abs(combined[1].score - expected_single_score) < 1e-5
    assert abs(combined[2].score - expected_single_score) < 1e-5


def test_combine_with_limit(vector_results, bm25_results):
    """Test limiting the number of combined results."""
    combiner = HybridSearchCombiner()
    combined = combiner.combine(vector_results, bm25_results, limit=2)
    
    # Should return top 2 results
    assert len(combined) == 2
    # First should be the overlapping document
    assert combined[0].chunk_id == "1"


def test_combine_empty_vector_results(bm25_results):
    """Test combining with empty vector results using RRF."""
    combiner = HybridSearchCombiner()
    combined = combiner.combine([], bm25_results, limit=10)
    
    # Should return BM25 results with RRF scores (only BM25 component)
    assert len(combined) == 2
    # First result should have score 1/(k+1), second should have 1/(k+2)
    assert abs(combined[0].score - 1.0 / (60 + 1)) < 1e-5
    assert abs(combined[1].score - 1.0 / (60 + 2)) < 1e-5


def test_combine_empty_bm25_results(vector_results):
    """Test combining with empty BM25 results using RRF."""
    combiner = HybridSearchCombiner()
    combined = combiner.combine(vector_results, [], limit=10)
    
    # Should return vector results with RRF scores (only vector component)
    assert len(combined) == 2
    # First result should have score 1/(k+1), second should have 1/(k+2)
    assert abs(combined[0].score - 1.0 / (60 + 1)) < 1e-5
    assert abs(combined[1].score - 1.0 / (60 + 2)) < 1e-5


def test_combine_both_empty():
    """Test combining with both empty results."""
    combiner = HybridSearchCombiner()
    combined = combiner.combine([], [], limit=10)
    
    # Should return empty list
    assert len(combined) == 0


def test_combine_with_score_normalizer(vector_results, bm25_results):
    """Test combining with score normalizer."""
    normalizer = Mock(spec=ScoreNormalizer)
    normalizer.normalize_scores.side_effect = lambda results, search_type: results  # No-op
    
    combiner = HybridSearchCombiner(score_normalizer=normalizer)
    combined = combiner.combine(vector_results, bm25_results, limit=10)
    
    # Score normalizer should be called twice (once for each result set)
    assert normalizer.normalize_scores.call_count == 2
    
    # Should still produce valid results
    assert len(combined) == 3


def test_combine_exception_handling(vector_results):
    """Test exception handling during combination."""
    # Mock score normalizer to raise exception
    normalizer = Mock(spec=ScoreNormalizer)
    normalizer.normalize_scores.side_effect = Exception("Normalization failed")
    
    combiner = HybridSearchCombiner(score_normalizer=normalizer)
    
    bm25_results = [
        SearchResult(
            chunk_id="3",
            path="/test3.txt",
            title="Test 3",
            content="BM25 content 3",
            chunk_index=0,
            language="en",
            score=0.6,
        )
    ]
    
    # Should handle gracefully and return fallback results
    combined = combiner.combine(vector_results, bm25_results, limit=10)
    
    # Should fallback to vector results when normalization fails
    assert len(combined) == len(vector_results)


def test_combine_fallback_on_exception():
    """Test fallback behavior when combination fails."""
    # Mock score normalizer to raise exception
    normalizer = Mock(spec=ScoreNormalizer)
    normalizer.normalize_scores.side_effect = Exception("Normalization failed")
    
    combiner = HybridSearchCombiner(score_normalizer=normalizer)
    
    # Should not raise exception but handle gracefully
    combined = combiner.combine([], [], limit=10)
    assert combined == []


def test_rrf_scoring_order():
    """Test that RRF scoring preserves expected order."""
    # Create results with clear ranking
    vector_results = [
        SearchResult(chunk_id=f"v{i}", path=f"/v{i}.txt", title=f"V{i}", 
                    content=f"Vector {i}", chunk_index=0, language="en", score=1.0-i*0.1)
        for i in range(5)
    ]
    
    bm25_results = [
        SearchResult(chunk_id=f"b{i}", path=f"/b{i}.txt", title=f"B{i}", 
                    content=f"BM25 {i}", chunk_index=0, language="en", score=1.0-i*0.1)
        for i in range(5)
    ]
    
    combiner = HybridSearchCombiner(rrf_k=60)
    combined = combiner.combine(vector_results, bm25_results, limit=10)
    
    # All results should be present
    assert len(combined) == 10
    
    # Check RRF scores are correctly calculated
    for result in combined:
        if result.chunk_id.startswith('v'):
            rank = int(result.chunk_id[1:]) + 1
            expected_score = 1.0 / (60 + rank)
        else:  # BM25 result
            rank = int(result.chunk_id[1:]) + 1
            expected_score = 1.0 / (60 + rank)
        assert abs(result.score - expected_score) < 1e-5