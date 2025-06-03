"""Tests for ResultMerger."""

import pytest

from oboyu.retriever.search.result_merger import ResultMerger
from oboyu.common.types import SearchResult


@pytest.fixture
def merger():
    """ResultMerger instance."""
    return ResultMerger()


@pytest.fixture
def sample_results_1():
    """First set of sample results."""
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
            score=0.7,
        ),
    ]


@pytest.fixture
def sample_results_2():
    """Second set of sample results."""
    return [
        SearchResult(
            chunk_id="1",  # Duplicate with higher score
            path="/test1.txt",
            title="Test 1 Updated",
            content="Updated content 1",
            chunk_index=0,
            language="en",
            score=0.95,  # Higher score
        ),
        SearchResult(
            chunk_id="3",
            path="/test3.txt",
            title="Test 3",
            content="Content 3",
            chunk_index=0,
            language="en",
            score=0.8,
        ),
    ]


def test_merge_empty_lists(merger):
    """Test merging empty result lists."""
    result = merger.merge([], [], limit=10)
    assert result == []


def test_merge_single_list(merger, sample_results_1):
    """Test merging with single result list."""
    result = merger.merge(sample_results_1, limit=10)
    assert len(result) == 2
    assert result[0].score == 0.9  # Sorted by score
    assert result[1].score == 0.7


def test_merge_multiple_lists(merger, sample_results_1, sample_results_2):
    """Test merging multiple result lists."""
    result = merger.merge(sample_results_1, sample_results_2, limit=10)
    
    # Should have 3 unique chunks (1, 2, 3)
    assert len(result) == 3
    
    # Should be sorted by score
    assert result[0].score == 0.95  # chunk_id="1" with higher score
    assert result[1].score == 0.8   # chunk_id="3"
    assert result[2].score == 0.7   # chunk_id="2"
    
    # Check that we kept the better version of chunk_id="1"
    chunk_1_result = next(r for r in result if r.chunk_id == "1")
    assert chunk_1_result.title == "Test 1 Updated"


def test_merge_with_limit(merger, sample_results_1, sample_results_2):
    """Test merging with result limit."""
    result = merger.merge(sample_results_1, sample_results_2, limit=2)
    
    assert len(result) == 2
    assert result[0].score == 0.95  # Top 2 scores
    assert result[1].score == 0.8


def test_merge_deduplication(merger):
    """Test proper deduplication behavior."""
    results_1 = [
        SearchResult(
            chunk_id="same",
            path="/test.txt",
            title="Lower Score",
            content="Content",
            chunk_index=0,
            language="en",
            score=0.5,
        )
    ]
    
    results_2 = [
        SearchResult(
            chunk_id="same",  # Same chunk_id
            path="/test.txt",
            title="Higher Score",
            content="Better content",
            chunk_index=0,
            language="en",
            score=0.8,  # Higher score
        )
    ]
    
    result = merger.merge(results_1, results_2, limit=10)
    
    assert len(result) == 1
    assert result[0].score == 0.8
    assert result[0].title == "Higher Score"


def test_merge_exception_handling(merger):
    """Test error handling in merger."""
    # Create malformed result that might cause issues
    malformed_results = [None]
    
    # The merger should handle the exception gracefully and return fallback
    result = merger.merge(malformed_results, limit=10)
    # Fallback returns the first non-empty list, which is [None][:10] = [None]
    assert result == [None]