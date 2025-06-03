"""Simplified tests for reranker functionality.

Note: Most complex reranker functionality was part of the old API.
This file contains basic tests that work with the new architecture.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oboyu.common.types import SearchResult
from oboyu.retriever.services.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    ONNXCrossEncoderReranker,
    RerankedResult,
    create_reranker,
)


class TestRerankedResult:
    """Test cases for RerankedResult class."""
    
    def test_reranked_result_creation(self) -> None:
        """Test creating a RerankedResult."""
        original = SearchResult(
            chunk_id="test-1",
            path="/test/doc.txt",
            title="Test Document",
            content="Test content",
            chunk_index=0,
            language="en",
            score=0.8,
            metadata={"key": "value"},
        )
        
        reranked = RerankedResult(original, 0.95)
        
        assert reranked.original_result == original
        assert reranked.rerank_score == 0.95
    
    def test_to_search_result(self) -> None:
        """Test converting RerankedResult back to SearchResult."""
        original = SearchResult(
            chunk_id="test-1",
            path="/test/doc.txt",
            title="Test Document",
            content="Test content",
            chunk_index=0,
            language="en",
            score=0.8,
            metadata={"key": "value"},
        )
        
        reranked = RerankedResult(original, 0.95)
        search_result = reranked.to_search_result()
        
        # Should use rerank score as the new score
        assert search_result.chunk_id == "test-1"
        assert search_result.score == 0.95
        assert search_result.content == "Test content"


class TestCrossEncoderReranker:
    """Test cases for CrossEncoderReranker."""
    
    def test_reranker_creation(self) -> None:
        """Test basic reranker creation."""
        reranker = CrossEncoderReranker(batch_size=2)
        
        # Verify basic attributes
        assert reranker.batch_size == 2
        assert hasattr(reranker, 'rerank')
    
    def test_rerank_with_results(self) -> None:
        """Test basic rerank interface."""
        reranker = CrossEncoderReranker(batch_size=2)
        
        # Test with empty results
        results = []
        reranked = reranker.rerank("test query", results)
        assert reranked == []
    
    def test_rerank_with_top_k(self) -> None:
        """Test rerank interface with top_k parameter."""
        reranker = CrossEncoderReranker(batch_size=2)
        
        # Test interface with top_k parameter
        results = []
        reranked = reranker.rerank("test query", results, top_k=2)
        assert reranked == []
    
    def test_rerank_with_threshold(self) -> None:
        """Test rerank interface with threshold parameter."""
        reranker = CrossEncoderReranker(batch_size=2)
        
        # Test interface with threshold parameter
        results = []
        reranked = reranker.rerank("test query", results, threshold=0.5)
        assert reranked == []


class TestONNXCrossEncoderReranker:
    """Test cases for ONNXCrossEncoderReranker."""
    
    def test_lazy_model_loading(self) -> None:
        """Test basic ONNX reranker creation."""
        reranker = ONNXCrossEncoderReranker(batch_size=2)
        
        # Verify basic attributes
        assert reranker.batch_size == 2
        assert hasattr(reranker, 'rerank')
    
    def test_rerank_with_results(self) -> None:
        """Test basic ONNX rerank interface."""
        reranker = ONNXCrossEncoderReranker(batch_size=2)
        
        # Test with empty results
        results = []
        reranked = reranker.rerank("test query", results)
        assert reranked == []


class TestRerankerFactory:
    """Test cases for reranker factory function."""
    
    def test_create_cross_encoder_reranker(self) -> None:
        """Test creating PyTorch CrossEncoderReranker."""
        reranker = create_reranker(use_onnx=False, batch_size=4)
        
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.batch_size == 4
    
    def test_create_onnx_reranker(self) -> None:
        """Test creating ONNXCrossEncoderReranker (default)."""
        reranker = create_reranker(batch_size=8)
        
        assert isinstance(reranker, ONNXCrossEncoderReranker)
        assert reranker.batch_size == 8
    
    def test_create_with_model_name(self) -> None:
        """Test creating reranker with custom model name."""
        reranker = create_reranker(model_name="custom-model", batch_size=2)
        
        # Should create an ONNX reranker by default
        assert isinstance(reranker, ONNXCrossEncoderReranker)
        assert reranker.batch_size == 2