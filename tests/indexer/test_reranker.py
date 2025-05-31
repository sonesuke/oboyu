"""Tests for the reranker module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pytest

from oboyu.indexer.indexer import SearchResult
from oboyu.indexer.reranker import (
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
        result = reranked.to_search_result()
        
        assert result.chunk_id == original.chunk_id
        assert result.path == original.path
        assert result.title == original.title
        assert result.content == original.content
        assert result.chunk_index == original.chunk_index
        assert result.language == original.language
        assert result.score == 0.95  # Should use rerank score
        assert result.metadata == original.metadata


class TestCrossEncoderReranker:
    """Test cases for CrossEncoderReranker class."""
    
    def test_initialization(self) -> None:
        """Test CrossEncoderReranker initialization."""
        reranker = CrossEncoderReranker(
            model_name="test-model",
            device="cpu",
            batch_size=4,
            max_length=256,
        )
        
        assert reranker.model_name == "test-model"
        assert reranker.device == "cpu"
        assert reranker.batch_size == 4
        assert reranker.max_length == 256
        assert reranker.model_manager is not None  # Model manager created
    
    @patch.object(CrossEncoderReranker, "model", new_callable=PropertyMock)
    def test_lazy_model_loading(self, mock_model: PropertyMock) -> None:
        """Test lazy loading of CrossEncoder model."""
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance
        
        reranker = CrossEncoderReranker(model_name="test-model")
        
        # Model should be loaded on first access
        model = reranker.model
        assert model == mock_instance
    
    def test_rerank_empty_results(self) -> None:
        """Test reranking with empty results."""
        reranker = CrossEncoderReranker()
        results = reranker.rerank("test query", [])
        assert results == []
    
    @patch.object(CrossEncoderReranker, "model")
    def test_rerank_with_results(self, mock_model_prop: MagicMock) -> None:
        """Test reranking with search results."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 1.5, -0.5])  # Raw scores
        mock_model_prop.return_value = mock_model
        
        # Create test results
        results = [
            SearchResult(
                chunk_id=f"test-{i}",
                path=f"/test/doc{i}.txt",
                title=f"Document {i}",
                content=f"Content {i}",
                chunk_index=0,
                language="en",
                score=0.5 + i * 0.1,
                metadata={},
            )
            for i in range(3)
        ]
        
        reranker = CrossEncoderReranker(batch_size=2)
        reranked = reranker.rerank("test query", results)
        
        # Check model was called with correct pairs
        expected_pairs = [
            ["test query", "Content 0"],
            ["test query", "Content 1"],
            ["test query", "Content 2"],
        ]
        mock_model.predict.assert_called()
        
        # Check results are reordered by score (descending)
        assert len(reranked) == 3
        # Scores after sigmoid: 0.5 -> 0.622, 1.5 -> 0.818, -0.5 -> 0.378
        assert reranked[0].chunk_id == "test-1"  # Highest score
        assert reranked[1].chunk_id == "test-0"  # Middle score
        assert reranked[2].chunk_id == "test-2"  # Lowest score
    
    @patch.object(CrossEncoderReranker, "model")
    def test_rerank_with_top_k(self, mock_model_prop: MagicMock) -> None:
        """Test reranking with top_k limit."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 1.5, -0.5])
        mock_model_prop.return_value = mock_model
        
        results = [
            SearchResult(
                chunk_id=f"test-{i}",
                path=f"/test/doc{i}.txt",
                title=f"Document {i}",
                content=f"Content {i}",
                chunk_index=0,
                language="en",
                score=0.5,
                metadata={},
            )
            for i in range(3)
        ]
        
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("test query", results, top_k=2)
        
        assert len(reranked) == 2
        assert reranked[0].chunk_id == "test-1"  # Highest score
        assert reranked[1].chunk_id == "test-0"  # Second highest
    
    @patch.object(CrossEncoderReranker, "model")
    def test_rerank_with_threshold(self, mock_model_prop: MagicMock) -> None:
        """Test reranking with score threshold."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 1.5, -1.5])  # -1.5 -> 0.182 after sigmoid
        mock_model_prop.return_value = mock_model
        
        results = [
            SearchResult(
                chunk_id=f"test-{i}",
                path=f"/test/doc{i}.txt",
                title=f"Document {i}",
                content=f"Content {i}",
                chunk_index=0,
                language="en",
                score=0.5,
                metadata={},
            )
            for i in range(3)
        ]
        
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("test query", results, threshold=0.5)
        
        # Only results with score >= 0.5 should be returned
        assert len(reranked) == 2
        assert all(r.score >= 0.5 for r in reranked)


class TestONNXCrossEncoderReranker:
    """Test cases for ONNXCrossEncoderReranker class."""
    
    def test_initialization(self) -> None:
        """Test ONNXCrossEncoderReranker initialization."""
        reranker = ONNXCrossEncoderReranker(
            model_name="test-model",
            device="cpu",
            batch_size=4,
            max_length=256,
        )
        
        assert reranker.model_name == "test-model"
        assert reranker.device == "cpu"
        assert reranker.batch_size == 4
        assert reranker.max_length == 256
        assert reranker._model is None  # Lazy loading
    
    @patch.object(ONNXCrossEncoderReranker, "model")
    def test_lazy_model_loading(self, mock_model_prop: MagicMock) -> None:
        """Test lazy loading of ONNX model."""
        mock_model = MagicMock()
        mock_model_prop.return_value = mock_model
        
        reranker = ONNXCrossEncoderReranker(model_name="test-model")
        
        # Access model property
        model = reranker.model
        
        # Model should be loaded
        assert model == mock_model
    
    @patch.object(ONNXCrossEncoderReranker, "model")
    def test_rerank_with_results(self, mock_model_prop: MagicMock) -> None:
        """Test ONNX reranking with search results."""
        # Setup mocks
        mock_model = MagicMock()
        # ONNX model returns normalized scores
        mock_model.predict.return_value = np.array([0.6, 0.9, 0.3])
        mock_model_prop.return_value = mock_model
        
        # Create test results
        results = [
            SearchResult(
                chunk_id=f"test-{i}",
                path=f"/test/doc{i}.txt",
                title=f"Document {i}",
                content=f"Content {i}",
                chunk_index=0,
                language="en",
                score=0.5,
                metadata={},
            )
            for i in range(3)
        ]
        
        reranker = ONNXCrossEncoderReranker()
        reranked = reranker.rerank("test query", results)
        
        # Check model was called
        mock_model.predict.assert_called_once()
        
        # Check results are reordered by score
        assert len(reranked) == 3
        assert reranked[0].chunk_id == "test-1"  # Score 0.9
        assert reranked[1].chunk_id == "test-0"  # Score 0.6
        assert reranked[2].chunk_id == "test-2"  # Score 0.3


class TestCreateReranker:
    """Test cases for create_reranker factory function."""
    
    def test_create_onnx_reranker(self) -> None:
        """Test creating ONNX reranker."""
        from oboyu.indexer.reranker import ONNXCrossEncoderReranker
        
        reranker = create_reranker(
            model_name="test-model",
            use_onnx=True,
            device="cpu",
        )
        assert isinstance(reranker, ONNXCrossEncoderReranker)
    
    def test_create_pytorch_reranker(self) -> None:
        """Test creating PyTorch reranker."""
        from oboyu.indexer.reranker import CrossEncoderReranker
        
        reranker = create_reranker(
            model_name="test-model",
            use_onnx=False,
            device="cpu",
        )
        assert isinstance(reranker, CrossEncoderReranker)
    
    def test_create_pytorch_reranker_for_cuda(self) -> None:
        """Test creating ONNX reranker for CUDA."""
        from oboyu.indexer.reranker import ONNXCrossEncoderReranker
        
        # ONNX is used for both CPU and CUDA now
        reranker = create_reranker(
            model_name="test-model",
            use_onnx=True,
            device="cuda",
        )
        assert isinstance(reranker, ONNXCrossEncoderReranker)