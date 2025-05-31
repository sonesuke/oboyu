"""Tests for the reranker module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

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
    
    @patch("oboyu.indexer.reranker._import_cross_encoder")
    def test_initialization(self, mock_import_func: MagicMock) -> None:
        """Test CrossEncoderReranker initialization."""
        # Setup mock to return a mock CrossEncoder class
        mock_cross_encoder_class = MagicMock()
        mock_import_func.return_value = mock_cross_encoder_class
        
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
        assert reranker._model is None  # Lazy loading
    
    @patch("oboyu.indexer.reranker._import_cross_encoder")
    def test_lazy_model_loading(self, mock_import_func: MagicMock) -> None:
        """Test lazy loading of CrossEncoder model."""
        mock_cross_encoder_class = MagicMock()
        mock_model = MagicMock()
        mock_cross_encoder_class.return_value = mock_model
        mock_import_func.return_value = mock_cross_encoder_class
        
        reranker = CrossEncoderReranker(model_name="test-model")
        
        # Model should not be loaded yet
        mock_cross_encoder_class.assert_not_called()
        
        # Access model property
        model = reranker.model
        
        # Now model should be loaded
        mock_cross_encoder_class.assert_called_once_with(
            "test-model",
            device="cpu",
            max_length=512,
            trust_remote_code=True,
        )
        assert model == mock_model
        assert reranker._model == mock_model
    
    @patch("oboyu.indexer.reranker._import_cross_encoder")
    def test_rerank_empty_results(self, mock_import_func: MagicMock) -> None:
        """Test reranking with empty results."""
        reranker = CrossEncoderReranker()
        results = reranker.rerank("test query", [])
        assert results == []
    
    @patch("oboyu.indexer.reranker._import_cross_encoder")
    def test_rerank_with_results(self, mock_import_func: MagicMock) -> None:
        """Test reranking with search results."""
        # Setup mock model
        mock_cross_encoder_class = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 1.5, -0.5])  # Raw scores
        mock_cross_encoder_class.return_value = mock_model
        mock_import_func.return_value = mock_cross_encoder_class
        
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
    
    @patch("oboyu.indexer.reranker._import_cross_encoder")
    def test_rerank_with_top_k(self, mock_import_func: MagicMock) -> None:
        """Test reranking with top_k limit."""
        mock_cross_encoder_class = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 1.5, -0.5])
        mock_cross_encoder_class.return_value = mock_model
        mock_import_func.return_value = mock_cross_encoder_class
        
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
    
    @patch("oboyu.indexer.reranker._import_cross_encoder")
    def test_rerank_with_threshold(self, mock_import_func: MagicMock) -> None:
        """Test reranking with score threshold."""
        mock_cross_encoder_class = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 1.5, -1.5])  # -1.5 -> 0.182 after sigmoid
        mock_cross_encoder_class.return_value = mock_model
        mock_import_func.return_value = mock_cross_encoder_class
        
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
    
    @patch("oboyu.indexer.reranker.get_or_convert_cross_encoder_onnx_model")
    @patch("oboyu.indexer.onnx_converter.ONNXCrossEncoderModel")
    def test_lazy_model_loading(
        self,
        mock_onnx_model_class: MagicMock,
        mock_get_or_convert: MagicMock,
    ) -> None:
        """Test lazy loading of ONNX model."""
        mock_onnx_path = Path("/test/model.onnx")
        mock_get_or_convert.return_value = mock_onnx_path
        
        mock_model = MagicMock()
        mock_onnx_model_class.return_value = mock_model
        
        reranker = ONNXCrossEncoderReranker(model_name="test-model")
        
        # Model should not be loaded yet
        mock_get_or_convert.assert_not_called()
        
        # Access model property
        model = reranker.model
        
        # Now model should be loaded
        mock_get_or_convert.assert_called_once()
        mock_onnx_model_class.assert_called_once_with(
            model_path=mock_onnx_path,
            max_seq_length=512,
            optimization_level="none",
        )
        assert model == mock_model
    
    @patch("oboyu.indexer.reranker.get_or_convert_cross_encoder_onnx_model")
    @patch("oboyu.indexer.onnx_converter.ONNXCrossEncoderModel")
    def test_rerank_with_results(
        self,
        mock_onnx_model_class: MagicMock,
        mock_get_or_convert: MagicMock,
    ) -> None:
        """Test ONNX reranking with search results."""
        # Setup mocks
        mock_get_or_convert.return_value = Path("/test/model.onnx")
        
        mock_model = MagicMock()
        # ONNX model returns normalized scores
        mock_model.predict.return_value = np.array([0.6, 0.9, 0.3])
        mock_onnx_model_class.return_value = mock_model
        
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
        reranker = create_reranker(
            model_name="test-model",
            use_onnx=True,
            device="cpu",
        )
        assert isinstance(reranker, ONNXCrossEncoderReranker)
    
    def test_create_pytorch_reranker(self) -> None:
        """Test creating PyTorch reranker."""
        reranker = create_reranker(
            model_name="test-model",
            use_onnx=False,
            device="cpu",
        )
        assert isinstance(reranker, CrossEncoderReranker)
    
    def test_create_pytorch_reranker_for_cuda(self) -> None:
        """Test creating PyTorch reranker for CUDA."""
        # ONNX is only used for CPU
        reranker = create_reranker(
            model_name="test-model",
            use_onnx=True,
            device="cuda",
        )
        assert isinstance(reranker, CrossEncoderReranker)