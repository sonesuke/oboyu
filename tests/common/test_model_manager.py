"""Tests for unified model management."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from oboyu.common.model_manager import (
    EmbeddingModelManager,
    ModelManager,
    ONNXModelCache,
    RerankerModelManager,
    create_model_manager,
)


class TestModelManager:
    """Test the abstract ModelManager class."""

    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        # Create concrete implementations for testing
        manager1 = EmbeddingModelManager(
            model_name="test-model",
            use_onnx=True,
            max_seq_length=512,
        )
        
        manager2 = EmbeddingModelManager(
            model_name="test-model",
            use_onnx=True,
            max_seq_length=512,
        )
        
        # Same configuration should generate same cache key
        assert manager1._cache_key == manager2._cache_key
        
        # Different configuration should generate different cache key
        manager3 = EmbeddingModelManager(
            model_name="test-model",
            use_onnx=False,  # Changed to different config to generate different key
            max_seq_length=512,
        )
        
        assert manager1._cache_key != manager3._cache_key

    def test_clear_cache(self):
        """Test that model cache can be cleared."""
        manager = EmbeddingModelManager(model_name="test-model")
        
        # Clear cache
        ModelManager.clear_cache()
        
        # Should not raise any errors
        assert True

    def test_get_cache_dir(self):
        """Test cache directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = EmbeddingModelManager(
                model_name="test/model",
                cache_dir=temp_dir,
            )
            
            cache_dir = manager.get_cache_dir()
            
            # Should create directory and sanitize model name
            assert cache_dir.exists()
            assert "test_model" in str(cache_dir)


class TestEmbeddingModelManager:
    """Test the EmbeddingModelManager class."""

    @patch("oboyu.common.model_manager.ONNXModelCache.get_or_convert_onnx_model")
    @patch("oboyu.common.onnx.embedding_model.ONNXEmbeddingModel")
    def test_onnx_model_loading(self, mock_onnx_model, mock_get_onnx):
        """Test ONNX model loading."""
        mock_get_onnx.return_value = Path("/fake/path/model.onnx")
        mock_instance = Mock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_onnx_model.return_value = mock_instance
        
        manager = EmbeddingModelManager(
            model_name="test-model",
            use_onnx=True,
            max_seq_length=512,
        )
        
        # Access model property to trigger loading
        model = manager.model
        
        # Verify ONNX model was created
        mock_get_onnx.assert_called_once()
        mock_onnx_model.assert_called_once()
        assert model is mock_instance

    @patch("sentence_transformers.SentenceTransformer")
    def test_pytorch_model_loading(self, mock_sentence_transformer):
        """Test PyTorch model loading."""
        mock_instance = Mock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_instance
        
        manager = EmbeddingModelManager(
            model_name="test-model",
            use_onnx=False,
            max_seq_length=512,
        )
        
        # Access model property to trigger loading
        model = manager.model
        
        # Verify SentenceTransformer was created
        mock_sentence_transformer.assert_called_once()
        assert model is mock_instance
        assert model.max_seq_length == 512

    def test_get_dimensions(self):
        """Test getting embedding dimensions."""
        with patch.object(EmbeddingModelManager, "model") as mock_model:
            mock_model.get_sentence_embedding_dimension.return_value = 768
            
            manager = EmbeddingModelManager(model_name="test-model")
            dimensions = manager.get_dimensions()
            
            assert dimensions == 768


class TestRerankerModelManager:
    """Test the RerankerModelManager class."""

    @patch("oboyu.common.model_manager.ONNXModelCache.get_or_convert_onnx_model")
    @patch("oboyu.common.onnx.cross_encoder_model.ONNXCrossEncoderModel")
    def test_onnx_model_loading(self, mock_onnx_model, mock_get_onnx):
        """Test ONNX reranker model loading."""
        mock_get_onnx.return_value = Path("/fake/path/model.onnx")
        mock_instance = Mock()
        mock_onnx_model.return_value = mock_instance
        
        manager = RerankerModelManager(
            model_name="test-reranker",
            use_onnx=True,
            max_length=512,
        )
        
        # Access model property to trigger loading
        model = manager.model
        
        # Verify ONNX model was created
        mock_get_onnx.assert_called_once()
        mock_onnx_model.assert_called_once()
        assert model is mock_instance

    @patch("sentence_transformers.CrossEncoder")
    def test_pytorch_model_loading(self, mock_cross_encoder):
        """Test PyTorch reranker model loading."""
        mock_instance = Mock()
        mock_cross_encoder.return_value = mock_instance
        
        manager = RerankerModelManager(
            model_name="test-reranker",
            use_onnx=False,
            max_length=512,
        )
        
        # Access model property to trigger loading
        model = manager.model
        
        # Verify CrossEncoder was created
        mock_cross_encoder.assert_called_once_with(
            "test-reranker",
            device="cpu",
            max_length=512,
            trust_remote_code=True,
        )
        assert model is mock_instance


class TestONNXModelCache:
    """Test the ONNXModelCache class."""

    def test_get_onnx_path_quantized(self):
        """Test getting ONNX path with quantized preference."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            # Use the correct directory name format (model_name.replace("/", "_"))
            model_dir = cache_dir / "onnx" / "test-model"
            model_dir.mkdir(parents=True)
            
            # Create quantized model
            quantized_path = model_dir / "model_quantized.onnx"
            quantized_path.write_text("fake quantized model")
            
            # Also create other models
            (model_dir / "model_optimized.onnx").write_text("fake optimized model")
            (model_dir / "model.onnx").write_text("fake basic model")
            
            result = ONNXModelCache.get_onnx_path(
                "test-model",
                "embedding",
                cache_dir,
                quantized=True,
            )
            
            assert result == quantized_path

    def test_get_onnx_path_fallback(self):
        """Test ONNX path fallback logic."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            # Use the correct directory name format (model_name.replace("/", "_"))
            model_dir = cache_dir / "onnx" / "test-model"
            model_dir.mkdir(parents=True)
            
            # Only create basic model
            basic_path = model_dir / "model.onnx"
            basic_path.write_text("fake basic model")
            
            result = ONNXModelCache.get_onnx_path(
                "test-model",
                "embedding",
                cache_dir,
                quantized=True,
            )
            
            assert result == basic_path

    def test_get_onnx_path_not_found(self):
        """Test ONNX path when model not found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            
            with pytest.raises(FileNotFoundError):
                ONNXModelCache.get_onnx_path(
                    "nonexistent-model",
                    "embedding",
                    cache_dir,
                )

    @patch("oboyu.common.onnx.embedding_model.convert_to_onnx")
    def test_convert_to_onnx_embedding(self, mock_convert):
        """Test ONNX conversion for embedding model."""
        mock_convert.return_value = Path("/fake/path/model.onnx")
        
        result = ONNXModelCache.convert_to_onnx(
            "test-model",
            "embedding",
            apply_quantization=True,
        )
        
        mock_convert.assert_called_once()
        assert result == Path("/fake/path/model.onnx")

    @patch("oboyu.common.onnx.cross_encoder_model.convert_cross_encoder_to_onnx")
    def test_convert_to_onnx_reranker(self, mock_convert):
        """Test ONNX conversion for reranker model."""
        mock_convert.return_value = Path("/fake/path/model.onnx")
        
        result = ONNXModelCache.convert_to_onnx(
            "test-reranker",
            "reranker",
            apply_quantization=True,
        )
        
        mock_convert.assert_called_once()
        assert result == Path("/fake/path/model.onnx")

    def test_convert_to_onnx_invalid_type(self):
        """Test ONNX conversion with invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            ONNXModelCache.convert_to_onnx(
                "test-model",
                "invalid-type",
            )

    @patch.object(ONNXModelCache, "get_onnx_path")
    @patch.object(ONNXModelCache, "convert_to_onnx")
    def test_get_or_convert_existing(self, mock_convert, mock_get_path):
        """Test get_or_convert when model exists."""
        mock_get_path.return_value = Path("/fake/path/model.onnx")
        
        result = ONNXModelCache.get_or_convert_onnx_model(
            "test-model",
            "embedding",
        )
        
        mock_get_path.assert_called_once()
        mock_convert.assert_not_called()
        assert result == Path("/fake/path/model.onnx")

    @patch.object(ONNXModelCache, "get_onnx_path")
    @patch.object(ONNXModelCache, "convert_to_onnx")
    def test_get_or_convert_not_found(self, mock_convert, mock_get_path):
        """Test get_or_convert when model needs conversion."""
        mock_get_path.side_effect = FileNotFoundError("Not found")
        mock_convert.return_value = Path("/fake/path/model.onnx")
        
        result = ONNXModelCache.get_or_convert_onnx_model(
            "test-model",
            "embedding",
        )
        
        mock_get_path.assert_called_once()
        mock_convert.assert_called_once()
        assert result == Path("/fake/path/model.onnx")


class TestCreateModelManager:
    """Test the create_model_manager factory function."""

    def test_create_embedding_manager(self):
        """Test creating embedding model manager."""
        manager = create_model_manager(
            "embedding",
            "test-model",
            use_onnx=True,
            max_seq_length=512,
        )
        
        assert isinstance(manager, EmbeddingModelManager)
        assert manager.model_name == "test-model"
        assert manager.device == "cpu"
        assert manager.use_onnx is True

    def test_create_reranker_manager(self):
        """Test creating reranker model manager."""
        manager = create_model_manager(
            "reranker",
            "test-reranker",
            use_onnx=False,
            max_length=256,
        )
        
        assert isinstance(manager, RerankerModelManager)
        assert manager.model_name == "test-reranker"
        assert manager.device == "cpu"  # Always CPU-only
        assert manager.use_onnx is False

    def test_create_invalid_type(self):
        """Test creating model manager with invalid type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            create_model_manager(
                "invalid-type",
                "test-model",
            )