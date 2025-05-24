"""Tests for ONNX converter module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oboyu.indexer.onnx_converter import ONNXEmbeddingModel, convert_to_onnx, get_or_convert_onnx_model


class TestONNXEmbeddingModel:
    """Test cases for ONNXEmbeddingModel class."""

    def test_init_with_valid_model(self, tmp_path: Path) -> None:
        """Test initialization with a valid ONNX model."""
        # Create a mock ONNX model file
        model_path = tmp_path / "model.onnx"
        model_path.touch()
        
        # Create mock tokenizer files
        tokenizer_files = ["tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]
        for file in tokenizer_files:
            (tmp_path / file).touch()
        
        # Mock the InferenceSession and AutoTokenizer
        with patch("oboyu.indexer.onnx_converter.InferenceSession") as mock_session, \
             patch("oboyu.indexer.onnx_converter.AutoTokenizer.from_pretrained") as mock_tokenizer:
            
            # Configure mocks
            mock_output = MagicMock()
            mock_output.shape = [1, 256]  # Batch size, embedding dim
            mock_session.return_value.get_outputs.return_value = [mock_output]
            
            # Create model
            model = ONNXEmbeddingModel(model_path)
            
            # Verify initialization
            assert model.model_path == model_path
            assert model.tokenizer_path == tmp_path
            assert model.max_seq_length == 8192
            assert model.pooling_strategy == "mean"
            assert model.dimensions == 256

    def test_encode_single_sentence(self, tmp_path: Path) -> None:
        """Test encoding a single sentence."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()
        
        with patch("oboyu.indexer.onnx_converter.InferenceSession") as mock_session, \
             patch("oboyu.indexer.onnx_converter.AutoTokenizer.from_pretrained") as mock_tokenizer:
            
            # Configure session mock
            mock_output = MagicMock()
            mock_output.shape = [1, 256]
            mock_session.return_value.get_outputs.return_value = [mock_output]
            
            # Mock tokenizer
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.return_value = {
                "input_ids": np.array([[101, 2023, 2003, 102]]),  # Mock token IDs
                "attention_mask": np.array([[1, 1, 1, 1]]),
            }
            
            # Mock model output (token embeddings)
            token_embeddings = np.random.randn(1, 4, 256).astype(np.float32)
            mock_session.return_value.run.return_value = [token_embeddings]
            
            # Create model and encode
            model = ONNXEmbeddingModel(model_path)
            embedding = model.encode("This is a test sentence")
            
            # Verify result
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (256,)
            
            # Check normalization
            norm = np.linalg.norm(embedding)
            assert np.isclose(norm, 1.0, rtol=1e-5)

    def test_encode_batch(self, tmp_path: Path) -> None:
        """Test encoding multiple sentences in a batch."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()
        
        with patch("oboyu.indexer.onnx_converter.InferenceSession") as mock_session, \
             patch("oboyu.indexer.onnx_converter.AutoTokenizer.from_pretrained") as mock_tokenizer:
            
            # Configure mocks
            mock_output = MagicMock()
            mock_output.shape = [1, 256]
            mock_session.return_value.get_outputs.return_value = [mock_output]
            
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            # Mock tokenizer for batch - it will be called twice (batch size = 2)
            # First batch: 2 sentences, second batch: 1 sentence
            batch_outputs = [
                {
                    "input_ids": np.array([[101, 2023, 102], [101, 2003, 102]]),
                    "attention_mask": np.array([[1, 1, 1], [1, 1, 1]]),
                },
                {
                    "input_ids": np.array([[101, 3231, 102]]),
                    "attention_mask": np.array([[1, 1, 1]]),
                }
            ]
            mock_tokenizer_instance.side_effect = batch_outputs
            
            # Mock model output for each batch
            token_embeddings_batch1 = np.random.randn(2, 3, 256).astype(np.float32)
            token_embeddings_batch2 = np.random.randn(1, 3, 256).astype(np.float32)
            mock_session.return_value.run.side_effect = [
                [token_embeddings_batch1],
                [token_embeddings_batch2],
            ]
            
            # Create model and encode batch
            model = ONNXEmbeddingModel(model_path, max_seq_length=512)
            sentences = ["First sentence", "Second sentence", "Third sentence"]
            embeddings = model.encode(sentences, batch_size=2)
            
            # Verify results
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (3, 256)
            
            # Check normalization for each embedding
            for i in range(3):
                norm = np.linalg.norm(embeddings[i])
                assert np.isclose(norm, 1.0, rtol=1e-5)

    def test_pooling_strategies(self, tmp_path: Path) -> None:
        """Test different pooling strategies."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()
        
        with patch("oboyu.indexer.onnx_converter.InferenceSession") as mock_session, \
             patch("oboyu.indexer.onnx_converter.AutoTokenizer.from_pretrained") as mock_tokenizer:
            
            # Configure mocks
            mock_output = MagicMock()
            mock_output.shape = [1, 256]
            mock_session.return_value.get_outputs.return_value = [mock_output]
            
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            # Test mean pooling
            model = ONNXEmbeddingModel(model_path, pooling_strategy="mean")
            token_embeddings = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.float32)
            attention_mask = np.array([[1, 1, 0]], dtype=np.int64)
            
            pooled = model._pool_embeddings(token_embeddings, attention_mask)
            expected = np.array([[2, 3]], dtype=np.float32)  # Mean of first two tokens
            np.testing.assert_array_almost_equal(pooled, expected)
            
            # Test max pooling
            model = ONNXEmbeddingModel(model_path, pooling_strategy="max")
            pooled = model._pool_embeddings(token_embeddings, attention_mask)
            expected = np.array([[3, 4]], dtype=np.float32)  # Max of first two tokens
            np.testing.assert_array_almost_equal(pooled, expected)
            
            # Test CLS pooling
            model = ONNXEmbeddingModel(model_path, pooling_strategy="cls")
            pooled = model._pool_embeddings(token_embeddings, attention_mask)
            expected = np.array([[1, 2]], dtype=np.float32)  # First token
            np.testing.assert_array_almost_equal(pooled, expected)


class TestONNXConversion:
    """Test cases for ONNX conversion functions."""

    @patch("oboyu.indexer.onnx_converter.SentenceTransformer")
    @patch("oboyu.indexer.onnx_converter.torch.onnx.export")
    def test_convert_to_onnx(self, mock_export: MagicMock, mock_st: MagicMock, tmp_path: Path) -> None:
        """Test converting a SentenceTransformer model to ONNX."""
        # Configure mock SentenceTransformer
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        # Mock transformer component
        mock_transformer = MagicMock()
        mock_model.__getitem__.return_value.auto_model = mock_transformer
        mock_model.max_seq_length = 512
        mock_model.get_sentence_embedding_dimension.return_value = 256
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_model.tokenizer = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        
        # Convert model
        model_name = "test-model"
        onnx_path = convert_to_onnx(model_name, tmp_path, optimize=False)
        
        # Verify conversion
        assert onnx_path == tmp_path / "model.onnx"
        mock_st.assert_called_once_with(model_name)
        mock_export.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once_with(tmp_path)
        
        # Check config file
        config_path = tmp_path / "onnx_config.json"
        assert config_path.exists()
        with open(config_path) as f:
            config = json.load(f)
        assert config["max_seq_length"] == 512
        assert config["embedding_dimension"] == 256
        assert config["pooling_strategy"] == "mean"

    @patch("oboyu.indexer.onnx_converter.convert_to_onnx")
    def test_get_or_convert_onnx_model_cached(self, mock_convert: MagicMock, tmp_path: Path) -> None:
        """Test getting cached ONNX model."""
        model_name = "test/model"
        cache_dir = tmp_path / "cache"
        
        # Create cached model
        model_dir = cache_dir / "onnx" / "test_model"
        model_dir.mkdir(parents=True)
        onnx_path = model_dir / "model.onnx"
        onnx_path.touch()
        
        # Get model (should use cache)
        result = get_or_convert_onnx_model(model_name, cache_dir)
        
        assert result == onnx_path
        mock_convert.assert_not_called()

    @patch("oboyu.indexer.onnx_converter.convert_to_onnx")
    def test_get_or_convert_onnx_model_new(self, mock_convert: MagicMock, tmp_path: Path) -> None:
        """Test converting model when not cached."""
        model_name = "test/model"
        cache_dir = tmp_path / "cache"
        
        # Mock conversion
        expected_path = cache_dir / "onnx" / "test_model" / "model.onnx"
        mock_convert.return_value = expected_path
        
        # Get model (should convert)
        result = get_or_convert_onnx_model(model_name, cache_dir)
        
        assert result == expected_path
        mock_convert.assert_called_once_with(model_name, cache_dir / "onnx" / "test_model", optimize=False)