"""Tests for the indexer configuration handling functionality."""

from oboyu.indexer.config import (
    DEFAULT_EMBEDDING_MODEL,
    IndexerConfig,
    load_default_config,
)


class TestIndexerConfig:
    """Test cases for the IndexerConfig class."""

    def test_default_config(self) -> None:
        """Test loading default configuration with explicit db_path."""
        # Now we need to provide a db_path explicitly
        test_db_path = "test.db"
        config = IndexerConfig(config_dict={"indexer": {"db_path": test_db_path}})

        # Check default values
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 256
        assert config.embedding_model == "cl-nagoya/ruri-v3-30m"
        assert config.embedding_device == "cpu"
        assert config.batch_size == 128
        assert config.max_seq_length == 8192
        assert config.document_prefix == "検索文書: "
        assert config.query_prefix == "検索クエリ: "
        assert config.topic_prefix == "トピック: "
        assert config.general_prefix == ""
        # Check db_path is properly set
        assert config.db_path == test_db_path
        assert config.ef_construction == 128
        assert config.ef_search == 64
        assert config.m == 16
        assert config.m0 is None
        assert config.max_workers == 4

    def test_config_from_dict(self) -> None:
        """Test loading configuration from dictionary."""
        config_dict = {
            "indexer": {
                "chunk_size": 512,
                "chunk_overlap": 128,
                "embedding_model": "cl-nagoya/ruri-v3-30m",
                "embedding_device": "cuda",
                "batch_size": 16,
                "max_seq_length": 4096,
                "document_prefix": "doc: ",
                "query_prefix": "query: ",
                "topic_prefix": "topic: ",
                "general_prefix": "general: ",
                "db_path": "custom.db",
                "ef_construction": 256,
                "ef_search": 128,
                "m": 32,
                "m0": 64,
                "max_workers": 8,
            }
        }

        config = IndexerConfig(config_dict=config_dict)

        # Check values from dict
        assert config.chunk_size == 512
        assert config.chunk_overlap == 128
        assert config.embedding_model == "cl-nagoya/ruri-v3-30m"
        assert config.embedding_device == "cuda"
        assert config.batch_size == 16
        assert config.max_seq_length == 4096
        assert config.document_prefix == "doc: "
        assert config.query_prefix == "query: "
        assert config.topic_prefix == "topic: "
        assert config.general_prefix == "general: "
        assert config.db_path == "custom.db"
        assert config.ef_construction == 256
        assert config.ef_search == 128
        assert config.m == 32
        assert config.m0 == 64
        assert config.max_workers == 8

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        # Test with invalid values
        config_dict = {
            "indexer": {
                "chunk_size": -1,  # Invalid: negative
                "chunk_overlap": 2000,  # Invalid: greater than chunk_size
                "embedding_device": "invalid",  # Invalid: not 'cpu' or 'cuda'
                "batch_size": 0,  # Invalid: non-positive
                "max_seq_length": -1,  # Invalid: negative
                "ef_construction": 0,  # Invalid: non-positive
                "ef_search": -1,  # Invalid: negative
                "m": 0,  # Invalid: non-positive
                "m0": -1,  # Invalid: negative
                "max_workers": 0,  # Invalid: non-positive
            }
        }

        # This should not raise an exception
        config = IndexerConfig(config_dict=config_dict)

        # All values should now be valid defaults
        assert config.chunk_size == 1024  # Default
        assert config.chunk_overlap == 256  # Default
        assert config.embedding_device == "cpu"  # Default
        assert config.batch_size == 128  # Default
        assert config.max_seq_length == 8192  # Default
        assert config.ef_construction == 128  # Default
        assert config.ef_search == 64  # Default
        assert config.m == 16  # Default
        assert config.m0 is None  # Default
        assert config.max_workers == 4  # Default

    def test_partial_config(self) -> None:
        """Test partial configuration override."""
        # Only override some values
        config_dict = {
            "indexer": {
                "chunk_size": 512,
                "chunk_overlap": 128,
                # Other values not specified - will be set to defaults
            }
        }

        config = IndexerConfig(config_dict=config_dict)

        # Check overridden values
        assert config.chunk_size == 512
        assert config.chunk_overlap == 128

        # Check default values for non-overridden
        # For simplicity, we just verify it's a non-empty string
        assert isinstance(config.embedding_model, str) and config.embedding_model
        assert config.embedding_device == "cpu"
        # For simplicity, we just verify it's a non-empty string
        assert isinstance(config.db_path, str) and config.db_path
    
    def test_use_onnx_property(self) -> None:
        """Test use_onnx property."""
        config = IndexerConfig(config_dict={"indexer": {"db_path": "test.db", "use_onnx": True}})
        assert config.use_onnx is True
        
        config = IndexerConfig(config_dict={"indexer": {"db_path": "test.db", "use_onnx": False}})
        assert config.use_onnx is False
    
    def test_onnx_quantization_properties(self) -> None:
        """Test ONNX quantization properties."""
        # Test defaults
        config = IndexerConfig(config_dict={"indexer": {"db_path": "test.db"}})
        assert config.onnx_quantization_enabled is True
        assert config.onnx_quantization_method == "dynamic"
        assert config.onnx_quantization_weight_type == "uint8"
        assert config.onnx_quantization_config == {
            "enabled": True,
            "method": "dynamic",
            "weight_type": "uint8"
        }
        
        # Test custom config
        custom_config = {
            "indexer": {
                "db_path": "test.db",
                "onnx_quantization": {
                    "enabled": False,
                    "method": "static",
                    "weight_type": "int8"
                }
            }
        }
        config = IndexerConfig(config_dict=custom_config)
        assert config.onnx_quantization_enabled is False
        assert config.onnx_quantization_method == "static"
        assert config.onnx_quantization_weight_type == "int8"
    
    def test_onnx_quantization_validation(self) -> None:
        """Test ONNX quantization configuration validation."""
        # Test with invalid values
        config_dict = {
            "indexer": {
                "db_path": "test.db",
                "onnx_quantization": {
                    "enabled": "yes",  # Should be boolean
                    "method": "invalid",  # Should be dynamic/static/fp16
                    "weight_type": "float32"  # Should be uint8/int8
                }
            }
        }
        
        config = IndexerConfig(config_dict=config_dict)
        
        # Should be set to defaults due to invalid values
        assert config.onnx_quantization_enabled is True
        assert config.onnx_quantization_method == "dynamic"
        assert config.onnx_quantization_weight_type == "uint8"