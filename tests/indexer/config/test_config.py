"""Tests for the indexer configuration handling functionality."""

from pathlib import Path

from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.config.model_config import ModelConfig
from oboyu.indexer.config.processing_config import ProcessingConfig
from oboyu.indexer.config.search_config import SearchConfig


class TestIndexerConfig:
    """Test cases for the IndexerConfig class."""

    def test_default_config(self) -> None:
        """Test loading default configuration."""
        # Create config with default sub-configs
        config = IndexerConfig()

        # Check default values through properties
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 256
        assert config.embedding_model == "cl-nagoya/ruri-v3-30m"
        assert config.use_reranker is False

    def test_config_with_custom_values(self) -> None:
        """Test configuration with custom values."""
        # Create custom sub-configs
        model_config = ModelConfig(
            embedding_model="custom-model",
            use_onnx=False,
        )
        
        processing_config = ProcessingConfig(
            db_path=Path("custom.db"),
            chunk_size=512,
            chunk_overlap=128,
        )
        
        search_config = SearchConfig(
            use_reranker=True,
            top_k_multiplier=3,
        )
        
        config = IndexerConfig(
            model=model_config,
            processing=processing_config,
            search=search_config,
        )

        # Check custom values
        assert config.chunk_size == 512
        assert config.chunk_overlap == 128
        assert config.embedding_model == "custom-model"
        assert config.use_reranker is True
        assert str(config.db_path) == "custom.db"

    def test_backward_compatibility_properties(self) -> None:
        """Test backward compatibility properties."""
        # Create config with specific db_path
        processing_config = ProcessingConfig(db_path=Path("test.db"))
        config = IndexerConfig(processing=processing_config)

        # Test db_path property
        assert config.db_path == Path("test.db")
        
        # Test db_path setter
        config.db_path = "new_path.db"
        assert config.db_path == Path("new_path.db")

    def test_post_init_creates_defaults(self) -> None:
        """Test that post_init creates default sub-configs if not provided."""
        # Create config without any sub-configs
        config = IndexerConfig(model=None, processing=None, search=None)
        
        # Verify defaults were created
        assert config.model is not None
        assert config.processing is not None
        assert config.search is not None
        
        # Verify they have default values
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.search, SearchConfig)

    def test_reranker_model_property(self) -> None:
        """Test reranker_model property."""
        model_config = ModelConfig(reranker_model="custom-reranker")
        config = IndexerConfig(model=model_config)
        
        assert config.reranker_model == "custom-reranker"

    def test_use_reranker_combines_configs(self) -> None:
        """Test that use_reranker combines model and search configs."""
        # Test with search config enabled
        search_config = SearchConfig(use_reranker=True)
        model_config = ModelConfig(use_reranker=False)
        config = IndexerConfig(model=model_config, search=search_config)
        assert config.use_reranker is True
        
        # Test with model config enabled
        search_config = SearchConfig(use_reranker=False)
        model_config = ModelConfig(use_reranker=True)
        config = IndexerConfig(model=model_config, search=search_config)
        assert config.use_reranker is True
        
        # Test with both disabled
        search_config = SearchConfig(use_reranker=False)
        model_config = ModelConfig(use_reranker=False)
        config = IndexerConfig(model=model_config, search=search_config)
        assert config.use_reranker is False