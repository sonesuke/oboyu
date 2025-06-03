"""Tests for ServiceRegistry."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.config.model_config import ModelConfig
from oboyu.indexer.config.processing_config import ProcessingConfig
from oboyu.indexer.config.search_config import SearchConfig
from oboyu.indexer.orchestrators.service_registry import ServiceRegistry


@pytest.fixture
def test_config(tmp_path: Path) -> IndexerConfig:
    """Create test configuration."""
    db_path = tmp_path / "test.db"
    
    return IndexerConfig(
        model=ModelConfig(),
        search=SearchConfig(),
        processing=ProcessingConfig(db_path=db_path),
    )


@pytest.fixture
def service_registry(test_config: IndexerConfig) -> ServiceRegistry:
    """Create service registry for testing."""
    with patch("oboyu.indexer.orchestrators.service_registry.EmbeddingService") as mock_embedding, \
         patch("oboyu.indexer.orchestrators.service_registry.DatabaseService") as mock_db:
        
        # Mock embedding service dimensions
        mock_embedding.return_value.dimensions = 384
        
        # Mock database service initialization
        mock_db.return_value.initialize.return_value = None
        
        return ServiceRegistry(test_config)


def test_service_registry_initialization(service_registry: ServiceRegistry, test_config: IndexerConfig) -> None:
    """Test service registry initialization."""
    assert service_registry.config == test_config
    assert service_registry.get_database_service() is not None
    assert service_registry.get_embedding_service() is not None
    assert service_registry.get_document_processor() is not None
    assert service_registry.get_bm25_indexer() is not None
    assert service_registry.get_tokenizer_service() is not None
    assert service_registry.get_change_detector() is not None


def test_service_registry_services_not_none(service_registry: ServiceRegistry) -> None:
    """Test that all core services are properly initialized."""
    # All basic services should be available
    assert service_registry.get_database_service() is not None
    assert service_registry.get_embedding_service() is not None
    assert service_registry.get_document_processor() is not None
    assert service_registry.get_bm25_indexer() is not None
    assert service_registry.get_tokenizer_service() is not None
    assert service_registry.get_change_detector() is not None


def test_service_registry_database_service(service_registry: ServiceRegistry) -> None:
    """Test that database service is accessible."""
    db_service = service_registry.get_database_service()
    assert db_service is not None
    # Verify it's the right type (would be mocked in our test)
    assert hasattr(db_service, 'initialize')


def test_service_registry_missing_configs() -> None:
    """Test service registry with missing configurations."""
    config = IndexerConfig()
    config.processing = None
    
    with pytest.raises(AssertionError, match="ProcessingConfig should be initialized"):
        ServiceRegistry(config)