"""Tests for IndexingService."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oboyu.cli.services.indexing_service import IndexingService
from oboyu.common.config import ConfigManager


@pytest.fixture
def mock_config_manager():
    """Create a mock config manager."""
    config_manager = Mock(spec=ConfigManager)
    config_manager.get_section.return_value = {
        "db_path": "test.db",
        "embedding_model": "test-model",
        "chunk_size": 500,
        "chunk_overlap": 50,
    }
    config_manager.resolve_db_path.return_value = Path("test.db")
    return config_manager


@pytest.fixture
def indexing_service(mock_config_manager):
    """Create an IndexingService instance."""
    return IndexingService(mock_config_manager)


class TestIndexingService:
    """Test IndexingService functionality."""
    
    def test_init(self, mock_config_manager):
        """Test service initialization."""
        service = IndexingService(mock_config_manager)
        assert service.config_manager == mock_config_manager
    
    @patch('oboyu.cli.services.indexing_service.create_indexer_config')
    @patch('oboyu.cli.services.indexing_service.build_indexer_config')
    def test_create_indexer_config(self, mock_build, mock_create, indexing_service):
        """Test indexer config creation."""
        mock_create.return_value = {"db_path": "test.db"}
        mock_build.return_value = Mock()
        
        result = indexing_service.create_indexer_config(
            chunk_size=1000,
            chunk_overlap=100,
            embedding_model="custom-model",
            db_path=Path("custom.db")
        )
        
        mock_create.assert_called_once_with(
            indexing_service.config_manager,
            1000,  # chunk_size
            100,   # chunk_overlap
            "custom-model",  # embedding_model
            Path("custom.db")  # db_path
        )
        mock_build.assert_called_once()
        assert result == mock_build.return_value
    
    def test_get_database_path(self, indexing_service):
        """Test database path resolution."""
        with patch('oboyu.cli.services.indexing_service.create_indexer_config') as mock_create:
            mock_create.return_value = {"db_path": "/resolved/path/test.db"}
            
            result = indexing_service.get_database_path()
            
            assert result == "/resolved/path/test.db"
            mock_create.assert_called_once_with(
                indexing_service.config_manager,
                None, None, None, None
            )
    
    def test_get_database_path_with_override(self, indexing_service):
        """Test database path resolution with override."""
        with patch('oboyu.cli.services.indexing_service.create_indexer_config') as mock_create:
            mock_create.return_value = {"db_path": "/custom/path/custom.db"}
            
            result = indexing_service.get_database_path(Path("/custom/path/custom.db"))
            
            assert result == "/custom/path/custom.db"
            mock_create.assert_called_once_with(
                indexing_service.config_manager,
                None, None, None, Path("/custom/path/custom.db")
            )
    
    @patch('oboyu.cli.services.indexing_service.Indexer')
    def test_clear_index(self, mock_indexer_class, indexing_service):
        """Test clearing the index."""
        mock_indexer = Mock()
        mock_indexer_class.return_value = mock_indexer
        
        with patch.object(indexing_service, 'create_indexer_config') as mock_create_config:
            mock_config = Mock()
            mock_create_config.return_value = mock_config
            
            indexing_service.clear_index(Path("test.db"))
            
            mock_create_config.assert_called_once_with(db_path=Path("test.db"))
            mock_indexer_class.assert_called_once_with(config=mock_config)
            mock_indexer.clear_index.assert_called_once()
            mock_indexer.close.assert_called_once()
    
    @patch('oboyu.cli.services.indexing_service.Indexer')
    def test_clear_index_no_db_path(self, mock_indexer_class, indexing_service):
        """Test clearing the index without db_path."""
        mock_indexer = Mock()
        mock_indexer_class.return_value = mock_indexer
        
        with patch.object(indexing_service, 'create_indexer_config') as mock_create_config:
            mock_config = Mock()
            mock_create_config.return_value = mock_config
            
            indexing_service.clear_index()
            
            mock_create_config.assert_called_once_with(db_path=None)
            mock_indexer_class.assert_called_once_with(config=mock_config)
            mock_indexer.clear_index.assert_called_once()
            mock_indexer.close.assert_called_once()