"""Tests for MCPService."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oboyu.cli.services.mcp_service import MCPService
from oboyu.common.config import ConfigManager


@pytest.fixture
def mock_config_manager():
    """Create a mock config manager."""
    config_manager = Mock(spec=ConfigManager)
    config_manager.get_section.return_value = {
        "db_path": "test.db",
        "use_reranker": False,
    }
    config_manager.resolve_db_path.return_value = Path("test.db")
    return config_manager


@pytest.fixture
def mcp_service(mock_config_manager):
    """Create an MCPService instance."""
    return MCPService(mock_config_manager)


class TestMCPService:
    """Test MCPService functionality."""
    
    def test_init(self, mock_config_manager):
        """Test service initialization."""
        service = MCPService(mock_config_manager)
        assert service.config_manager == mock_config_manager
    
    def test_create_indexer_config(self, mcp_service):
        """Test indexer config creation."""
        result = mcp_service.create_indexer_config("custom.db")
        
        # Verify config manager calls
        mcp_service.config_manager.get_section.assert_called_once_with("indexer")
        mcp_service.config_manager.resolve_db_path.assert_called_once_with(
            Path("custom.db"), 
            mcp_service.config_manager.get_section.return_value
        )
        
        # Verify the result is an IndexerConfig
        from oboyu.indexer.config.indexer_config import IndexerConfig
        assert isinstance(result, IndexerConfig)
    
    def test_create_indexer_config_no_path(self, mcp_service):
        """Test indexer config creation without custom path."""
        result = mcp_service.create_indexer_config()
        
        # Verify resolve_db_path called with None
        mcp_service.config_manager.resolve_db_path.assert_called_with(
            None, 
            mcp_service.config_manager.get_section.return_value
        )
        
        # Verify the result is an IndexerConfig
        from oboyu.indexer.config.indexer_config import IndexerConfig
        assert isinstance(result, IndexerConfig)
    
    def test_get_database_path(self, mcp_service):
        """Test database path resolution."""
        with patch.object(mcp_service, 'create_indexer_config') as mock_create_config:
            mock_config = Mock()
            mock_config.processing.db_path = Path("/resolved/test.db")
            mock_create_config.return_value = mock_config
            
            result = mcp_service.get_database_path()
            
            mock_create_config.assert_called_once_with(db_path=None)
            assert result == "/resolved/test.db"
    
    def test_get_database_path_with_override(self, mcp_service):
        """Test database path resolution with override."""
        with patch.object(mcp_service, 'create_indexer_config') as mock_create_config:
            mock_config = Mock()
            mock_config.processing.db_path = Path("/custom/custom.db")
            mock_create_config.return_value = mock_config
            
            result = mcp_service.get_database_path(Path("/custom/custom.db"))
            
            mock_create_config.assert_called_once_with(db_path="/custom/custom.db")
            assert result == "/custom/custom.db"
    
    @patch('oboyu.cli.services.mcp_service.db_path_global')
    @patch('oboyu.cli.services.mcp_service.mcp')
    def test_start_server(self, mock_mcp, mock_db_path_global, mcp_service):
        """Test starting the MCP server."""
        with patch.object(mcp_service, 'create_indexer_config') as mock_create_config:
            mock_config = Mock()
            mock_config.processing.db_path = Path("/test/server.db")
            mock_create_config.return_value = mock_config
            
            mcp_service.start_server(Path("/test/server.db"), "stdio", 8080)
            
            # Verify config creation
            mock_create_config.assert_called_once_with(db_path="/test/server.db")
            
            # Verify global DB path is set
            assert mock_db_path_global.value == "/test/server.db"
            
            # Verify MCP server is started
            mock_mcp.run.assert_called_once_with("stdio")
    
    @patch('oboyu.cli.services.mcp_service.db_path_global')
    @patch('oboyu.cli.services.mcp_service.mcp')
    def test_start_server_no_path(self, mock_mcp, mock_db_path_global, mcp_service):
        """Test starting the MCP server without custom path."""
        with patch.object(mcp_service, 'create_indexer_config') as mock_create_config:
            mock_config = Mock()
            mock_config.processing.db_path = Path("/default/server.db")
            mock_create_config.return_value = mock_config
            
            mcp_service.start_server(transport="sse", port=None)
            
            # Verify config creation with no path
            mock_create_config.assert_called_once_with(db_path=None)
            
            # Verify global DB path is set
            assert mock_db_path_global.value == "/default/server.db"
            
            # Verify MCP server is started with correct transport
            mock_mcp.run.assert_called_once_with("sse")
    
    @patch('oboyu.cli.services.mcp_service.mcp')
    def test_start_server_exception(self, mock_mcp, mcp_service):
        """Test starting the MCP server when it raises an exception."""
        mock_mcp.run.side_effect = Exception("Server failed to start")
        
        with patch.object(mcp_service, 'create_indexer_config'):
            with pytest.raises(Exception, match="Server failed to start"):
                mcp_service.start_server(transport="stdio")