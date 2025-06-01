"""Tests for CLI service classes."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer

from oboyu.cli.services import (
    CommandServices,
    ConfigurationService,
    ConsoleManager,
    DatabasePathResolver,
    IndexerFactory,
)


class TestConfigurationService:
    """Test the ConfigurationService class."""

    def test_get_config_manager_with_context(self):
        """Test getting config manager from context."""
        mock_config_manager = Mock()
        ctx = Mock()
        ctx.obj = {"config_manager": mock_config_manager}
        
        service = ConfigurationService(ctx)
        result = service.get_config_manager()
        
        assert result == mock_config_manager

    def test_get_config_manager_without_context(self):
        """Test getting config manager without context."""
        ctx = Mock()
        ctx.obj = None
        
        service = ConfigurationService(ctx)
        result = service.get_config_manager()
        
        assert result is not None

    def test_get_config_data_with_context(self):
        """Test getting config data from context."""
        config_data = {"key": "value"}
        ctx = Mock()
        ctx.obj = {"config_data": config_data}
        
        service = ConfigurationService(ctx)
        result = service.get_config_data()
        
        assert result == config_data

    def test_get_config_data_without_context(self):
        """Test getting config data without context."""
        ctx = Mock()
        ctx.obj = None
        
        service = ConfigurationService(ctx)
        result = service.get_config_data()
        
        assert result == {}


class TestConsoleManager:
    """Test the ConsoleManager class."""

    def test_initialization(self):
        """Test console manager initialization."""
        manager = ConsoleManager()
        
        assert manager.console is not None
        assert manager.logger is not None

    def test_print_database_path(self):
        """Test printing database path."""
        manager = ConsoleManager()
        mock_console = Mock()
        manager.console = mock_console
        
        manager.print_database_path("/path/to/db")
        
        mock_console.print.assert_called_once_with("Using database: /path/to/db")

    def test_confirm_database_operation_force(self):
        """Test confirming database operation with force."""
        manager = ConsoleManager()
        
        result = manager.confirm_database_operation("clear", force=True)
        
        assert result is True

    @patch('typer.confirm')
    def test_confirm_database_operation_user_confirms(self, mock_confirm):
        """Test confirming database operation with user confirmation."""
        mock_confirm.return_value = True
        manager = ConsoleManager()
        mock_console = Mock()
        manager.console = mock_console
        
        result = manager.confirm_database_operation("clear", force=False)
        
        assert result is True
        mock_confirm.assert_called_once()

    @patch('typer.confirm')
    def test_confirm_database_operation_user_cancels(self, mock_confirm):
        """Test confirming database operation with user cancellation."""
        mock_confirm.return_value = False
        manager = ConsoleManager()
        mock_console = Mock()
        manager.console = mock_console
        
        result = manager.confirm_database_operation("clear", force=False)
        
        assert result is False
        mock_confirm.assert_called_once()


class TestDatabasePathResolver:
    """Test the DatabasePathResolver class."""

    def test_resolve_db_path(self):
        """Test resolving database path."""
        mock_config_service = Mock()
        mock_config_manager = Mock()
        mock_config_service.get_config_manager.return_value = mock_config_manager
        mock_config_manager.get_section.return_value = {"db_path": "default.db"}
        mock_config_manager.resolve_db_path.return_value = Path("/resolved/path")
        
        resolver = DatabasePathResolver(mock_config_service)
        result = resolver.resolve_db_path("/custom/path")
        
        assert result == Path("/resolved/path")
        mock_config_manager.resolve_db_path.assert_called_once()


class TestIndexerFactory:
    """Test the IndexerFactory class."""

    def test_initialization(self):
        """Test indexer factory initialization."""
        mock_config_service = Mock()
        mock_db_resolver = Mock()
        
        factory = IndexerFactory(mock_config_service, mock_db_resolver)
        
        assert factory.config_service == mock_config_service
        assert factory.db_path_resolver == mock_db_resolver

    def test_create_indexer_config(self):
        """Test creating indexer configuration."""
        mock_config_service = Mock()
        mock_db_resolver = Mock()
        mock_config_manager = Mock()
        
        mock_config_service.get_config_manager.return_value = mock_config_manager
        mock_config_manager.get_section.return_value = {"use_reranker": False}
        mock_db_resolver.resolve_db_path.return_value = Path("/test/db")
        
        factory = IndexerFactory(mock_config_service, mock_db_resolver)
        result = factory.create_indexer_config("/custom/path")
        
        assert result is not None
        mock_db_resolver.resolve_db_path.assert_called_once_with("/custom/path", {})


class TestCommandServices:
    """Test the CommandServices container class."""

    def test_initialization(self):
        """Test command services initialization."""
        ctx = Mock()
        ctx.obj = None
        
        services = CommandServices(ctx)
        
        assert services.console_manager is not None
        assert services.config_service is not None
        assert services.db_path_resolver is not None
        assert services.indexer_factory is not None

    def test_console_property(self):
        """Test console property access."""
        ctx = Mock()
        ctx.obj = None
        
        services = CommandServices(ctx)
        console = services.console
        
        assert console == services.console_manager.console

    def test_logger_property(self):
        """Test logger property access."""
        ctx = Mock()
        ctx.obj = None
        
        services = CommandServices(ctx)
        logger = services.logger
        
        assert logger == services.console_manager.logger