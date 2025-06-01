"""Base class for CLI commands with common functionality.

This module provides a base class that consolidates common patterns
across CLI commands to reduce code duplication.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console

from oboyu.cli.hierarchical_logger import HierarchicalLogger
from oboyu.cli.services import CommandServices
from oboyu.common.config import ConfigManager
from oboyu.indexer import Indexer
from oboyu.indexer.config.indexer_config import IndexerConfig


class BaseCommand:
    """Base class for all CLI commands with common functionality.

    This class uses composition with CommandServices to provide
    focused, testable functionality through specialized service classes.
    """

    def __init__(self, ctx: typer.Context) -> None:
        """Initialize the base command.

        Args:
            ctx: Typer context containing configuration and options

        """
        self.ctx = ctx
        self.services = CommandServices(ctx)

    @property
    def console(self) -> Console:
        """Get the console instance for backward compatibility.
        
        Returns:
            Rich console instance
            
        """
        return self.services.console

    @property
    def logger(self) -> HierarchicalLogger:
        """Get the logger instance for backward compatibility.
        
        Returns:
            Hierarchical logger instance
            
        """
        return self.services.logger

    def get_config_manager(self) -> ConfigManager:
        """Get configuration manager from context.

        Returns:
            ConfigManager instance from context or a new one

        """
        return self.services.config_service.get_config_manager()

    def get_config_data(self) -> Dict[str, Any]:
        """Get configuration data from context.

        Returns:
            Configuration data dictionary

        """
        return self.services.config_service.get_config_data()

    def create_indexer_config(
        self,
        db_path: Optional[str] = None,
        **overrides: Any,  # noqa: ANN401
    ) -> IndexerConfig:
        """Create indexer configuration with proper precedence.

        Args:
            db_path: Optional database path override
            **overrides: Additional configuration overrides

        Returns:
            IndexerConfig instance with proper configuration

        """
        return self.services.indexer_factory.create_indexer_config(db_path, **overrides)

    def create_indexer(
        self,
        config: IndexerConfig,
        show_progress: bool = True,
        show_model_loading: bool = True,
    ) -> Indexer:
        """Create indexer with standardized loading messages.

        Args:
            config: IndexerConfig to use
            show_progress: Whether to show initialization progress
            show_model_loading: Whether to show model loading details

        Returns:
            Initialized Indexer instance

        """
        return self.services.indexer_factory.create_indexer(
            config,
            self.services.console_manager,
            show_progress,
            show_model_loading
        )

    def confirm_database_operation(
        self,
        operation_name: str,
        force: bool = False,
        db_path: Optional[str] = None,
    ) -> bool:
        """Confirm a database operation with the user.

        Args:
            operation_name: Name of the operation (e.g., "clear", "delete")
            force: Whether to skip confirmation
            db_path: Database path for display

        Returns:
            True if operation should proceed, False otherwise

        """
        return self.services.console_manager.confirm_database_operation(operation_name, force, db_path)

    def print_database_path(self, db_path: str) -> None:
        """Print the database path being used.

        Args:
            db_path: Database path to display

        """
        self.services.console_manager.print_database_path(db_path)

    def handle_clear_operation(
        self,
        db_path: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """Handle the common clear database operation.

        Args:
            db_path: Optional database path override
            force: Whether to skip confirmation

        """
        # Create indexer config
        config = self.create_indexer_config(db_path)
        resolved_db_path = config.processing.db_path if config.processing else Path("oboyu.db")

        # Show database path
        self.print_database_path(str(resolved_db_path))

        # Confirm operation
        if not self.confirm_database_operation("remove all indexed documents and search data from", force, str(resolved_db_path)):
            return

        # Perform clear operation with progress tracking
        with self.services.logger.live_display():
            indexer = self.create_indexer(config)

            # Clear the index
            clear_op = self.services.logger.start_operation("Clearing index database...")
            indexer.clear_index()
            self.services.logger.complete_operation(clear_op)

            # Clean up resources
            indexer.close()

        self.services.console.print("\nIndex database cleared successfully!")
