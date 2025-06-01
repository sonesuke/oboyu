"""Base class for CLI commands with common functionality.

This module provides a base class that consolidates common patterns
across CLI commands to reduce code duplication.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console

from oboyu.cli.hierarchical_logger import create_hierarchical_logger
from oboyu.common.config import ConfigManager
from oboyu.indexer import Indexer
from oboyu.indexer.config.indexer_config import IndexerConfig


class BaseCommand:
    """Base class for all CLI commands with common functionality.

    This class encapsulates common patterns like:
    - Configuration management
    - Database path resolution
    - Indexer initialization
    - Console and logging setup
    """

    def __init__(self, ctx: typer.Context) -> None:
        """Initialize the base command.

        Args:
            ctx: Typer context containing configuration and options

        """
        self.ctx = ctx
        self.console = Console()
        self.logger = create_hierarchical_logger(self.console)

    def get_config_manager(self) -> ConfigManager:
        """Get configuration manager from context.

        Returns:
            ConfigManager instance from context or a new one

        """
        return self.ctx.obj.get("config_manager") if self.ctx.obj else ConfigManager()

    def get_config_data(self) -> Dict[str, Any]:
        """Get configuration data from context.

        Returns:
            Configuration data dictionary

        """
        return self.ctx.obj.get("config_data", {}) if self.ctx.obj else {}

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
        config_manager = self.get_config_manager()
        indexer_config_dict = config_manager.get_section("indexer")

        # Handle database path with clear precedence
        from pathlib import Path

        resolved_db_path = config_manager.resolve_db_path(Path(db_path) if db_path else None, indexer_config_dict)
        indexer_config_dict["db_path"] = str(resolved_db_path)

        # Apply any additional overrides
        indexer_config_dict.update(overrides)

        from oboyu.indexer.config.model_config import ModelConfig
        from oboyu.indexer.config.processing_config import ProcessingConfig
        from oboyu.indexer.config.search_config import SearchConfig

        # Create modular config from dict
        model_config = ModelConfig()
        search_config = SearchConfig()
        processing_config = ProcessingConfig(db_path=Path(indexer_config_dict["db_path"]))

        return IndexerConfig(model=model_config, search=search_config, processing=processing_config)

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
        if show_progress:
            init_op = self.logger.start_operation("Initializing Oboyu indexer...")

            if show_model_loading:
                # Get model name from config for better user feedback
                model_name = config.model.embedding_model if config.model else "unknown"
                load_op = self.logger.start_operation(f"Loading embedding model ({model_name})...")
                indexer = Indexer(config=config)
                self.logger.complete_operation(load_op)
            else:
                indexer = Indexer(config=config)

            self.logger.complete_operation(init_op)
        else:
            indexer = Indexer(config=config)

        return indexer

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
        if force:
            return True

        if db_path:
            self.console.print(f"Using database: {db_path}")

        self.console.print(f"Warning: This will {operation_name} the index database.")
        confirm = typer.confirm("Are you sure you want to continue?")
        if not confirm:
            self.console.print("Operation cancelled.")
            return False

        return True

    def print_database_path(self, db_path: str) -> None:
        """Print the database path being used.

        Args:
            db_path: Database path to display

        """
        self.console.print(f"Using database: {db_path}")

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
        with self.logger.live_display():
            indexer = self.create_indexer(config)

            # Clear the index
            clear_op = self.logger.start_operation("Clearing index database...")
            indexer.clear_index()
            self.logger.complete_operation(clear_op)

            # Clean up resources
            indexer.close()

        self.console.print("\nIndex database cleared successfully!")
