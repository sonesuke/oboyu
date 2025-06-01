"""Indexer factory service for CLI commands.

This module provides indexer creation and configuration functionality
extracted from the BaseCommand class.
"""

from pathlib import Path
from typing import Any, Optional

from oboyu.cli.services.configuration_service import ConfigurationService
from oboyu.cli.services.database_path_resolver import DatabasePathResolver
from oboyu.indexer import Indexer
from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.config.model_config import ModelConfig
from oboyu.indexer.config.processing_config import ProcessingConfig
from oboyu.indexer.config.search_config import SearchConfig


class IndexerFactory:
    """Creates and configures indexer instances.
    
    This service handles indexer creation with proper configuration
    and standardized loading messages.
    """

    def __init__(
        self,
        config_service: ConfigurationService,
        db_path_resolver: DatabasePathResolver,
    ) -> None:
        """Initialize the indexer factory.

        Args:
            config_service: Configuration service for accessing config data
            db_path_resolver: Database path resolver service

        """
        self.config_service = config_service
        self.db_path_resolver = db_path_resolver

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
        config_manager = self.config_service.get_config_manager()
        indexer_config_dict = config_manager.get_section("indexer")

        # Handle database path with clear precedence
        resolved_db_path = self.db_path_resolver.resolve_db_path(db_path, overrides)
        indexer_config_dict["db_path"] = str(resolved_db_path)

        # Apply any additional overrides
        indexer_config_dict.update(overrides)

        # Create modular config from dict with overrides
        model_config = ModelConfig(
            use_reranker=indexer_config_dict.get("use_reranker", False)
        )
        search_config = SearchConfig(
            use_reranker=indexer_config_dict.get("use_reranker", False)
        )
        processing_config = ProcessingConfig(db_path=Path(indexer_config_dict["db_path"]))

        return IndexerConfig(model=model_config, search=search_config, processing=processing_config)

    def create_indexer(
        self,
        config: IndexerConfig,
        console_manager: Optional[Any] = None,  # noqa: ANN401
        show_progress: bool = True,
        show_model_loading: bool = True,
    ) -> Indexer:
        """Create indexer with standardized loading messages.

        Args:
            config: IndexerConfig to use
            console_manager: Console manager for progress display (optional)
            show_progress: Whether to show initialization progress
            show_model_loading: Whether to show model loading details

        Returns:
            Initialized Indexer instance

        """
        if show_progress and console_manager and show_model_loading:
            # Get model name from config for better user feedback
            model_name = config.model.embedding_model if config.model else "unknown"
            load_op = console_manager.logger.start_operation(f"Loading embedding model ({model_name})...")
            indexer = Indexer(config=config)
            console_manager.logger.complete_operation(load_op)
        else:
            indexer = Indexer(config=config)

        return indexer
