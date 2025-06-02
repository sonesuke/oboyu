"""Path resolution utilities for CLI commands."""

from pathlib import Path
from typing import Any, Dict, Optional

from oboyu.cli.utils.config import ConfigurationService


class DatabasePathResolver:
    """Resolves database paths with proper precedence.
    
    This service handles database path resolution logic, ensuring
    consistent precedence across all CLI commands.
    """

    def __init__(self, config_service: ConfigurationService) -> None:
        """Initialize the database path resolver.

        Args:
            config_service: Configuration service for accessing config data

        """
        self.config_service = config_service

    def resolve_db_path(
        self,
        db_path: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Resolve database path with proper precedence.

        Args:
            db_path: Optional database path override from command line
            config_overrides: Additional configuration overrides

        Returns:
            Resolved database path

        """
        config_manager = self.config_service.get_config_manager()
        indexer_config_dict = config_manager.get_section("indexer")
        
        if config_overrides:
            indexer_config_dict.update(config_overrides)

        return config_manager.resolve_db_path(
            Path(db_path) if db_path else None,
            indexer_config_dict
        )


# Legacy alias for backward compatibility
DatabasePathResolver = DatabasePathResolver
