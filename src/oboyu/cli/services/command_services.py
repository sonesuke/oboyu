"""Command services container for CLI commands.

This module provides a container class that holds all service instances
needed by CLI commands, implementing the composition pattern.
"""

import typer
from rich.console import Console

from oboyu.cli.hierarchical_logger import HierarchicalLogger
from oboyu.cli.services.configuration_service import ConfigurationService
from oboyu.cli.services.console_manager import ConsoleManager
from oboyu.cli.services.database_path_resolver import DatabasePathResolver
from oboyu.cli.services.indexer_factory import IndexerFactory


class CommandServices:
    """Container for all CLI command services.
    
    This class implements the composition pattern by holding instances
    of all specialized service classes needed by CLI commands.
    """

    def __init__(self, ctx: typer.Context) -> None:
        """Initialize all command services.

        Args:
            ctx: Typer context containing configuration and options

        """
        self.ctx = ctx
        
        # Initialize core services
        self.console_manager = ConsoleManager()
        self.config_service = ConfigurationService(ctx)
        self.db_path_resolver = DatabasePathResolver(self.config_service)
        self.indexer_factory = IndexerFactory(self.config_service, self.db_path_resolver)

    @property
    def console(self) -> Console:
        """Get the console instance for backward compatibility.
        
        Returns:
            Rich console instance
            
        """
        return self.console_manager.console

    @property
    def logger(self) -> HierarchicalLogger:
        """Get the logger instance for backward compatibility.
        
        Returns:
            Hierarchical logger instance
            
        """
        return self.console_manager.logger
