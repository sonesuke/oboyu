"""Configuration utilities for CLI commands."""

from typing import Any, Dict

import typer

from oboyu.common.config import ConfigManager


class ConfigurationService:
    """Handles configuration loading and management for CLI commands.
    
    This service encapsulates all configuration-related functionality,
    providing a clean interface for accessing configuration data.
    """

    def __init__(self, ctx: typer.Context) -> None:
        """Initialize the configuration service.

        Args:
            ctx: Typer context containing configuration and options

        """
        self.ctx = ctx

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


# Legacy alias for backward compatibility
ConfigurationService = ConfigurationService
