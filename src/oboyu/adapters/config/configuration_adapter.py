"""Configuration adapter implementation."""

import logging
from typing import Any, Dict

from ...cli.services.configuration_service import ConfigurationService
from ...ports.external.configuration_port import ConfigurationPort

logger = logging.getLogger(__name__)


class ConfigurationAdapter(ConfigurationPort):
    """Configuration adapter that wraps existing configuration."""
    
    def __init__(self, config_service: ConfigurationService) -> None:
        """Initialize with existing configuration service."""
        self._config_service = config_service
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration."""
        return self._config_service.get_search_config()
    
    def get_indexing_config(self) -> Dict[str, Any]:
        """Get indexing configuration."""
        return self._config_service.get_indexing_config()
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding model configuration."""
        return self._config_service.get_embedding_config()
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self._config_service.get_database_config()
    
    def get_crawler_config(self) -> Dict[str, Any]:
        """Get crawler configuration."""
        return self._config_service.get_crawler_config()
    
    def get_value(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        """Get a specific configuration value."""
        return getattr(self._config_service, key, default)
    
    def has_key(self, key: str) -> bool:
        """Check if configuration key exists."""
        return hasattr(self._config_service, key)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return self._config_service.to_dict() if hasattr(self._config_service, 'to_dict') else {}
