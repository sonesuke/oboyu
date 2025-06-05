"""Configuration port interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ConfigurationPort(ABC):
    """Abstract interface for configuration management."""
    
    @abstractmethod
    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration."""
        pass
    
    @abstractmethod
    def get_indexing_config(self) -> Dict[str, Any]:
        """Get indexing configuration."""
        pass
    
    @abstractmethod
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding model configuration."""
        pass
    
    @abstractmethod
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        pass
    
    @abstractmethod
    def get_crawler_config(self) -> Dict[str, Any]:
        """Get crawler configuration."""
        pass
    
    @abstractmethod
    def get_value(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        """Get a specific configuration value."""
        pass
    
    @abstractmethod
    def has_key(self, key: str) -> bool:
        """Check if configuration key exists."""
        pass
    
    @abstractmethod
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        pass
