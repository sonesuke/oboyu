"""Configuration builder for tracking configuration sources and building immutable configs."""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigSource(Enum):
    """Sources of configuration values in order of precedence."""

    CLI = auto()        # Command line arguments (highest precedence)
    FILE = auto()       # Configuration file
    ENV = auto()        # Environment variables
    DEFAULT = auto()    # System defaults (lowest precedence)


@dataclass(frozen=True)
class ConfigValue(Generic[T]):
    """A configuration value with its source tracked."""

    value: T
    source: ConfigSource
    key: str
    
    def __str__(self) -> str:
        """Return string representation of configuration value."""
        return f"{self.key}={self.value} (from {self.source.name})"


@dataclass
class ConfigurationBuilder:
    """Builds configuration with explicit source tracking."""
    
    _values: Dict[str, ConfigValue] = field(default_factory=dict)
    
    def set_from_cli(self, key: str, value: object) -> None:
        """Set a value from command line arguments."""
        if value is not None:  # Only set if explicitly provided
            self._values[key] = ConfigValue(value=value, source=ConfigSource.CLI, key=key)
            logger.debug(f"Set {key}={value} from CLI")
    
    def set_from_file(self, key: str, value: object) -> None:
        """Set a value from configuration file."""
        # Only set if not already set by higher precedence source
        if key not in self._values or self._values[key].source.value > ConfigSource.FILE.value:
            self._values[key] = ConfigValue(value=value, source=ConfigSource.FILE, key=key)
            logger.debug(f"Set {key}={value} from config file")
    
    def set_from_env(self, key: str, value: object) -> None:
        """Set a value from environment variable."""
        # Only set if not already set by higher precedence source
        if key not in self._values or self._values[key].source.value > ConfigSource.ENV.value:
            self._values[key] = ConfigValue(value=value, source=ConfigSource.ENV, key=key)
            logger.debug(f"Set {key}={value} from environment")
    
    def set_default(self, key: str, value: object) -> None:
        """Set a default value."""
        # Only set if not already set
        if key not in self._values:
            self._values[key] = ConfigValue(value=value, source=ConfigSource.DEFAULT, key=key)
            logger.debug(f"Set {key}={value} as default")
    
    def get(self, key: str, default: object | None = None) -> object | None:
        """Get a configuration value."""
        if key in self._values:
            return self._values[key].value
        return default
    
    def get_with_source(self, key: str) -> Optional[ConfigValue]:
        """Get a configuration value with its source."""
        return self._values.get(key)
    
    def has_explicit_value(self, key: str) -> bool:
        """Check if a value was explicitly set (not from defaults)."""
        return key in self._values and self._values[key].source != ConfigSource.DEFAULT
    
    def log_configuration(self) -> None:
        """Log all configuration values with their sources."""
        if not self._values:
            logger.info("📋 No configuration values set")
            return
            
        logger.info("📋 All configuration values:")
        
        # Group by source for better readability
        by_source: dict[ConfigSource, list[ConfigValue]] = {}
        for config_value in self._values.values():
            source = config_value.source
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(config_value)
        
        # Display in order of precedence
        source_order = [ConfigSource.CLI, ConfigSource.FILE, ConfigSource.ENV, ConfigSource.DEFAULT]
        
        for source in source_order:
            if source in by_source:
                icon = {"CLI": "⌨️", "FILE": "📄", "ENV": "🌍", "DEFAULT": "⚙️"}[source.name]
                logger.info(f"  {icon} From {source.name}:")
                for config_value in sorted(by_source[source], key=lambda x: x.key):
                    logger.info(f"    {config_value.key} = {config_value.value}")
        
        # Highlight explicit vs default settings
        explicit_count = sum(1 for cv in self._values.values() if cv.source != ConfigSource.DEFAULT)
        default_count = sum(1 for cv in self._values.values() if cv.source == ConfigSource.DEFAULT)
        
        logger.info(f"  📊 Summary: {explicit_count} explicit, {default_count} default values")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary of values."""
        return {key: cv.value for key, cv in self._values.items()}
    
    def merge_from(self, other: 'ConfigurationBuilder') -> None:
        """Merge configuration from another builder, respecting precedence."""
        for key, other_value in other._values.items():
            if key not in self._values:
                self._values[key] = other_value
            elif other_value.source.value < self._values[key].source.value:
                # Other has higher precedence
                old_value = self._values[key]
                self._values[key] = other_value
                logger.info(f"🔄 Overriding {key}: {old_value.value} ({old_value.source.name}) → {other_value.value} ({other_value.source.name})")
    
    def check_configuration_conflicts(self) -> None:
        """Check for potential configuration conflicts and log warnings."""
        warnings = []
        
        # Check for reranker configuration conflicts
        use_reranker = self.get("search.use_reranker")
        indexer_use_reranker = self.get("indexer.use_reranker")
        
        if use_reranker is not None and indexer_use_reranker is not None:
            if use_reranker != indexer_use_reranker:
                search_source = self.get_with_source("search.use_reranker")
                indexer_source = self.get_with_source("indexer.use_reranker")
                if search_source is not None and indexer_source is not None:
                    warnings.append(
                        f"⚠️ Reranker setting conflict: search.use_reranker={use_reranker} ({search_source.source.name}) vs "
                        f"indexer.use_reranker={indexer_use_reranker} ({indexer_source.source.name})"
                    )
        
        # Check for model configuration conflicts
        search_model = self.get("search.reranker_model")
        indexer_model = self.get("indexer.reranker_model")
        
        if search_model and indexer_model and search_model != indexer_model:
            warnings.append(
                f"⚠️ Reranker model mismatch: search={search_model} vs indexer={indexer_model}"
            )
        
        # Log all warnings
        if warnings:
            logger.warning("Configuration conflicts detected:")
            for warning in warnings:
                logger.warning(f"  {warning}")
        else:
            logger.debug("✅ No configuration conflicts detected")


