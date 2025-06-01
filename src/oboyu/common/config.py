"""Unified configuration management system for oboyu."""

from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union

import yaml

from oboyu.common.config_schema import (
    ConfigSchema,
    CrawlerConfigSchema,
    IndexerConfigSchema,
    QueryConfigSchema,
)
from oboyu.common.paths import DEFAULT_CONFIG_PATH, DEFAULT_DB_PATH
from oboyu.crawler.config import DEFAULT_CONFIG as CRAWLER_DEFAULTS
from oboyu.indexer.config import DEFAULT_CONFIG as INDEXER_DEFAULTS

QUERY_ENGINE_DEFAULTS = {
    "top_k": 10,
    "rerank": True,
    "rerank_model": "cl-nagoya/ruri-reranker-small",
    "show_scores": False,
    "interactive": False,
}

T = TypeVar("T", CrawlerConfigSchema, IndexerConfigSchema, QueryConfigSchema)


class ConfigManager:
    """Unified configuration manager with proper precedence handling."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the configuration manager.

        Args:
            config_path: Path to configuration file. Uses default if not provided.

        """
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._config_data: Optional[Dict[str, Any]] = None
        self._defaults = self._build_defaults()

    def _build_defaults(self) -> Dict[str, Any]:
        """Build complete default configuration from all modules.

        Returns:
            Dictionary containing all default configurations.

        """
        # Handle nested structure in CRAWLER_DEFAULTS
        crawler_defaults = CRAWLER_DEFAULTS.get("crawler", CRAWLER_DEFAULTS)
        # Handle nested structure in INDEXER_DEFAULTS
        indexer_defaults = INDEXER_DEFAULTS.get("indexer", INDEXER_DEFAULTS)

        return {
            "crawler": crawler_defaults.copy() if isinstance(crawler_defaults, dict) else {},
            "indexer": indexer_defaults.copy() if isinstance(indexer_defaults, dict) else {},
            "query": QUERY_ENGINE_DEFAULTS.copy(),
        }

    def load_config(self) -> Dict[str, Any]:
        """Load configuration with proper precedence: file > defaults.

        Returns:
            Complete configuration dictionary.

        """
        if self._config_data is not None:
            return self._config_data

        self._config_data = self._defaults.copy()

        if self._config_path.exists():
            try:
                with open(self._config_path) as f:
                    file_config = yaml.safe_load(f) or {}

                # Deep merge file config with defaults
                for section, values in file_config.items():
                    if section in self._config_data and isinstance(values, dict):
                        self._config_data[section].update(values)
                    else:
                        self._config_data[section] = values
            except Exception as e:
                # If config file is invalid, use defaults
                import warnings

                warnings.warn(f"Failed to load config from {self._config_path}: {e}. Using defaults.")

        return self._config_data

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section with defaults applied.

        Args:
            section: Configuration section name (e.g., 'indexer', 'crawler', 'query').

        Returns:
            Configuration dictionary for the specified section.

        """
        config = self.load_config()
        return dict(config.get(section, {}))

    def merge_cli_overrides(self, section: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge CLI arguments with configuration.

        Precedence: CLI args > config file > defaults

        Args:
            section: Configuration section name.
            overrides: Dictionary of CLI override values.

        Returns:
            Merged configuration dictionary.

        """
        base_config = self.get_section(section)

        # Filter out None values from overrides
        filtered_overrides = {k: v for k, v in overrides.items() if v is not None}

        # Apply overrides
        base_config.update(filtered_overrides)

        return dict(base_config)

    def save_config(self, config_data: Optional[Dict[str, Any]] = None) -> None:
        """Save current configuration to file.

        Args:
            config_data: Configuration to save. Uses current config if not provided.

        """
        data_to_save = config_data or self.load_config()

        # Ensure parent directory exists
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._config_path, "w") as f:
            yaml.safe_dump(data_to_save, f, default_flow_style=False, sort_keys=False)

    def resolve_db_path(self, cli_db_path: Optional[Path] = None, section_config: Optional[Dict[str, Any]] = None) -> Path:
        """Resolve database path with proper precedence.

        Precedence: CLI arg > config file > default

        Args:
            cli_db_path: Database path from CLI arguments.
            section_config: Configuration section containing db_path.

        Returns:
            Resolved database path.

        """
        if cli_db_path is not None:
            return cli_db_path

        if section_config and "db_path" in section_config:
            return Path(section_config["db_path"])

        return DEFAULT_DB_PATH

    @property
    def config_path(self) -> Path:
        """Get the configuration file path."""
        return self._config_path

    def get_schema(self) -> ConfigSchema:
        """Get typed configuration schema.

        Returns:
            Complete typed configuration.

        """
        config_dict = self.load_config()
        return ConfigSchema.from_dict(config_dict)

    def get_section_schema(self, section: str) -> Union[CrawlerConfigSchema, IndexerConfigSchema, QueryConfigSchema]:
        """Get typed configuration for a specific section.

        Args:
            section: Configuration section name.

        Returns:
            Typed configuration object for the section.

        Raises:
            ValueError: If section is invalid.

        """
        schema = self.get_schema()

        if section == "crawler":
            return schema.crawler
        elif section == "indexer":
            return schema.indexer
        elif section == "query":
            return schema.query
        else:
            raise ValueError(f"Invalid configuration section: {section}")
