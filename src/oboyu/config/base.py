"""Unified configuration management system for oboyu."""

from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union

import yaml

from oboyu.common.paths import DEFAULT_CONFIG_PATH, DEFAULT_DB_PATH
from oboyu.config.schema import (
    ConfigSchema,
    CrawlerConfigSchema,
    IndexerConfigSchema,
    QueryConfigSchema,
)
from oboyu.config.simplified_schema import (
    AutoOptimizer,
    BackwardCompatibilityMapper,
    SimplifiedConfig,
)

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

    def __init__(self, config_path: Optional[Path] = None, use_simplified: bool = True) -> None:
        """Initialize the configuration manager.

        Args:
            config_path: Path to configuration file. Uses default if not provided.
            use_simplified: Whether to use simplified configuration schema (recommended).

        """
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._config_data: Optional[Dict[str, Any]] = None
        self._use_simplified = use_simplified
        self._simplified_config: Optional[SimplifiedConfig] = None
        self._defaults = self._build_defaults()

    def _build_defaults(self) -> Dict[str, Any]:
        """Build complete default configuration from all modules.

        Returns:
            Dictionary containing all default configurations.

        """
        # Import defaults here to avoid circular imports
        from oboyu.config.crawler import DEFAULT_CONFIG as CRAWLER_DEFAULTS
        from oboyu.config.indexer import DEFAULT_CONFIG as INDEXER_DEFAULTS
        from oboyu.config.query import DEFAULT_CONFIG as QUERY_DEFAULTS
        
        return {
            "crawler": CRAWLER_DEFAULTS["crawler"].__dict__.copy(),
            "indexer": {
                "model": INDEXER_DEFAULTS["indexer"]["model"].__dict__.copy(),
                "search": INDEXER_DEFAULTS["indexer"]["search"].__dict__.copy(),
                "processing": INDEXER_DEFAULTS["indexer"]["processing"].__dict__.copy(),
            },
            "query": QUERY_DEFAULTS["query"].__dict__.copy(),
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

    def get_simplified_config(self) -> SimplifiedConfig:
        """Get simplified configuration with backward compatibility.

        Returns:
            Simplified configuration object with auto-optimized defaults.

        """
        if self._simplified_config is not None:
            return self._simplified_config

        if self._use_simplified:
            # Load raw config data
            raw_config = self.load_config()
            
            # Apply backward compatibility migration
            self._simplified_config = BackwardCompatibilityMapper.migrate_config(raw_config)
        else:
            # Fallback to default simplified config
            self._simplified_config = SimplifiedConfig()

        return self._simplified_config

    def get_auto_optimized_config(self, section: str) -> Dict[str, Any]:
        """Get configuration with auto-optimized parameters for removed options.

        Args:
            section: Configuration section name.

        Returns:
            Configuration with auto-optimized values for performance parameters.

        """
        if self._use_simplified:
            simplified = self.get_simplified_config()
            base_config = getattr(simplified, section).model_dump()
        else:
            base_config = self.get_section(section)

        # Add auto-optimized parameters based on section
        if section == "indexer":
            auto_params = {
                "batch_size": AutoOptimizer.get_optimal_batch_size(),
                "max_workers": AutoOptimizer.get_optimal_max_workers(),
                **AutoOptimizer.get_optimal_hnsw_params(),
                **AutoOptimizer.get_optimal_bm25_params(),
                "reranker_batch_size": min(16, AutoOptimizer.get_optimal_batch_size() // 4),
                "reranker_max_length": 512,
                "use_onnx": True,
                "onnx_quantization": {"enabled": True, "method": "dynamic"},
            }
            base_config.update(auto_params)
            
        elif section == "crawler":
            auto_params = {
                "max_workers": AutoOptimizer.get_optimal_max_workers(),
                "timeout": 30,
                "max_depth": 10,
                "min_doc_length": 50,
                "max_file_size": 10 * 1024 * 1024,  # 10MB
                "follow_symlinks": False,
                "encoding": "utf-8",  # Auto-detected at runtime
            }
            base_config.update(auto_params)
            
        elif section == "query":
            auto_params = {
                "rrf_k": 60,
                "show_scores": False,
                "interactive": False,
                "snippet_length": 200,
                "highlight_matches": True,
            }
            base_config.update(auto_params)

        return base_config

    def save_simplified_config(self, config: Optional[SimplifiedConfig] = None) -> None:
        """Save simplified configuration to file.

        Args:
            config: Simplified configuration to save. Uses current if not provided.

        """
        config_to_save = config or self.get_simplified_config()
        
        # Ensure parent directory exists
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and save
        config_dict = config_to_save.to_dict()
        with open(self._config_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def migrate_to_simplified(self) -> None:
        """Migrate current configuration to simplified format and save.
        
        This will load the current config, migrate it to simplified format,
        and save the new version. Deprecated options will be removed.
        """
        if self._config_path.exists():
            # Load current config
            with open(self._config_path) as f:
                old_config = yaml.safe_load(f) or {}
            
            # Migrate to simplified
            simplified = BackwardCompatibilityMapper.migrate_config(old_config)
            
            # Save new simplified config
            self.save_simplified_config(simplified)
            
            print(f"Configuration migrated to simplified format and saved to {self._config_path}")
        else:
            # Create new simplified config with defaults
            simplified = SimplifiedConfig()
            self.save_simplified_config(simplified)
            print(f"New simplified configuration created at {self._config_path}")
