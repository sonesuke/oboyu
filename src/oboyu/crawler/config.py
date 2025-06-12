"""Configuration handling for Oboyu crawler.

This module provides utilities for loading and validating crawler configuration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Default configuration values
DEFAULT_CONFIG = {
    "crawler": {
        "depth": 10,
        "include_patterns": [
            "*.txt",
            "*.md",
            "*.html",
            "*.py",
            "*.java",
            "*.pdf",
        ],
        "exclude_patterns": [
            "*/node_modules/*",
            "*/.venv/*",
        ],
        "max_workers": 4,  # Default number of worker threads
        "respect_gitignore": True,  # Whether to respect .gitignore files
    }
}

# Default values for individual settings, used for validation
DEFAULT_DEPTH = 10
DEFAULT_INCLUDE_PATTERNS = ["*.txt", "*.md", "*.html", "*.py", "*.java", "*.pdf"]
DEFAULT_EXCLUDE_PATTERNS = ["*/node_modules/*", "*/venv/*"]
DEFAULT_MAX_WORKERS = 4
DEFAULT_RESPECT_GITIGNORE = True

# Hard-coded values (no longer configurable)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
FOLLOW_SYMLINKS = False


class CrawlerConfig:
    """Configuration handler for the crawler."""

    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize crawler configuration.

        Args:
            config_dict: Optional configuration dictionary
            config_path: Optional path to configuration file

        Note:
            If both config_dict and config_path are provided, config_dict takes precedence.
            If neither is provided, default configuration is used.

        """
        self.config: Dict[str, Any] = {}

        # Start with default configuration
        self.config.update(DEFAULT_CONFIG)

        # If config path is provided, load from file
        if config_path:
            self._load_from_file(config_path)

        # If config dict is provided, override with it
        if config_dict and "crawler" in config_dict:
            self.config["crawler"].update(config_dict["crawler"])

        # Validate the configuration
        self._validate()

    def _load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from a file.

        Args:
            config_path: Path to configuration file

        Note:
            This is a placeholder. In a real implementation, this would use
            a YAML or JSON parser to load the configuration.

        """
        # In a real implementation, this would parse a YAML or JSON file
        # For example:
        # import yaml
        # with open(config_path, "r") as f:
        #     loaded_config = yaml.safe_load(f)
        #     if loaded_config and "crawler" in loaded_config:
        #         self.config["crawler"].update(loaded_config["crawler"])
        pass

    def _validate(self) -> None:
        """Validate the configuration values."""
        # Get the crawler config dict
        crawler_config = self.config["crawler"]

        # Initialize with default values for testing
        # We need to completely reset problematic values rather than update them

        # Validate depth - must be a positive integer
        if not isinstance(crawler_config.get("depth"), int) or crawler_config.get("depth", 0) <= 0:
            # Reset to default instead of updating
            crawler_config["depth"] = DEFAULT_DEPTH

        # Validate include_patterns - must be a list
        if not isinstance(crawler_config.get("include_patterns"), list):
            crawler_config["include_patterns"] = DEFAULT_INCLUDE_PATTERNS[:]

        # Validate exclude_patterns - must be a list and not None
        exclude_patterns = crawler_config.get("exclude_patterns")
        if not isinstance(exclude_patterns, list) or exclude_patterns is None:
            crawler_config["exclude_patterns"] = DEFAULT_EXCLUDE_PATTERNS[:]


        # Validate max_workers - must be a positive integer
        if not isinstance(crawler_config.get("max_workers"), int) or crawler_config.get("max_workers", 0) <= 0:
            crawler_config["max_workers"] = DEFAULT_MAX_WORKERS

        # Validate respect_gitignore - must be a boolean
        if not isinstance(crawler_config.get("respect_gitignore"), bool):
            crawler_config["respect_gitignore"] = DEFAULT_RESPECT_GITIGNORE

    @property
    def depth(self) -> int:
        """Maximum directory traversal depth."""
        return int(self.config["crawler"]["depth"])

    @property
    def include_patterns(self) -> List[str]:
        """File patterns to include."""
        return list(self.config["crawler"]["include_patterns"])

    @property
    def exclude_patterns(self) -> List[str]:
        """Patterns to exclude."""
        return list(self.config["crawler"]["exclude_patterns"])


    @property
    def max_workers(self) -> int:
        """Maximum number of worker threads for parallel processing."""
        return int(self.config["crawler"]["max_workers"])

    @property
    def respect_gitignore(self) -> bool:
        """Whether to respect .gitignore files."""
        return bool(self.config["crawler"]["respect_gitignore"])


def load_default_config() -> CrawlerConfig:
    """Load the default crawler configuration.

    Returns:
        Default crawler configuration

    """
    return CrawlerConfig()


def load_config_from_file(config_path: Union[str, Path]) -> CrawlerConfig:
    """Load crawler configuration from a file.

    Args:
        config_path: Path to configuration file

    Returns:
        Crawler configuration loaded from file

    """
    return CrawlerConfig(config_path=config_path)
