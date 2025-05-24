"""Configuration handling for Oboyu CLI.

This module provides utilities for loading and managing configuration for the CLI.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from rich.console import Console

from oboyu.common.paths import DEFAULT_CONFIG_PATH, DEFAULT_DB_PATH

# Console for output
console = Console()


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from a file.

    Args:
        config_path: Path to configuration file. If None, use default path.

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If the configuration file is not found
        ValueError: If the configuration file is invalid

    """
    # Use default path if not specified
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # Convert to Path object
    path = Path(config_path)

    # Verify file exists
    if not path.exists():
        # Create default configuration if using default path
        if path == DEFAULT_CONFIG_PATH:
            return create_default_config(path)
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Load configuration
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Verify configuration is a dictionary
        if not isinstance(config, dict):
            raise ValueError("Configuration file must be a valid YAML dictionary")

        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")


def create_default_config(path: Path) -> Dict[str, Any]:
    """Create default configuration file.

    Args:
        path: Path to configuration file

    Returns:
        Default configuration dictionary

    """
    # Import here to avoid circular imports
    from oboyu.crawler.config import DEFAULT_CONFIG as CRAWLER_DEFAULT_CONFIG
    from oboyu.indexer.config import DEFAULT_CONFIG as INDEXER_DEFAULT_CONFIG

    # Combine default configurations
    config = {}
    config.update(CRAWLER_DEFAULT_CONFIG)
    config.update(INDEXER_DEFAULT_CONFIG)

    # Set default database path (using centralized path definition)
    config["indexer"]["db_path"] = str(DEFAULT_DB_PATH)

    # Add query engine default config
    config.update({
        "query": {
            "default_mode": "hybrid",  # Default search mode
            "vector_weight": 0.7,  # Weight for vector scores in hybrid search
            "bm25_weight": 0.3,  # Weight for BM25 scores in hybrid search
            "top_k": 5,  # Number of results to return
            "snippet_length": 160,  # Character length for snippets
            "highlight_matches": True,  # Whether to highlight matching terms
        }
    })

    # Ensure directory exists
    os.makedirs(path.parent, exist_ok=True)

    # Write configuration to file
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        console.print(f"Created default configuration at [cyan]{path}[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Warning:[/bold red] Could not create default configuration: {e}")

    return config
