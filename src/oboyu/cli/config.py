"""Configuration handling for Oboyu CLI.

This module provides utilities for loading and managing configuration for the CLI.
This is now a thin wrapper around the unified ConfigManager.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from rich.console import Console

from oboyu.common.config import ConfigManager

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
    # Convert to Path if string
    if isinstance(config_path, str):
        config_path = Path(config_path)

    # Use ConfigManager for loading
    manager = ConfigManager(config_path)

    # If config file doesn't exist
    if not manager.config_path.exists():
        # For default path, create it
        if config_path is None:
            create_default_config(manager.config_path)
        else:
            # For explicit path, raise error as before
            raise FileNotFoundError(f"Configuration file not found: {manager.config_path}")

    # Check if file is valid YAML
    try:
        with open(manager.config_path) as f:
            test_load = yaml.safe_load(f)
            if not isinstance(test_load, dict):
                raise ValueError("Configuration file must be a valid YAML dictionary")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")

    return manager.load_config()


def create_default_config(path: Path) -> Dict[str, Any]:
    """Create default configuration file.

    Args:
        path: Path to configuration file

    Returns:
        Default configuration dictionary

    """
    # Use ConfigManager to create default config
    manager = ConfigManager(path)
    config_data = manager.load_config()  # This will use defaults

    # Save the default configuration
    try:
        manager.save_config(config_data)
        console.print(f"Created default configuration at {path}")
    except Exception as e:
        console.print(f"Warning: Could not create default configuration: {e}", style="yellow")

    return config_data
