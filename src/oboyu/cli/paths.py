"""Path definitions for Oboyu CLI.

This module provides centralized path definitions for the application,
following the XDG Base Directory specification.
"""

from pathlib import Path

# Base config directory path (following XDG Base Directory specification)
CONFIG_BASE_DIR = Path.home() / ".config" / "oboyu"

# Default configuration file path
DEFAULT_CONFIG_PATH = CONFIG_BASE_DIR / "config.yaml"

# Default database file path
DEFAULT_DB_PATH = CONFIG_BASE_DIR / "oboyu.db"

# Embedding models and cache directory
EMBEDDING_DIR = CONFIG_BASE_DIR / "embedding"
EMBEDDING_MODELS_DIR = EMBEDDING_DIR / "models"
EMBEDDING_CACHE_DIR = EMBEDDING_DIR / "cache"

# Helper function to ensure config directories exist
def ensure_config_dirs() -> None:
    """Ensure that the necessary config directories exist."""
    CONFIG_BASE_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
