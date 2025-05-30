"""Path definitions for Oboyu CLI.

This module provides centralized path definitions for the application,
following the XDG Base Directory specification.
"""

from xdg_base_dirs import (
    xdg_cache_home,
    xdg_config_home,
    xdg_data_home,
    xdg_state_home,
)

# Application name
APP_NAME = "oboyu"

# Base directories for different types of data
CONFIG_BASE_DIR = xdg_config_home() / APP_NAME
DATA_BASE_DIR = xdg_data_home() / APP_NAME
CACHE_BASE_DIR = xdg_cache_home() / APP_NAME
STATE_BASE_DIR = xdg_state_home() / APP_NAME

# Configuration file path
DEFAULT_CONFIG_PATH = CONFIG_BASE_DIR / "config.yaml"

# Database file path
DEFAULT_DB_PATH = DATA_BASE_DIR / "index.db"

# Embedding models and cache directory
EMBEDDING_MODELS_DIR = DATA_BASE_DIR / "embedding" / "models"
EMBEDDING_CACHE_DIR = CACHE_BASE_DIR / "embedding" / "cache"


# Helper function to ensure all required directories exist
def ensure_config_dirs() -> None:
    """Ensure all required directories exist."""
    CONFIG_BASE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_BASE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
