"""Path definitions for Oboyu CLI.

This module provides centralized path definitions for the application,
following the XDG Base Directory specification.
"""

import os
from pathlib import Path

# XDG Base Directory environment variables with defaults
XDG_CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
XDG_CACHE_HOME = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
XDG_STATE_HOME = os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))

# Application name
APP_NAME = "oboyu"

# Base directories for different types of data
CONFIG_BASE_DIR = Path(XDG_CONFIG_HOME) / APP_NAME
DATA_BASE_DIR = Path(XDG_DATA_HOME) / APP_NAME
CACHE_BASE_DIR = Path(XDG_CACHE_HOME) / APP_NAME
STATE_BASE_DIR = Path(XDG_STATE_HOME) / APP_NAME

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
