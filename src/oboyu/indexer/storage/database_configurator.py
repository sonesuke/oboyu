"""Database configuration management."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


class DatabaseConfigurator:
    """Handles database configuration settings."""

    def __init__(self, enable_experimental_features: bool = False) -> None:
        """Initialize the database configurator.
        
        Args:
            enable_experimental_features: Whether to enable experimental features

        """
        self.enable_experimental_features = enable_experimental_features

    def configure_database(self, conn: "DuckDBPyConnection") -> None:
        """Configure database settings for optimal performance."""
        try:
            # Memory and performance settings
            conn.execute("SET memory_limit='2GB'")
            conn.execute("SET threads=4")
            conn.execute("SET enable_progress_bar=false")
            
            # Enable experimental features if requested
            if self.enable_experimental_features:
                try:
                    conn.execute("SET enable_experimental_features=true")
                except Exception:
                    # Try alternative setting name for newer DuckDB versions
                    try:
                        conn.execute("SET enable_external_access=true")
                    except Exception as inner_e:
                        logger.debug(f"Failed to set enable_external_access: {inner_e}")
            
            logger.debug("Database configuration applied successfully")
            
        except Exception as e:
            logger.warning(f"Failed to configure database settings: {e}")
