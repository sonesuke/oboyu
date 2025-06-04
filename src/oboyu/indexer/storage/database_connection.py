"""Database connection management for DuckDB."""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

import duckdb
from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Base class for database connection management."""

    def __init__(
        self,
        db_path: Union[str, Path],
        enable_experimental_features: bool = True,
    ) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to the database file
            enable_experimental_features: Enable experimental DuckDB features

        """
        self.db_path = Path(db_path)
        self.enable_experimental_features = enable_experimental_features
        self._connection: Optional[DuckDBPyConnection] = None

    def _ensure_connection(self) -> DuckDBPyConnection:
        """Ensure database connection is available with retry logic.

        Returns:
            Active database connection

        Raises:
            RuntimeError: If connection cannot be established after retries

        """
        if self._connection is None:
            max_retries = 3
            retry_delay = 0.1  # 100ms initial delay
            
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Creating new DuckDB connection to {self.db_path} (attempt {attempt + 1}/{max_retries})")
                    self._connection = duckdb.connect(str(self.db_path))
                    
                    # Configure DuckDB settings for better concurrent access
                    self._configure_connection()
                    
                    logger.debug("Database connection established successfully")
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to connect to database (attempt {attempt + 1}): {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to connect to database after {max_retries} attempts: {e}")
                        raise RuntimeError(f"Cannot establish database connection: {e}")
                
        if self._connection is None:
            raise RuntimeError("Failed to establish database connection")
        return self._connection
    
    def _configure_connection(self) -> None:
        """Configure database connection for optimal performance and reliability."""
        if not self._connection:
            return
            
        try:
            # Enable parallelism and performance optimizations
            self._connection.execute("SET threads TO 4")
            self._connection.execute("SET memory_limit = '2GB'")
            self._connection.execute("SET enable_progress_bar = false")
            
            # Set timeout for acquiring locks (10 seconds)
            self._connection.execute("SET lock_timeout = '10s'")
            
            if self.enable_experimental_features:
                # Enable experimental features for better performance
                try:
                    self._connection.execute("SET enable_experimental_features = true")
                except Exception:
                    # Try alternative setting name
                    try:
                        self._connection.execute("SET enable_external_access = true")
                    except Exception:
                        logger.debug("Failed to set enable_external_access")
                        
        except Exception as e:
            logger.warning(f"Failed to configure some database settings: {e}")

    @contextmanager
    def transaction(self) -> Generator[DuckDBPyConnection, None, None]:
        """Context manager for database transactions with retry logic.

        Yields:
            Database connection with active transaction

        Raises:
            Exception: If transaction fails after retries

        """
        max_retries = 3
        retry_delay = 0.1  # 100ms initial delay
        
        for attempt in range(max_retries):
            try:
                conn = self._ensure_connection()
                
                # Start transaction with retry on lock timeout
                try:
                    conn.execute("BEGIN TRANSACTION")
                except Exception as e:
                    if "lock" in str(e).lower() and attempt < max_retries - 1:
                        logger.warning(f"Database locked, retrying transaction (attempt {attempt + 1})")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    raise
                
                try:
                    yield conn
                    conn.execute("COMMIT")
                    return  # Success, exit the retry loop
                    
                except Exception as e:
                    conn.execute("ROLLBACK")
                    
                    # Check if this is a recoverable error
                    error_msg = str(e).lower()
                    if ("lock" in error_msg or "concurrent" in error_msg) and attempt < max_retries - 1:
                        logger.warning(f"Transaction failed due to concurrent access, retrying (attempt {attempt + 1}): {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        logger.error(f"Transaction rolled back due to error: {e}")
                        raise
                        
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Transaction failed after {max_retries} attempts: {e}")
                    raise
                    
        # This should not be reached, but just in case
        raise RuntimeError("Transaction failed after all retry attempts")

    def close(self) -> None:
        """Close database connection and clean up resources."""
        if self._connection:
            logger.debug("Closing database connection")
            self._connection.close()
            self._connection = None
