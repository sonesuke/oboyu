"""Database connection management for DuckDB."""

import logging
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
        """Ensure database connection is available.

        Returns:
            Active database connection

        """
        if self._connection is None:
            logger.debug(f"Creating new DuckDB connection to {self.db_path}")
            self._connection = duckdb.connect(str(self.db_path))
            
            # Configure DuckDB settings
            if self.enable_experimental_features:
                # Enable parallelism and performance optimizations
                self._connection.execute("SET threads TO 4")
                self._connection.execute("SET memory_limit = '2GB'")
                self._connection.execute("SET enable_progress_bar = false")
                
        return self._connection

    @contextmanager
    def transaction(self) -> Generator[DuckDBPyConnection, None, None]:
        """Context manager for database transactions.

        Yields:
            Database connection with active transaction

        Raises:
            Exception: If transaction fails and needs to be rolled back

        """
        conn = self._ensure_connection()
        conn.execute("BEGIN TRANSACTION")
        
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            logger.error("Transaction rolled back due to error")
            raise

    def close(self) -> None:
        """Close database connection and clean up resources."""
        if self._connection:
            logger.debug("Closing database connection")
            self._connection.close()
            self._connection = None
