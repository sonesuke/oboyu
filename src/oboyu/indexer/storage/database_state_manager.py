"""Database state manager for cross-process reliability.

This module provides a DatabaseStateManager that ensures database integrity
and consistency across different processes and operations.

Key features:
- Database integrity validation
- Cross-process state management
- VSS extension state verification
- HNSW index validation and recovery
- Transactional state management
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

import duckdb
from duckdb import DuckDBPyConnection

from oboyu.indexer.storage.index_manager import HNSWIndexParams

from .database_configurator import DatabaseConfigurator
from .index_validator import IndexValidator
from .schema_initializer import SchemaInitializer
from .schema_validator import SchemaValidator
from .vss_extension_manager import VSSExtensionManager

logger = logging.getLogger(__name__)


class DatabaseStateManager:
    """Manages database state for cross-process reliability.
    
    This class ensures that database connections are properly initialized
    and maintained across process boundaries, with proper validation and
    recovery mechanisms.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        embedding_dimensions: int = 256,
        hnsw_params: Optional[HNSWIndexParams] = None,
        enable_experimental_features: bool = True,
    ) -> None:
        """Initialize database state manager.

        Args:
            db_path: Path to the database file
            embedding_dimensions: Dimensions of the embedding vectors
            hnsw_params: HNSW index parameters
            enable_experimental_features: Enable experimental DuckDB features

        """
        self.db_path = Path(db_path)
        self.embedding_dimensions = embedding_dimensions
        self.enable_experimental_features = enable_experimental_features
        self.hnsw_params = hnsw_params or HNSWIndexParams(
            ef_construction=128,
            ef_search=64,
            m=16,
            m0=None,
        )
        
        self._connection: Optional[DuckDBPyConnection] = None
        self._is_validated = False
        
        # Initialize component managers
        self._configurator = DatabaseConfigurator(enable_experimental_features)
        self._vss_manager = VSSExtensionManager()
        self._schema_validator = SchemaValidator()
        self._schema_initializer = SchemaInitializer(self.db_path, embedding_dimensions)
        self._index_validator = IndexValidator(embedding_dimensions, hnsw_params)

    def ensure_initialized(self) -> DuckDBPyConnection:
        """Ensure database is properly initialized and return connection.
        
        This method performs a complete validation and initialization sequence:
        1. Validates database file integrity
        2. Ensures VSS extension is loaded
        3. Validates schema integrity
        4. Validates index integrity
        5. Performs recovery if needed
        
        Returns:
            Validated database connection
            
        Raises:
            RuntimeError: If database cannot be initialized or recovered

        """
        if self._connection is None or not self._is_validated:
            self._connection = self._create_validated_connection()
            self._is_validated = True
            
        return self._connection

    def _create_validated_connection(self) -> DuckDBPyConnection:
        """Create a new connection with full validation and concurrency protection.
        
        Returns:
            Validated database connection
            
        Raises:
            RuntimeError: If validation or recovery fails

        """
        from oboyu.indexer.storage.database_lock import DatabaseLock
        
        logger.debug(f"Creating validated connection to {self.db_path}")
        
        # Use a lock to prevent concurrent database initialization
        lock = DatabaseLock(self.db_path, "db_init")
        
        try:
            with lock.acquire(timeout=30.0):
                logger.debug("Acquired lock for database initialization")
                
                # Create connection with corruption recovery
                try:
                    conn = duckdb.connect(str(self.db_path))
                except Exception as e:
                    if "not a valid DuckDB database file" in str(e):
                        logger.warning(f"Corrupted database file detected: {e}")
                        logger.info("Removing corrupted file and creating fresh database")
                        
                        # Remove corrupted file and create fresh database
                        if self.db_path.exists():
                            self.db_path.unlink()
                        
                        conn = duckdb.connect(str(self.db_path))
                        logger.info("Created fresh database after corruption recovery")
                    else:
                        raise
                
                try:
                    # Step 1: Configure database settings
                    self._configurator.configure_database(conn)
                    
                    # Step 2: Ensure VSS extension is loaded
                    self._vss_manager.ensure_vss_extension(conn)
                    
                    # Step 3: Validate schema integrity
                    if not self._schema_validator.validate_schema_integrity(conn):
                        logger.info("Schema validation failed, initializing fresh schema")
                        self._schema_initializer.initialize_fresh_schema(conn, self._schema_validator)
                    
                    # Step 4: Validate index integrity
                    if not self._index_validator.validate_index_integrity(conn):
                        logger.info("Index validation failed, rebuilding indexes")
                        self._index_validator.rebuild_indexes(conn)
                    
                    logger.info("Database state validation completed successfully")
                    return conn
                    
                except Exception as e:
                    logger.error(f"Failed to create validated connection: {e}")
                    conn.close()
                    raise RuntimeError(f"Database initialization failed: {e}") from e
                    
        except TimeoutError as e:
            logger.error(f"Could not acquire lock for database initialization: {e}")
            # Try to create a simple connection without full initialization
            try:
                conn = duckdb.connect(str(self.db_path))
                # Quick validation that database is usable
                conn.execute("SELECT 1").fetchone()
                logger.info("Database already initialized, using existing connection")
                return conn
            except Exception as conn_error:
                logger.error(f"Failed to create fallback connection: {conn_error}")
                raise RuntimeError("Database initialization timed out and fallback failed")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise RuntimeError(f"Database initialization failed: {e}") from e


    @contextmanager
    def transaction(self) -> Generator[DuckDBPyConnection, None, None]:
        """Context manager for database transactions with state validation.
        
        Yields:
            Validated database connection with active transaction
            
        Raises:
            Exception: If transaction fails and needs to be rolled back

        """
        conn = self.ensure_initialized()
        conn.execute("BEGIN TRANSACTION")
        
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            logger.error("Transaction rolled back due to error")
            # Invalidate state after rollback to force re-validation
            self._is_validated = False
            raise

    def reset_state(self) -> None:
        """Reset state manager to force re-validation on next access.
        
        This should be called after operations that might affect database state
        such as clear operations or schema changes.
        """
        self._is_validated = False
        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                logger.debug(f"Error closing connection during reset: {e}")
            self._connection = None
        
        logger.debug("Database state reset, will re-validate on next access")

    def close(self) -> None:
        """Close database connection and clean up resources."""
        if self._connection:
            try:
                self._connection.close()
                self._connection = None
                self._is_validated = False
                logger.debug("Database state manager closed")
            except Exception as e:
                logger.error(f"Failed to close database connection: {e}")

    def __enter__(self) -> "DatabaseStateManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit."""
        self.close()
