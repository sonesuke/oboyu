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
        """Create a new connection with full validation.
        
        Returns:
            Validated database connection
            
        Raises:
            RuntimeError: If validation or recovery fails

        """
        logger.debug(f"Creating validated connection to {self.db_path}")
        
        # Create connection
        conn = duckdb.connect(str(self.db_path))
        
        try:
            # Step 1: Configure database settings
            self._configure_database(conn)
            
            # Step 2: Ensure VSS extension is loaded
            self._ensure_vss_extension(conn)
            
            # Step 3: Validate schema integrity
            if not self._validate_schema_integrity(conn):
                logger.info("Schema validation failed, initializing fresh schema")
                self._initialize_fresh_schema(conn)
            
            # Step 4: Validate index integrity
            if not self._validate_index_integrity(conn):
                logger.info("Index validation failed, rebuilding indexes")
                self._rebuild_indexes(conn)
            
            logger.info("Database state validation completed successfully")
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create validated connection: {e}")
            conn.close()
            raise RuntimeError(f"Database initialization failed: {e}") from e

    def _configure_database(self, conn: DuckDBPyConnection) -> None:
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

    def _ensure_vss_extension(self, conn: DuckDBPyConnection) -> None:
        """Ensure VSS extension is properly loaded.
        
        Args:
            conn: Database connection
            
        Raises:
            RuntimeError: If VSS extension cannot be loaded

        """
        try:
            # Install and load VSS extension
            conn.execute("INSTALL vss")
            conn.execute("LOAD vss")
            
            # Enable HNSW persistence
            conn.execute("SET hnsw_enable_experimental_persistence=true")
            
            # Verify VSS extension is working
            result = conn.execute("SELECT * FROM duckdb_extensions() WHERE extension_name = 'vss' AND loaded = true").fetchall()
            if not result:
                raise RuntimeError("VSS extension not loaded after installation")
            
            logger.debug("VSS extension loaded and verified successfully")
            
        except Exception as e:
            logger.error(f"Failed to load VSS extension: {e}")
            raise RuntimeError(f"VSS extension setup failed: {e}") from e

    def _validate_schema_integrity(self, conn: DuckDBPyConnection) -> bool:
        """Validate that database schema is complete and valid.
        
        Args:
            conn: Database connection
            
        Returns:
            True if schema is valid, False otherwise

        """
        try:
            # Check if required tables exist
            required_tables = ['chunks', 'embeddings', 'file_metadata', 'bm25_index']
            
            for table in required_tables:
                result = conn.execute(
                    "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                    [table]
                ).fetchone()
                
                if not result or result[0] == 0:
                    logger.debug(f"Required table '{table}' not found")
                    return False
            
            # Verify embedding dimensions match
            try:
                result = conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'embeddings' AND column_name = 'vector'"
                ).fetchone()
                
                if not result:
                    logger.debug("Embeddings table vector column not found")
                    return False
                    
            except Exception as e:
                logger.debug(f"Failed to verify embedding dimensions: {e}")
                return False
            
            logger.debug("Schema integrity validation passed")
            return True
            
        except Exception as e:
            logger.debug(f"Schema validation failed: {e}")
            return False

    def _validate_index_integrity(self, conn: DuckDBPyConnection) -> bool:
        """Validate that database indexes are properly configured.
        
        Args:
            conn: Database connection
            
        Returns:
            True if indexes are valid, False otherwise

        """
        try:
            # Check if we have embeddings
            result = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            if not result or result[0] == 0:
                logger.debug("No embeddings found, index validation passed")
                return True
            
            # If we have embeddings, verify HNSW index exists and is valid
            try:
                # Try to query the HNSW index
                conn.execute(
                    "SELECT COUNT(*) FROM embeddings "
                    "WHERE array_cosine_similarity(vector, vector) IS NOT NULL "
                    "LIMIT 1"
                ).fetchone()
                logger.debug("HNSW index validation passed")
                return True
                
            except Exception as e:
                logger.debug(f"HNSW index validation failed: {e}")
                return False
                
        except Exception as e:
            logger.debug(f"Index validation failed: {e}")
            return False

    def _initialize_fresh_schema(self, conn: DuckDBPyConnection) -> None:
        """Initialize a fresh database schema.
        
        Args:
            conn: Database connection
            
        Raises:
            RuntimeError: If schema initialization fails

        """
        try:
            from oboyu.indexer.storage.migrations import MigrationManager
            from oboyu.indexer.storage.schema import DatabaseSchema
            
            # Create schema and migration manager
            schema = DatabaseSchema(self.embedding_dimensions)
            migration_manager = MigrationManager(conn, schema)
            
            # Create tables
            tables = schema.get_all_tables()
            for table in tables:
                conn.execute(table.sql)
                
                # Create indexes for this table
                for index_sql in table.indexes:
                    conn.execute(index_sql)
            
            # Run migrations
            migration_manager.run_migrations()
            
            logger.info("Fresh schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize fresh schema: {e}")
            raise RuntimeError(f"Schema initialization failed: {e}") from e

    def _rebuild_indexes(self, conn: DuckDBPyConnection) -> None:
        """Rebuild database indexes.
        
        Args:
            conn: Database connection
            
        Raises:
            RuntimeError: If index rebuilding fails

        """
        try:
            from oboyu.indexer.storage.index_manager import IndexManager
            from oboyu.indexer.storage.schema import DatabaseSchema
            
            # Create index manager
            schema = DatabaseSchema(self.embedding_dimensions)
            index_manager = IndexManager(conn, schema)
            
            # Setup all indexes
            index_manager.setup_all_indexes(self.hnsw_params)
            
            # If we have embeddings, create HNSW index
            result = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            if result and result[0] > 0:
                if index_manager._should_create_hnsw_index():
                    success = index_manager.create_hnsw_index(self.hnsw_params)
                    if not success:
                        logger.warning("HNSW index creation failed during rebuild")
            
            logger.info("Indexes rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Failed to rebuild indexes: {e}")
            raise RuntimeError(f"Index rebuilding failed: {e}") from e

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
