"""Database manager for connection lifecycle and initialization."""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

from duckdb import DuckDBPyConnection

from oboyu.indexer.storage.database_connection import DatabaseConnection
from oboyu.indexer.storage.database_state_manager import DatabaseStateManager
from oboyu.indexer.storage.index_manager import HNSWIndexParams, IndexManager
from oboyu.indexer.storage.migrations import MigrationManager
from oboyu.indexer.storage.schema import DatabaseSchema

logger = logging.getLogger(__name__)


class DatabaseManager(DatabaseConnection):
    """Manages database connection lifecycle, initialization, and configuration.
    
    This class is responsible for:
    - Database connection management
    - Schema creation and migrations
    - Extension setup (VSS)
    - Index management coordination
    - Database configuration and optimization
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        embedding_dimensions: int = 256,
        hnsw_params: Optional[HNSWIndexParams] = None,
        auto_vacuum: bool = True,
        enable_experimental_features: bool = True,
    ) -> None:
        """Initialize database manager.

        Args:
            db_path: Path to the database file
            embedding_dimensions: Dimensions of the embedding vectors
            hnsw_params: HNSW index parameters
            auto_vacuum: Enable automatic database maintenance
            enable_experimental_features: Enable experimental DuckDB features

        """
        # Initialize parent for connection management
        super().__init__(db_path, enable_experimental_features)
        
        self.embedding_dimensions = embedding_dimensions
        self.auto_vacuum = auto_vacuum

        # Set default HNSW parameters if not provided
        self.hnsw_params = hnsw_params or HNSWIndexParams(
            ef_construction=128,
            ef_search=64,
            m=16,
            m0=None,
        )

        # Initialize schema
        self.schema = DatabaseSchema(embedding_dimensions)

        # Initialize state manager for cross-process reliability
        self.state_manager = DatabaseStateManager(
            db_path=db_path,
            embedding_dimensions=embedding_dimensions,
            hnsw_params=self.hnsw_params,
            enable_experimental_features=enable_experimental_features,
        )

        # Managers will be initialized after connection is established
        self.migration_manager: Optional[MigrationManager] = None
        self.index_manager: Optional[IndexManager] = None

        # Connection state
        self.conn: Optional[DuckDBPyConnection] = None
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize the database schema and extensions using state manager."""
        if self._is_initialized:
            return

        try:
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Use state manager to get validated connection
            self.conn = self.state_manager.ensure_initialized()
            self._connection = self.conn  # Set the parent class connection

            # Initialize managers with validated connection
            self.migration_manager = MigrationManager(self.conn, self.schema)
            self.index_manager = IndexManager(self.conn, self.schema)

            self._is_initialized = True
            logger.info(f"Database initialized successfully at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.state_manager.close()
            self.conn = None
            self._connection = None
            raise

    def _configure_database(self) -> None:
        """Configure database settings for optimal performance."""
        if not self.conn:
            return

        try:
            # Memory and performance settings
            self.conn.execute("SET memory_limit='2GB'")
            self.conn.execute("SET threads=4")

            # Enable experimental features if requested
            if self.enable_experimental_features:
                try:
                    self.conn.execute("SET enable_experimental_features=true")
                except Exception:
                    # Try alternative setting name for newer DuckDB versions
                    try:
                        self.conn.execute("SET enable_external_access=true")
                    except Exception as inner_e:
                        logger.debug(f"Failed to set enable_external_access: {inner_e}")

            # Configure auto-vacuum
            if self.auto_vacuum:
                try:
                    self.conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
                except Exception as e:
                    # Auto-vacuum might not be supported in newer DuckDB versions
                    logger.debug(f"Auto-vacuum configuration failed: {e}")

        except Exception as e:
            logger.warning(f"Failed to configure database settings: {e}")
    
    def _configure_vss_settings(self) -> None:
        """Configure VSS-specific settings after extension is loaded."""
        if not self.conn:
            return
            
        try:
            # Enable HNSW experimental persistence for file-based databases
            # This must be done after VSS extension is loaded
            self.conn.execute("SET hnsw_enable_experimental_persistence=true")
            logger.debug("HNSW persistence configured")
        except Exception as e:
            logger.warning(f"Failed to configure VSS settings: {e}")

    def _setup_vss_extension(self) -> None:
        """Install and load the VSS extension for vector operations with state validation."""
        if not self.conn:
            return

        try:
            # Check if VSS extension is already loaded
            try:
                # Try a simple VSS operation to check if it's loaded
                self.conn.execute("SELECT [1.0, 2.0, 3.0]::FLOAT[3]").fetchone()
                logger.debug("VSS extension already loaded")
            except Exception:
                # VSS not loaded, need to install and load
                logger.debug("VSS extension not loaded, installing...")
                
                # Install VSS extension if not already installed
                try:
                    self.conn.execute("INSTALL vss")
                    logger.debug("VSS extension installed")
                except Exception as install_error:
                    # Installation might fail if already installed, which is fine
                    if "already installed" not in str(install_error).lower():
                        logger.warning(f"VSS installation warning: {install_error}")
                
                # Load VSS extension
                self.conn.execute("LOAD vss")
                logger.debug("VSS extension loaded")
            
            # Configure VSS-specific settings
            self._configure_vss_settings()

        except Exception as e:
            logger.error(f"Failed to setup VSS extension: {e}")
            raise

    def _create_schema(self) -> None:
        """Create database schema using schema definitions."""
        if not self.conn:
            return

        try:
            # Get all table definitions
            tables = self.schema.get_all_tables()

            # Create tables in dependency order
            for table in tables:
                self.conn.execute(table.sql)

                # Create indexes for this table
                for index_sql in table.indexes:
                    self.conn.execute(index_sql)

            logger.debug("Database schema created successfully")

        except Exception as e:
            logger.error(f"Failed to create database schema: {e}")
            raise

    def ensure_hnsw_index(self) -> None:
        """Ensure HNSW index exists if there are embeddings."""
        if not self.index_manager:
            return
            
        # Check if index already exists
        if self.index_manager.hnsw_index_exists():
            return
            
        # Create index if we have embeddings but no index
        if self.index_manager._should_create_hnsw_index():
            logger.info("Creating HNSW index after embeddings were added")
            success = self.index_manager.create_hnsw_index(self.hnsw_params)
            if success:
                logger.info("HNSW index created successfully")
            else:
                logger.warning("HNSW index creation failed")
    
    def validate_database_state(self) -> Dict[str, Any]:
        """Validate the current database state and check for inconsistencies.
        
        Returns:
            Dictionary with validation results and any issues found

        """
        validation_results: Dict[str, Any] = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Check if connection is valid
            if not self.conn:
                validation_results["is_valid"] = False
                validation_results["issues"].append("No database connection")
                return validation_results
            
            # Test the connection
            try:
                self.conn.execute("SELECT 1").fetchone()
            except Exception as e:
                validation_results["is_valid"] = False
                validation_results["issues"].append(f"Database connection test failed: {e}")
                return validation_results
            
            # Check VSS extension
            try:
                # Try to create a simple vector to test VSS functionality
                self.conn.execute("SELECT [1.0, 2.0, 3.0]::FLOAT[3]").fetchone()
                validation_results["stats"]["vss_loaded"] = True
            except Exception as e:
                if "vss" in str(e).lower() or "vector" in str(e).lower():
                    validation_results["warnings"].append("VSS extension may not be loaded properly")
                    validation_results["stats"]["vss_loaded"] = False
                else:
                    # VSS is loaded but there might be another issue
                    validation_results["stats"]["vss_loaded"] = True
            
            # Check table existence
            required_tables = ["chunks", "embeddings", "file_metadata"]
            for table in required_tables:
                try:
                    result = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                    count = result[0] if result else 0
                    validation_results["stats"][f"{table}_count"] = count
                except Exception as e:
                    validation_results["is_valid"] = False
                    validation_results["issues"].append(f"Table '{table}' missing or inaccessible: {e}")
            
            # Check HNSW index status
            if self.index_manager:
                has_index = self.index_manager.hnsw_index_exists()
                should_have_index = self.index_manager._should_create_hnsw_index()
                
                validation_results["stats"]["hnsw_index_exists"] = has_index
                validation_results["stats"]["hnsw_index_needed"] = should_have_index
                
                if should_have_index and not has_index:
                    validation_results["warnings"].append("HNSW index missing but embeddings exist")
            
            # Check for orphaned embeddings
            try:
                result = self.conn.execute("""
                    SELECT COUNT(*)
                    FROM embeddings e
                    LEFT JOIN chunks c ON e.chunk_id = c.id
                    WHERE c.id IS NULL
                """).fetchone()
                orphaned = result[0] if result else 0
                
                if orphaned > 0:
                    validation_results["warnings"].append(f"Found {orphaned} orphaned embeddings")
                    validation_results["stats"]["orphaned_embeddings"] = orphaned
            except Exception as e:
                validation_results["warnings"].append(f"Could not check for orphaned embeddings: {e}")
            
            logger.info(f"Database validation completed: {validation_results}")
            
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Validation failed: {e}")
            logger.error(f"Database validation failed: {e}")
        
        return validation_results

    def backup_database(self, backup_path: Union[str, Path]) -> bool:
        """Create a backup of the database.

        Args:
            backup_path: Path for the backup file

        Returns:
            True if backup was successful

        """
        try:
            if self.db_path.exists():
                shutil.copy2(self.db_path, backup_path)
                logger.info(f"Database backed up to {backup_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False

    def get_connection(self) -> DuckDBPyConnection:
        """Get the active database connection.
        
        Returns:
            Active database connection
            
        Raises:
            RuntimeError: If database is not initialized

        """
        if not self._is_initialized or not self.conn:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.conn

    def close(self) -> None:
        """Close database connection and clean up resources."""
        try:
            self.state_manager.close()
            self.conn = None
            self._connection = None
            self._is_initialized = False
            logger.debug("Database connection closed")
        except Exception as e:
            logger.error(f"Failed to close database: {e}")

    def __enter__(self) -> "DatabaseManager":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit."""
        self.close()
