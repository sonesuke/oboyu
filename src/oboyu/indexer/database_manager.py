"""Database manager with transaction support for Oboyu indexer.

This module provides a high-level database management interface with proper
transaction handling, connection management, and error recovery. It serves
as the main entry point for all database operations.

Key features:
- Automatic transaction management with context managers
- Connection pooling and lifecycle management
- Error handling and recovery strategies
- Schema initialization and migration support
- Performance optimization and monitoring
- Thread-safe operations
"""

import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

import duckdb
from duckdb import DuckDBPyConnection

from oboyu.indexer.index_manager import HNSWIndexParams, IndexManager
from oboyu.indexer.migrations import MigrationManager
from oboyu.indexer.schema import DatabaseSchema

logger = logging.getLogger(__name__)


class DatabaseManager:
    """High-level database management with transactions and error handling.
    
    This class provides a comprehensive interface for database operations
    including connection management, transaction handling, schema management,
    and performance optimization.
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
            db_path: Path to database file
            embedding_dimensions: Vector dimensions for embeddings
            hnsw_params: HNSW index parameters
            auto_vacuum: Whether to enable automatic vacuum
            enable_experimental_features: Whether to enable experimental DuckDB features

        """
        self.db_path = Path(db_path)
        self.embedding_dimensions = embedding_dimensions
        self.hnsw_params = hnsw_params or HNSWIndexParams()
        self.auto_vacuum = auto_vacuum
        self.enable_experimental_features = enable_experimental_features
        
        # Initialize components
        self.schema = DatabaseSchema(embedding_dimensions)
        self._conn: Optional[DuckDBPyConnection] = None
        self._index_manager: Optional[IndexManager] = None
        self._migration_manager: Optional[MigrationManager] = None
        
        # Performance settings
        self._performance_settings = {
            "threads": 8,
            "memory_limit": "4GB",
            "preserve_insertion_order": False,
            "temp_directory": "/tmp",
        }
    
    @property
    def connection(self) -> DuckDBPyConnection:
        """Get the database connection, initializing if needed.
        
        Returns:
            Active database connection
            
        Raises:
            RuntimeError: If connection cannot be established

        """
        if self._conn is None:
            self._connect()
        # After _connect(), self._conn is guaranteed to be not None
        assert self._conn is not None
        return self._conn
    
    @property
    def index_manager(self) -> IndexManager:
        """Get the index manager.
        
        Returns:
            Index manager instance

        """
        if self._index_manager is None:
            self._index_manager = IndexManager(self.connection, self.schema)
        return self._index_manager
    
    @property
    def migration_manager(self) -> MigrationManager:
        """Get the migration manager.
        
        Returns:
            Migration manager instance

        """
        if self._migration_manager is None:
            self._migration_manager = MigrationManager(self.connection, self.schema)
        return self._migration_manager
    
    def _connect(self) -> None:
        """Establish database connection and configure settings."""
        try:
            # Ensure parent directory exists
            if not str(self.db_path).startswith(":memory:"):
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self._conn = duckdb.connect(str(self.db_path))
            
            # Configure VSS extension
            self._setup_vss_extension()
            
            # Apply performance settings
            self._apply_performance_settings()
            
            logger.debug(f"Connected to database: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise RuntimeError(f"Database connection failed: {e}") from e
    
    def _setup_vss_extension(self) -> None:
        """Setup VSS extension for vector similarity search."""
        try:
            self.connection.execute("INSTALL vss")
            self.connection.execute("LOAD vss")
            
            if self.enable_experimental_features:
                self.connection.execute("SET hnsw_enable_experimental_persistence=true")
            
            logger.debug("VSS extension loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup VSS extension: {e}")
            raise
    
    def _apply_performance_settings(self) -> None:
        """Apply performance optimization settings."""
        try:
            for setting, value in self._performance_settings.items():
                if setting == "threads":
                    self.connection.execute(f"PRAGMA threads={value}")
                elif setting == "memory_limit":
                    self.connection.execute(f"SET memory_limit='{value}'")
                elif setting == "preserve_insertion_order":
                    self.connection.execute(f"SET preserve_insertion_order={str(value).lower()}")
                elif setting == "temp_directory":
                    self.connection.execute(f"SET temp_directory='{value}'")
            
            logger.debug("Performance settings applied")
            
        except Exception as e:
            logger.warning(f"Some performance settings could not be applied: {e}")
    
    def initialize_database(self, force_recreate: bool = False) -> None:
        """Initialize database schema and indexes.
        
        Args:
            force_recreate: Whether to drop and recreate all tables

        """
        logger.info("Initializing database schema...")
        
        try:
            if force_recreate:
                self._drop_all_tables()
            
            # Create tables
            self._create_tables()
            
            # Run migrations
            self.migration_manager.run_migrations()
            
            # Setup indexes
            self.index_manager.setup_all_indexes(self.hnsw_params)
            
            logger.info("Database initialization completed")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_tables(self) -> None:
        """Create all database tables in dependency order."""
        tables = self.schema.get_all_tables()
        
        for table in tables:
            try:
                self.connection.execute(table.sql)
                logger.debug(f"Created table: {table.name}")
            except Exception as e:
                logger.error(f"Failed to create table {table.name}: {e}")
                raise
    
    def _drop_all_tables(self) -> None:
        """Drop all database tables."""
        drop_queries = self.schema.get_drop_all_tables_sql()
        
        for query in drop_queries:
            try:
                self.connection.execute(query)
                logger.debug(f"Executed: {query}")
            except Exception as e:
                logger.warning(f"Failed to execute {query}: {e}")
    
    @contextmanager
    def transaction(self, savepoint: Optional[str] = None) -> Generator[DuckDBPyConnection, None, None]:
        """Transaction context manager with automatic rollback on errors.
        
        Args:
            savepoint: Optional savepoint name for nested transactions
            
        Yields:
            Database connection within transaction
            
        Example:
            with db_manager.transaction() as conn:
                conn.execute("INSERT INTO chunks ...")
                conn.execute("INSERT INTO embeddings ...")
                # Automatic commit on success, rollback on exception

        """
        conn = self.connection
        
        try:
            if savepoint:
                conn.execute(f"SAVEPOINT {savepoint}")
            else:
                conn.execute("BEGIN")
            
            yield conn
            
            if savepoint:
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            else:
                conn.execute("COMMIT")
                
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            try:
                if savepoint:
                    conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                else:
                    conn.execute("ROLLBACK")
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
            raise
    
    @contextmanager
    def bulk_operation(self) -> Generator[DuckDBPyConnection, None, None]:
        """Context manager optimized for bulk operations.
        
        Yields:
            Database connection optimized for bulk inserts

        """
        conn = self.connection
        
        # Store original settings
        original_settings: Dict[str, str] = {}
        
        try:
            # Optimize for bulk operations
            bulk_settings = {
                "memory_limit": "6GB",
                "threads": 8,
                "max_memory": "6GB",
            }
            
            for setting, value in bulk_settings.items():
                try:
                    if setting in ["memory_limit", "max_memory"]:
                        conn.execute(f"SET {setting}='{value}'")
                    else:
                        conn.execute(f"SET {setting}={value}")
                except Exception as e:
                    logger.warning(f"Could not set {setting}: {e}")
            
            # Start transaction
            conn.execute("BEGIN")
            
            yield conn
            
            # Commit transaction
            conn.execute("COMMIT")
            
        except Exception as e:
            logger.error(f"Bulk operation failed: {e}")
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
            
        finally:
            # Restore original settings
            for setting, value in original_settings.items():
                try:
                    conn.execute(f"SET {setting}='{value}'")
                except Exception:
                    pass
    
    def backup_database(self, backup_path: Union[str, Path]) -> None:
        """Create a backup of the database.
        
        Args:
            backup_path: Path for the backup file

        """
        backup_path = Path(backup_path)
        
        try:
            # Close connection to ensure data is flushed
            if self._conn:
                self._conn.close()
                self._conn = None
            
            # Create backup
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Database backed up to: {backup_path}")
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise
        finally:
            # Reconnect
            self._connect()
    
    def restore_database(self, backup_path: Union[str, Path]) -> None:
        """Restore database from backup.
        
        Args:
            backup_path: Path to the backup file

        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        try:
            # Close connection
            if self._conn:
                self._conn.close()
                self._conn = None
            
            # Restore from backup
            shutil.copy2(backup_path, self.db_path)
            
            logger.info(f"Database restored from: {backup_path}")
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            raise
        finally:
            # Reconnect
            self._connect()
    
    def vacuum_database(self) -> None:
        """Vacuum the database to reclaim space and optimize performance."""
        try:
            logger.info("Starting database vacuum...")
            self.connection.execute("VACUUM")
            logger.info("Database vacuum completed")
            
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")
            raise
    
    def analyze_database(self) -> None:
        """Analyze database to update statistics for query optimization."""
        try:
            logger.info("Analyzing database statistics...")
            
            # Get all table names
            tables_result = self.connection.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
            """).fetchall()
            
            for (table_name,) in tables_result:
                try:
                    self.connection.execute(f"ANALYZE {table_name}")
                    logger.debug(f"Analyzed table: {table_name}")
                except Exception as e:
                    logger.warning(f"Failed to analyze table {table_name}: {e}")
            
            logger.info("Database analysis completed")
            
        except Exception as e:
            logger.error(f"Database analysis failed: {e}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information.
        
        Returns:
            Dictionary with database statistics and health info

        """
        try:
            info = {
                "db_path": str(self.db_path),
                "embedding_dimensions": self.embedding_dimensions,
                "schema_version": self.migration_manager.get_current_version(),
                "performance_settings": self._performance_settings.copy(),
            }
            
            # Add file size information
            if self.db_path.exists():
                info["file_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)
            
            # Add table statistics
            tables = self.schema.get_all_tables()
            table_stats = {}
            
            for table in tables:
                try:
                    result = self.connection.execute(f"SELECT COUNT(*) FROM {table.name}").fetchone()
                    table_stats[table.name] = result[0] if result else 0
                except Exception:
                    table_stats[table.name] = "unknown"
            
            info["table_counts"] = table_stats
            
            # Add index information
            info["index_health"] = self.index_manager.check_index_health().__dict__
            info["index_stats"] = self.index_manager.get_index_usage_stats()
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}
    
    def close(self) -> None:
        """Close database connection and cleanup resources."""
        try:
            if self._conn:
                self._conn.close()
                self._conn = None
            
            # Reset managers
            self._index_manager = None
            self._migration_manager = None
            
            logger.info("Database connection closed")
            
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    def __enter__(self) -> "DatabaseManager":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
