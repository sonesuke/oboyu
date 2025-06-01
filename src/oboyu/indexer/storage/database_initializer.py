"""Database initialization logic for DuckDB with VSS extension."""

import logging
from pathlib import Path

import duckdb
from duckdb import DuckDBPyConnection

from oboyu.indexer.storage.index_manager import HNSWIndexParams, IndexManager
from oboyu.indexer.storage.migrations import MigrationManager
from oboyu.indexer.storage.schema import DatabaseSchema

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handles database initialization and configuration."""

    def __init__(
        self,
        db_path: Path,
        schema: DatabaseSchema,
        auto_vacuum: bool = True,
        enable_experimental_features: bool = True
    ) -> None:
        """Initialize the database initializer.
        
        Args:
            db_path: Path to the database file
            schema: Database schema instance
            auto_vacuum: Enable automatic database maintenance
            enable_experimental_features: Enable experimental DuckDB features

        """
        self.db_path = db_path
        self.schema = schema
        self.auto_vacuum = auto_vacuum
        self.enable_experimental_features = enable_experimental_features

    def initialize(self, hnsw_params: HNSWIndexParams) -> DuckDBPyConnection:
        """Initialize the database schema and extensions.
        
        Args:
            hnsw_params: HNSW index parameters
            
        Returns:
            Database connection
            
        Raises:
            Exception: If initialization fails

        """
        try:
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            conn = duckdb.connect(str(self.db_path))

            # Configure database settings
            self._configure_database(conn)

            # Install and load VSS extension
            self._setup_vss_extension(conn)

            # Initialize managers with connection
            migration_manager = MigrationManager(conn, self.schema)
            index_manager = IndexManager(conn, self.schema)

            # Create database schema
            self._create_schema(conn)

            # Run migrations
            migration_manager.run_migrations()

            # Initialize HNSW index
            index_manager.setup_all_indexes(hnsw_params)

            logger.info(f"Database initialized successfully at {self.db_path}")
            return conn

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            if 'conn' in locals():
                conn.close()
            raise

    def _configure_database(self, conn: DuckDBPyConnection) -> None:
        """Configure database settings for optimal performance."""
        try:
            # Memory and performance settings
            conn.execute("SET memory_limit='2GB'")
            conn.execute("SET threads=4")
            
            # Enable HNSW experimental persistence for file-based databases
            conn.execute("SET hnsw_enable_experimental_persistence=true")

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

            # Configure auto-vacuum
            if self.auto_vacuum:
                try:
                    conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
                except Exception as e:
                    # Auto-vacuum might not be supported in newer DuckDB versions
                    logger.debug(f"Auto-vacuum configuration failed: {e}")

        except Exception as e:
            logger.warning(f"Failed to configure database settings: {e}")

    def _setup_vss_extension(self, conn: DuckDBPyConnection) -> None:
        """Install and load the VSS extension for vector operations."""
        try:
            # Install VSS extension
            conn.execute("INSTALL vss")
            conn.execute("LOAD vss")
            logger.debug("VSS extension loaded successfully")

        except Exception as e:
            logger.error(f"Failed to setup VSS extension: {e}")
            raise

    def _create_schema(self, conn: DuckDBPyConnection) -> None:
        """Create database schema using schema definitions."""
        try:
            # Get all table definitions
            tables = self.schema.get_all_tables()

            # Create tables in dependency order
            for table in tables:
                conn.execute(table.sql)

                # Create indexes for this table
                for index_sql in table.indexes:
                    conn.execute(index_sql)

            logger.debug("Database schema created successfully")

        except Exception as e:
            logger.error(f"Failed to create database schema: {e}")
            raise
