"""Schema initialization for database setup."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from .schema_validator import SchemaValidator

logger = logging.getLogger(__name__)


class SchemaInitializer:
    """Handles database schema initialization with concurrency protection."""

    def __init__(self, db_path: Path, embedding_dimensions: int) -> None:
        """Initialize the schema initializer.
        
        Args:
            db_path: Path to the database file
            embedding_dimensions: Number of embedding dimensions

        """
        self.db_path = db_path
        self.embedding_dimensions = embedding_dimensions

    def initialize_fresh_schema(self, conn: "DuckDBPyConnection", schema_validator: "SchemaValidator") -> None:
        """Initialize a fresh database schema with concurrency protection.
        
        Args:
            conn: Database connection
            schema_validator: Schema validator instance for integrity checks
            
        Raises:
            RuntimeError: If schema initialization fails

        """
        from oboyu.indexer.storage.database_lock import DatabaseLock
        
        # Use a lock to prevent concurrent schema initialization
        lock = DatabaseLock(self.db_path, "schema_init")
        
        try:
            with lock.acquire(timeout=30.0):
                logger.debug("Acquired lock for schema initialization")
                
                # Check if schema was already initialized by another process
                if schema_validator.validate_schema_integrity(conn):
                    logger.debug("Schema already initialized by another process")
                    return
                
                # Proceed with schema initialization
                from oboyu.indexer.storage.migrations import MigrationManager
                from oboyu.indexer.storage.schema import DatabaseSchema
                
                # Create schema and migration manager
                schema = DatabaseSchema(self.embedding_dimensions)
                migration_manager = MigrationManager(conn, schema)
                
                # Create tables with individual transactions to avoid write conflicts
                tables = schema.get_all_tables()
                for table in tables:
                    try:
                        # Use individual transactions for each table to minimize lock time
                        conn.execute("BEGIN TRANSACTION")
                        conn.execute(table.sql)
                        
                        # Create indexes for this table
                        for index_sql in table.indexes:
                            conn.execute(index_sql)
                        
                        conn.execute("COMMIT")
                        logger.debug(f"Created table {table.name}")
                        
                    except Exception as table_error:
                        try:
                            conn.execute("ROLLBACK")
                        except Exception as rollback_error:
                            logger.debug(f"Rollback failed: {rollback_error}")
                        
                        # Check if the error is because table already exists
                        error_msg = str(table_error).lower()
                        if "already exists" in error_msg or "relation" in error_msg:
                            logger.debug(f"Table {table.name} already exists, skipping")
                            continue
                        else:
                            logger.error(f"Failed to create table {table.name}: {table_error}")
                            raise
                
                # Run migrations
                try:
                    migration_manager.run_migrations()
                except Exception as migration_error:
                    # Migrations might fail if already run, which is acceptable
                    logger.debug(f"Migration warning: {migration_error}")
                
                logger.info("Fresh schema initialized successfully")
                
        except TimeoutError as e:
            logger.error(f"Could not acquire lock for schema initialization: {e}")
            # If we can't get the lock, check if schema is already valid
            if schema_validator.validate_schema_integrity(conn):
                logger.info("Schema already initialized, proceeding without lock")
                return
            raise RuntimeError("Schema initialization timed out and schema is not valid")
        except Exception as e:
            logger.error(f"Failed to initialize fresh schema: {e}")
            raise RuntimeError(f"Schema initialization failed: {e}") from e
