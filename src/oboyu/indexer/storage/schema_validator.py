"""Schema validation for database integrity checks."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates database schema integrity and completeness."""

    def validate_schema_integrity(self, conn: "DuckDBPyConnection") -> bool:
        """Validate that database schema is complete and valid with concurrent access protection.
        
        Args:
            conn: Database connection
            
        Returns:
            True if schema is valid, False otherwise

        """
        try:
            # Check if required tables exist with retry logic for concurrent access
            required_tables = ['chunks', 'embeddings', 'file_metadata', 'bm25_index']
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    missing_tables = []
                    
                    for table in required_tables:
                        result = conn.execute(
                            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                            [table]
                        ).fetchone()
                        
                        if not result or result[0] == 0:
                            missing_tables.append(table)
                    
                    if missing_tables:
                        logger.debug(f"Required tables not found: {missing_tables}")
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
                    
                    # Additional validation: test basic operations on each table
                    for table in required_tables:
                        try:
                            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                        except Exception as table_error:
                            logger.debug(f"Table {table} not accessible: {table_error}")
                            return False
                    
                    logger.debug("Schema integrity validation passed")
                    return True
                    
                except Exception as validation_error:
                    error_msg = str(validation_error).lower()
                    
                    # Handle catalog conflicts during validation
                    if "catalog" in error_msg and "conflict" in error_msg:
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(0.1 * (attempt + 1))
                            logger.debug(f"Retrying schema validation (attempt {attempt + 2}/{max_retries})")
                            continue
                        else:
                            logger.debug(f"Schema validation failed after retries: {validation_error}")
                            return False
                    else:
                        # Non-conflict error, fail immediately
                        logger.debug(f"Schema validation failed: {validation_error}")
                        return False
            
            # Should not reach here
            return False
            
        except Exception as e:
            logger.debug(f"Schema validation failed: {e}")
            return False
