"""Index validation and rebuilding for database integrity."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from oboyu.indexer.models.database import HNSWIndexParams

logger = logging.getLogger(__name__)


class IndexValidator:
    """Validates and rebuilds database indexes."""

    def __init__(self, embedding_dimensions: int, hnsw_params: "HNSWIndexParams") -> None:
        """Initialize the index validator.
        
        Args:
            embedding_dimensions: Number of embedding dimensions
            hnsw_params: HNSW index parameters

        """
        self.embedding_dimensions = embedding_dimensions
        self.hnsw_params = hnsw_params

    def validate_index_integrity(self, conn: "DuckDBPyConnection") -> bool:
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

    def rebuild_indexes(self, conn: "DuckDBPyConnection") -> None:
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
