"""Facade for coordinating specialized index management components.

This module provides a unified interface for database index management by
coordinating specialized components following the single responsibility principle.

Components:
- HNSWIndexManager: Handles HNSW vector index operations
- IndexHealthMonitor: Monitors index health and provides recommendations
- BM25IndexOptimizer: Optimizes BM25 search indexes
"""

import logging
from typing import Any, Dict, List, Optional

from duckdb import DuckDBPyConnection

from oboyu.indexer.storage.bm25_index_optimizer import BM25IndexOptimizer
from oboyu.indexer.storage.hnsw_index_manager import HNSWIndexManager, HNSWIndexParams
from oboyu.indexer.storage.index_health_monitor import IndexHealth, IndexHealthMonitor, IndexInfo
from oboyu.indexer.storage.schema import DatabaseSchema

logger = logging.getLogger(__name__)

# Export public API
__all__ = [
    "IndexManager",
    "HNSWIndexParams",
    "IndexHealth",
    "IndexInfo"
]


class IndexManager:
    """Facade coordinating specialized index management components.

    This class delegates index management responsibilities to specialized
    components while providing a unified interface for backward compatibility.
    """

    def __init__(self, conn: DuckDBPyConnection, schema: DatabaseSchema) -> None:
        """Initialize index manager facade.

        Args:
            conn: Database connection
            schema: Database schema manager

        """
        self.conn = conn
        self.schema = schema
        self._hnsw_params: Optional[HNSWIndexParams] = None
        
        # Initialize specialized components
        self.hnsw_manager = HNSWIndexManager(conn, schema)
        self.health_monitor = IndexHealthMonitor(conn)
        self.bm25_optimizer = BM25IndexOptimizer(conn)

    def setup_all_indexes(self, hnsw_params: Optional[HNSWIndexParams] = None) -> None:
        """Set up all database indexes.

        Args:
            hnsw_params: Optional HNSW index parameters

        """
        logger.debug("Setting up database indexes...")

        # Store HNSW parameters for later use
        if hnsw_params:
            self._hnsw_params = hnsw_params

        # Create standard table indexes
        self._create_table_indexes()

        # Create HNSW index if embeddings table has data
        if self._should_create_hnsw_index():
            success = self.create_hnsw_index(hnsw_params or HNSWIndexParams())
            if not success:
                logger.warning("HNSW index creation failed, but continuing with setup")
        else:
            logger.debug("Skipping HNSW index creation - no embeddings data yet")

        logger.info("Database indexes setup completed")

    def _create_table_indexes(self) -> None:
        """Create all standard table indexes."""
        tables = self.schema.get_all_tables()

        for table in tables:
            for index_sql in table.indexes:
                try:
                    self.conn.execute(index_sql)
                    logger.debug(f"Created index: {index_sql.split()[:3]}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

    def create_hnsw_index(self, params: HNSWIndexParams, force: bool = False) -> bool:
        """Create HNSW vector similarity index.

        Args:
            params: HNSW index parameters
            force: Whether to recreate if index exists

        Returns:
            True if index was created successfully

        """
        success = self.hnsw_manager.create_hnsw_index(params, force)
        if success:
            self._hnsw_params = params
        return success

    def drop_hnsw_index(self) -> bool:
        """Drop the HNSW vector index.

        Returns:
            True if index was dropped successfully

        """
        return self.hnsw_manager.drop_hnsw_index()

    def hnsw_index_exists(self) -> bool:
        """Check if HNSW index exists.

        Returns:
            True if HNSW index exists

        """
        return self.hnsw_manager.hnsw_index_exists()

    def _should_create_hnsw_index(self) -> bool:
        """Check if HNSW index should be created.

        Returns:
            True if embeddings table has data

        """
        return self.hnsw_manager.should_create_hnsw_index()

    def recreate_hnsw_index(self, params: Optional[HNSWIndexParams] = None) -> bool:
        """Recreate the HNSW index.

        Args:
            params: Optional new parameters (uses existing if not provided)

        Returns:
            True if recreation was successful

        """
        # Use existing parameters if not provided
        if params is None:
            params = self._hnsw_params or HNSWIndexParams()
        
        success = self.hnsw_manager.recreate_hnsw_index(params)
        if success:
            self._hnsw_params = params
        return success

    def validate_hnsw_index_params(self, expected_params: HNSWIndexParams) -> bool:
        """Validate that existing HNSW index has expected parameters.

        Args:
            expected_params: Expected index parameters

        Returns:
            True if parameters match or validation is not possible

        """
        is_valid = self.hnsw_manager.validate_hnsw_index_params(expected_params)
        if is_valid:
            self._hnsw_params = expected_params
        return is_valid

    def compact_hnsw_index(self) -> bool:
        """Compact the HNSW index to improve search performance.

        Returns:
            True if compaction was successful

        """
        return self.hnsw_manager.compact_hnsw_index()

    def optimize_bm25_indexes(self) -> bool:
        """Optimize BM25 search indexes.

        Returns:
            True if optimization was successful

        """
        return self.bm25_optimizer.optimize_bm25_indexes()

    def get_index_info(self) -> List[IndexInfo]:
        """Get information about all database indexes.

        Returns:
            List of index information

        """
        return self.health_monitor.get_index_info()

    def check_index_health(self) -> IndexHealth:
        """Check the health of all database indexes.

        Returns:
            Index health status and recommendations

        """
        return self.health_monitor.check_index_health(
            self.hnsw_index_exists,
            self._should_create_hnsw_index
        )

    def rebuild_all_indexes(self) -> bool:
        """Rebuild all database indexes.

        Returns:
            True if all indexes were rebuilt successfully

        """
        logger.info("Rebuilding all database indexes...")

        try:
            # Drop and recreate HNSW index
            if self.hnsw_index_exists():
                hnsw_success = self.recreate_hnsw_index()
                if not hnsw_success:
                    logger.error("Failed to rebuild HNSW index")
                    return False

            # Rebuild standard indexes
            self._create_table_indexes()

            # Optimize BM25 indexes
            self.bm25_optimizer.rebuild_bm25_indexes()

            logger.info("All indexes rebuilt successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to rebuild indexes: {e}")
            return False

    def get_index_usage_stats(self) -> Dict[str, Any]:
        """Get index usage statistics.

        Returns:
            Dictionary with index usage information

        """
        return self.health_monitor.get_index_usage_stats(
            self.hnsw_index_exists,
            self._hnsw_params
        )

    def get_performance_recommendations(self) -> List[str]:
        """Get performance recommendations for index optimization.

        Returns:
            List of recommendation strings

        """
        return self.health_monitor.get_performance_recommendations(
            self.hnsw_index_exists,
            self._should_create_hnsw_index
        )
