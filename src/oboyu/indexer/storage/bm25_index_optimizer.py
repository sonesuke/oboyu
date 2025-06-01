"""BM25 index optimization and maintenance for database indexes."""

import logging
from typing import Dict

from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


class BM25IndexOptimizer:
    """Handles BM25 index optimization and maintenance operations."""

    def __init__(self, conn: DuckDBPyConnection) -> None:
        """Initialize BM25 index optimizer.
        
        Args:
            conn: Database connection

        """
        self.conn = conn

    def optimize_bm25_indexes(self) -> bool:
        """Optimize BM25 search indexes.

        Returns:
            True if optimization was successful

        """
        try:
            # Analyze tables to update statistics
            bm25_tables = ["vocabulary", "inverted_index", "document_stats", "collection_stats"]

            for table in bm25_tables:
                try:
                    self.conn.execute(f"ANALYZE {table}")
                    logger.debug(f"Analyzed table: {table}")
                except Exception as e:
                    logger.warning(f"Failed to analyze table {table}: {e}")

            logger.info("BM25 indexes optimized")
            return True

        except Exception as e:
            logger.error(f"Failed to optimize BM25 indexes: {e}")
            return False

    def get_bm25_table_stats(self) -> Dict[str, int]:
        """Get statistics for BM25-related tables.

        Returns:
            Dictionary with table statistics

        """
        stats = {}
        bm25_tables = ["vocabulary", "inverted_index", "document_stats", "collection_stats"]
        
        for table in bm25_tables:
            try:
                result = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats[f"{table}_count"] = result[0] if result else 0
            except Exception as e:
                logger.warning(f"Failed to get stats for table {table}: {e}")
                stats[f"{table}_count"] = 0
                
        return stats

    def vacuum_bm25_tables(self) -> bool:
        """Vacuum BM25 tables to reclaim space and optimize performance.

        Returns:
            True if vacuum was successful

        """
        try:
            bm25_tables = ["vocabulary", "inverted_index", "document_stats", "collection_stats"]
            
            for table in bm25_tables:
                try:
                    # DuckDB uses VACUUM for table maintenance
                    self.conn.execute(f"VACUUM {table}")
                    logger.debug(f"Vacuumed table: {table}")
                except Exception as e:
                    logger.warning(f"Failed to vacuum table {table}: {e}")

            logger.info("BM25 tables vacuumed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to vacuum BM25 tables: {e}")
            return False

    def rebuild_bm25_indexes(self) -> bool:
        """Rebuild BM25 indexes for optimal performance.

        Returns:
            True if rebuild was successful

        """
        try:
            # First optimize the indexes
            if not self.optimize_bm25_indexes():
                logger.warning("BM25 optimization failed during rebuild")
                return False

            # Then vacuum the tables
            if not self.vacuum_bm25_tables():
                logger.warning("BM25 vacuum failed during rebuild")
                return False

            logger.info("BM25 indexes rebuilt successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to rebuild BM25 indexes: {e}")
            return False
