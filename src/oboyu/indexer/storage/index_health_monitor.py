"""Index health monitoring and performance analysis for database indexes."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


@dataclass
class IndexInfo:
    """Information about a database index."""

    name: str
    table: str
    columns: List[str]
    index_type: str
    is_unique: bool
    is_valid: bool
    size_estimate: Optional[int] = None


@dataclass
class IndexHealth:
    """Health status of database indexes."""

    total_indexes: int
    healthy_indexes: int
    broken_indexes: List[str]
    warnings: List[str]
    recommendations: List[str]


class IndexHealthMonitor:
    """Monitors database index health and provides performance recommendations."""

    def __init__(self, conn: DuckDBPyConnection) -> None:
        """Initialize index health monitor.
        
        Args:
            conn: Database connection

        """
        self.conn = conn

    def get_index_info(self) -> List[IndexInfo]:
        """Get information about all database indexes.

        Returns:
            List of index information

        """
        try:
            # Query DuckDB system tables for index information
            result = self.conn.execute("""
                SELECT
                    index_name,
                    table_name,
                    sql,
                    is_unique
                FROM duckdb_indexes
                WHERE NOT index_name LIKE 'pg_%'  -- Exclude system indexes
                ORDER BY table_name, index_name
            """).fetchall()

            indexes = []
            for row in result:
                index_name = row[0]
                table_name = row[1]
                sql = row[2] or ""
                is_unique = bool(row[3])

                # Parse index type from SQL
                index_type = "btree"  # Default
                if "HNSW" in sql.upper():
                    index_type = "hnsw"
                elif "GIN" in sql.upper():
                    index_type = "gin"

                # Extract columns (simplified parsing)
                columns = [index_name.replace("idx_", "").replace(f"{table_name}_", "")]

                indexes.append(
                    IndexInfo(
                        name=index_name,
                        table=table_name,
                        columns=columns,
                        index_type=index_type,
                        is_unique=is_unique,
                        is_valid=True,  # Assume valid since DuckDB doesn't track broken indexes
                    )
                )

            return indexes

        except Exception as e:
            logger.error(f"Failed to get index information: {e}")
            return []

    def check_index_health(self, hnsw_index_exists_func: Callable[[], bool], should_create_hnsw_func: Callable[[], bool]) -> IndexHealth:
        """Check the health of all database indexes.

        Args:
            hnsw_index_exists_func: Function to check if HNSW index exists
            should_create_hnsw_func: Function to check if HNSW index should be created

        Returns:
            Index health status and recommendations

        """
        indexes = self.get_index_info()
        warnings: List[str] = []
        recommendations: List[str] = []
        broken_indexes: List[str] = []

        # Check HNSW index status
        if should_create_hnsw_func() and not hnsw_index_exists_func():
            warnings.append("HNSW index missing despite having embedding data")
            recommendations.append("Create HNSW index for vector search performance")

        # Check for missing indexes on large tables
        try:
            # Check chunks table size
            chunk_count_result = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            chunk_count = chunk_count_result[0] if chunk_count_result else 0

            if chunk_count > 10000:
                chunk_indexes = [idx for idx in indexes if idx.table == "chunks"]
                if len(chunk_indexes) < 2:  # Should have path and language indexes
                    warnings.append(f"Large chunks table ({chunk_count} rows) has few indexes")
                    recommendations.append("Consider adding indexes on frequently queried columns")

            # Check inverted index table size
            ii_count_result = self.conn.execute("SELECT COUNT(*) FROM inverted_index").fetchone()
            ii_count = ii_count_result[0] if ii_count_result else 0

            if ii_count > 100000:
                ii_indexes = [idx for idx in indexes if idx.table == "inverted_index"]
                if len(ii_indexes) < 2:  # Should have term and chunk_id indexes
                    warnings.append(f"Large inverted_index table ({ii_count} rows) may need more indexes")
                    recommendations.append("Ensure BM25 search indexes are properly created")

        except Exception as e:
            warnings.append(f"Failed to check table sizes: {e}")

        # Performance recommendations
        if len(indexes) > 20:
            recommendations.append("Consider removing unused indexes to improve write performance")

        return IndexHealth(
            total_indexes=len(indexes),
            healthy_indexes=len(indexes) - len(broken_indexes),
            broken_indexes=broken_indexes,
            warnings=warnings,
            recommendations=recommendations,
        )

    def get_index_usage_stats(self, hnsw_index_exists_func: Callable[[], bool], hnsw_params: Optional[object]) -> Dict[str, Any]:
        """Get index usage statistics.

        Args:
            hnsw_index_exists_func: Function to check if HNSW index exists
            hnsw_params: Current HNSW parameters

        Returns:
            Dictionary with index usage information

        Note: DuckDB doesn't provide detailed index usage statistics.
        This method provides basic information available.

        """
        stats: Dict[str, Any] = {
            "total_indexes": 0,
            "hnsw_index_exists": hnsw_index_exists_func(),
            "hnsw_params": hnsw_params.__dict__ if hnsw_params else None,
            "indexes": [],
        }

        try:
            indexes = self.get_index_info()
            stats["total_indexes"] = len(indexes)
            stats["indexes"] = [
                {
                    "name": idx.name,
                    "table": idx.table,
                    "type": idx.index_type,
                    "unique": idx.is_unique
                }
                for idx in indexes
            ]

        except Exception as e:
            logger.error(f"Failed to get index usage stats: {e}")

        return stats

    def get_performance_recommendations(self, hnsw_index_exists_func: Callable[[], bool], should_create_hnsw_func: Callable[[], bool]) -> List[str]:
        """Get performance recommendations for index optimization.

        Args:
            hnsw_index_exists_func: Function to check if HNSW index exists
            should_create_hnsw_func: Function to check if HNSW index should be created

        Returns:
            List of recommendation strings

        """
        recommendations = []
        health = self.check_index_health(hnsw_index_exists_func, should_create_hnsw_func)

        # Add health-based recommendations
        recommendations.extend(health.recommendations)

        # Add general performance recommendations
        try:
            # Check if we should compact HNSW index
            if hnsw_index_exists_func():
                embedding_count_result = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
                embedding_count = embedding_count_result[0] if embedding_count_result else 0

                if embedding_count > 50000:
                    recommendations.append("Consider compacting HNSW index for large embedding collections")

            # Check for table maintenance
            chunk_count_result = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            chunk_count = chunk_count_result[0] if chunk_count_result else 0
            if chunk_count > 100000:
                recommendations.append("Consider running VACUUM or ANALYZE for large tables")
        except Exception as e:
            logger.error(f"Failed to generate performance recommendations: {e}")
        
        return recommendations
