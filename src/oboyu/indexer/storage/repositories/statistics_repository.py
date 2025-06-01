"""Repository for statistics database operations."""

import logging
from pathlib import Path
from typing import Any, Dict, Union

from duckdb import DuckDBPyConnection

from oboyu.indexer.storage.queries.data_models import CollectionStatsData, DocumentStatsData
from oboyu.indexer.storage.queries.statistics_queries import StatisticsQueries

logger = logging.getLogger(__name__)


class StatisticsRepository:
    """Repository for statistics-related database operations."""

    def __init__(self, connection: DuckDBPyConnection) -> None:
        """Initialize statistics repository.

        Args:
            connection: DuckDB database connection

        """
        self.connection = connection
        self.queries = StatisticsQueries()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics.

        Returns:
            Dictionary with database statistics including:
            - chunk_count: Total number of chunks
            - embedding_count: Total number of embeddings
            - unique_paths: Number of unique file paths
            - language_distribution: Distribution of chunks by language
            - avg_chunk_size: Average chunk content size

        """
        try:
            stats = {}

            # Chunk statistics
            result = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
            stats["chunk_count"] = result[0] if result else 0

            # Embedding statistics
            result = self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            stats["embedding_count"] = result[0] if result else 0

            # Unique paths
            result = self.connection.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()
            stats["unique_paths"] = result[0] if result else 0

            # Language distribution
            language_results = self.connection.execute(
                """
                SELECT language, COUNT(*) as count
                FROM chunks
                GROUP BY language
                ORDER BY count DESC
            """
            ).fetchall()
            stats["language_distribution"] = {lang: count for lang, count in language_results}

            # Average chunk size
            result = self.connection.execute(
                """
                SELECT AVG(LENGTH(content)) as avg_size
                FROM chunks
            """
            ).fetchone()
            stats["avg_chunk_size"] = int(result[0]) if result and result[0] else 0

            # Model distribution
            model_results = self.connection.execute(
                """
                SELECT model, COUNT(*) as count
                FROM embeddings
                GROUP BY model
                ORDER BY count DESC
            """
            ).fetchall()
            stats["model_distribution"] = {model: count for model, count in model_results}

            return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

    def get_path_statistics(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get statistics for a specific file path.

        Args:
            path: File path to get statistics for

        Returns:
            Dictionary with path-specific statistics

        """
        try:
            stats = {}
            path_str = str(path)

            # Chunk count for path
            result = self.connection.execute(
                """
                SELECT COUNT(*) FROM chunks WHERE path = ?
            """,
                [path_str],
            ).fetchone()
            stats["chunk_count"] = result[0] if result else 0

            # Embedding count for path
            result = self.connection.execute(
                """
                SELECT COUNT(*)
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.id
                WHERE c.path = ?
            """,
                [path_str],
            ).fetchone()
            stats["embedding_count"] = result[0] if result else 0

            # Total content size
            result = self.connection.execute(
                """
                SELECT SUM(LENGTH(content)) as total_size
                FROM chunks
                WHERE path = ?
            """,
                [path_str],
            ).fetchone()
            stats["total_content_size"] = int(result[0]) if result and result[0] else 0

            # Language
            result = self.connection.execute(
                """
                SELECT DISTINCT language
                FROM chunks
                WHERE path = ?
            """,
                [path_str],
            ).fetchone()
            stats["language"] = result[0] if result else None

            return stats

        except Exception as e:
            logger.error(f"Failed to get path statistics: {e}")
            return {}

    def get_chunk_statistics(self) -> Dict[str, Any]:
        """Get detailed chunk statistics.

        Returns:
            Dictionary with chunk-related statistics

        """
        try:
            stats = {}

            # Total chunks
            result = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
            stats["total"] = result[0] if result else 0

            # Chunks by index position
            index_results = self.connection.execute(
                """
                SELECT chunk_index, COUNT(*) as count
                FROM chunks
                GROUP BY chunk_index
                ORDER BY chunk_index
                LIMIT 10
            """
            ).fetchall()
            stats["by_index"] = {idx: count for idx, count in index_results}

            # Content size distribution
            size_results = self.connection.execute(
                """
                SELECT
                    CASE
                        WHEN LENGTH(content) < 500 THEN 'small'
                        WHEN LENGTH(content) < 1000 THEN 'medium'
                        WHEN LENGTH(content) < 2000 THEN 'large'
                        ELSE 'very_large'
                    END as size_category,
                    COUNT(*) as count
                FROM chunks
                GROUP BY size_category
            """
            ).fetchall()
            stats["size_distribution"] = {category: count for category, count in size_results}

            return stats

        except Exception as e:
            logger.error(f"Failed to get chunk statistics: {e}")
            return {}

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get detailed embedding statistics.

        Returns:
            Dictionary with embedding-related statistics

        """
        try:
            stats = {}

            # Total embeddings
            result = self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            stats["total"] = result[0] if result else 0

            # Embeddings by model
            model_results = self.connection.execute(
                """
                SELECT model, COUNT(*) as count
                FROM embeddings
                GROUP BY model
            """
            ).fetchall()
            stats["by_model"] = {model: count for model, count in model_results}

            # Orphaned embeddings (chunks deleted but embeddings remain)
            result = self.connection.execute(
                """
                SELECT COUNT(*)
                FROM embeddings e
                LEFT JOIN chunks c ON e.chunk_id = c.id
                WHERE c.id IS NULL
            """
            ).fetchone()
            stats["orphaned"] = result[0] if result else 0

            # Coverage (chunks with embeddings)
            result = self.connection.execute(
                """
                SELECT
                    COUNT(DISTINCT c.id) as chunks_with_embeddings,
                    (SELECT COUNT(*) FROM chunks) as total_chunks
                FROM chunks c
                JOIN embeddings e ON c.id = e.chunk_id
            """
            ).fetchone()
            
            if result and result[0] is not None and result[1] is not None and result[1] > 0:
                stats["coverage"] = {
                    "chunks_with_embeddings": result[0],
                    "total_chunks": result[1],
                    "coverage_percentage": round((result[0] / result[1]) * 100, 2),
                }
            else:
                stats["coverage"] = {
                    "chunks_with_embeddings": 0,
                    "total_chunks": 0,
                    "coverage_percentage": 0.0,
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get embedding statistics: {e}")
            return {}

    def store_document_stats(self, stats_data: DocumentStatsData) -> None:
        """Store document statistics.

        Args:
            stats_data: Document statistics to store

        """
        try:
            sql, params = self.queries.upsert_document_stats(stats_data)
            self.connection.execute(sql, params)
        except Exception as e:
            logger.error(f"Failed to store document stats: {e}")

    def store_collection_stats(self, stats_data: CollectionStatsData) -> None:
        """Store collection statistics.

        Args:
            stats_data: Collection statistics to store

        """
        try:
            sql, params = self.queries.upsert_collection_stats(stats_data)
            self.connection.execute(sql, params)
        except Exception as e:
            logger.error(f"Failed to store collection stats: {e}")

    def get_latest_statistics_summary(self) -> Dict[str, Any]:
        """Get a summary of the latest statistics.

        Returns:
            Dictionary with summary statistics

        """
        try:
            # Get database stats
            db_stats = self.get_database_stats()
            
            # Get chunk stats
            chunk_stats = self.get_chunk_statistics()
            
            # Get embedding stats
            embedding_stats = self.get_embedding_statistics()

            # Combine into summary
            summary = {
                "database": {
                    "total_chunks": db_stats.get("chunk_count", 0),
                    "total_embeddings": db_stats.get("embedding_count", 0),
                    "unique_files": db_stats.get("unique_paths", 0),
                },
                "chunks": {
                    "size_distribution": chunk_stats.get("size_distribution", {}),
                    "average_size": db_stats.get("avg_chunk_size", 0),
                },
                "embeddings": {
                    "coverage_percentage": embedding_stats.get("coverage", {}).get("coverage_percentage", 0.0),
                    "models": list(db_stats.get("model_distribution", {}).keys()),
                },
                "languages": list(db_stats.get("language_distribution", {}).keys()),
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to get statistics summary: {e}")
            return {}
