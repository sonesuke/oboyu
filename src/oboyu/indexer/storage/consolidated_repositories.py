"""Consolidated repository classes for database operations."""
# pylint: disable=too-many-lines

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from duckdb import DuckDBPyConnection
from numpy.typing import NDArray

from oboyu.common.types import Chunk

from .consolidated_queries import (
    ChunkData,
    ChunkQueries,
    CollectionStatsData,
    DocumentStatsData,
    EmbeddingData,
    EmbeddingQueries,
    StatisticsQueries,
)

logger = logging.getLogger(__name__)


class ChunkRepository:
    """Repository for chunk-related database operations."""

    def __init__(self, connection: DuckDBPyConnection) -> None:
        """Initialize chunk repository."""
        self.connection = connection
        self.queries = ChunkQueries()

    def store_chunks(
        self,
        chunks: List[Chunk],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Store document chunks in the database."""
        if not chunks:
            return

        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_data = ChunkData(
                id=chunk.id,
                path=str(chunk.path),
                title=chunk.title,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                language=chunk.language,
                created_at=chunk.created_at or datetime.now(),
                modified_at=chunk.modified_at or datetime.now(),
                metadata=chunk.metadata or {},
            )

            sql, params = self.queries.upsert_chunk(chunk_data)
            self.connection.execute(sql, params)

            if progress_callback and (i % 100 == 0 or i == total_chunks - 1):
                progress_callback("storing", i + 1, total_chunks)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID."""
        try:
            result = self.connection.execute(
                """
                SELECT c.*, e.vector as embedding
                FROM chunks c
                LEFT JOIN embeddings e ON c.id = e.chunk_id
                WHERE c.id = ?
            """,
                [chunk_id],
            ).fetchone()

            if result:
                columns = [
                    "id",
                    "path",
                    "title",
                    "content",
                    "chunk_index",
                    "language",
                    "created_at",
                    "modified_at",
                    "metadata",
                    "embedding",
                ]
                chunk_data = dict(zip(columns, result))

                if chunk_data.get("metadata"):
                    try:
                        chunk_data["metadata"] = json.loads(chunk_data["metadata"])
                    except json.JSONDecodeError:
                        chunk_data["metadata"] = {}

                return chunk_data

            return None

        except Exception as e:
            logger.error(f"Failed to get chunk by ID: {e}")
            return None

    def delete_chunks_by_path(self, path: Union[str, Path]) -> int:
        """Delete all chunks for a specific file path."""
        try:
            result = self.connection.execute(
                "DELETE FROM chunks WHERE path = ?",
                [str(path)],
            )

            return result.rowcount if hasattr(result, "rowcount") else 0

        except Exception as e:
            logger.error(f"Failed to delete chunks by path: {e}")
            return 0

    def get_chunk_count(self) -> int:
        """Get total number of chunks in the database."""
        try:
            result = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get chunk count: {e}")
            return 0

    def get_paths_with_chunks(self) -> List[str]:
        """Get list of file paths that have chunks in the database."""
        try:
            results = self.connection.execute("SELECT DISTINCT path FROM chunks").fetchall()
            return [result[0] for result in results]
        except Exception as e:
            logger.error(f"Failed to get paths with chunks: {e}")
            return []

    def get_chunks_by_path(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Get all chunks for a specific file path."""
        try:
            results = self.connection.execute(
                """
                SELECT c.*, e.vector as embedding
                FROM chunks c
                LEFT JOIN embeddings e ON c.id = e.chunk_id
                WHERE c.path = ?
                ORDER BY c.chunk_index
            """,
                [str(path)],
            ).fetchall()

            chunks = []
            for result in results:
                columns = [
                    "id",
                    "path",
                    "title",
                    "content",
                    "chunk_index",
                    "language",
                    "created_at",
                    "modified_at",
                    "metadata",
                    "embedding",
                ]
                chunk_data = dict(zip(columns, result))

                if chunk_data.get("metadata"):
                    try:
                        chunk_data["metadata"] = json.loads(chunk_data["metadata"])
                    except json.JSONDecodeError:
                        chunk_data["metadata"] = {}

                chunks.append(chunk_data)

            return chunks

        except Exception as e:
            logger.error(f"Failed to get chunks by path: {e}")
            return []

    def clear_all_chunks(self) -> None:
        """Clear all chunks from the database."""
        try:
            self.connection.execute("DELETE FROM chunks")
        except Exception as e:
            logger.error(f"Failed to clear chunks: {e}")

    def get_chunk_ids_without_embeddings(self, limit: Optional[int] = None) -> List[str]:
        """Get chunk IDs that don't have embeddings."""
        try:
            query = """
                SELECT c.id
                FROM chunks c
                LEFT JOIN embeddings e ON c.id = e.chunk_id
                WHERE e.id IS NULL
            """
            
            if limit:
                query += f" LIMIT {limit}"

            results = self.connection.execute(query).fetchall()
            return [result[0] for result in results]

        except Exception as e:
            logger.error(f"Failed to get chunks without embeddings: {e}")
            return []


class EmbeddingRepository:
    """Repository for embedding-related database operations."""

    def __init__(self, connection: DuckDBPyConnection) -> None:
        """Initialize embedding repository."""
        self.connection = connection
        self.queries = EmbeddingQueries()

    def store_embeddings(
        self,
        chunk_ids: List[str],
        embeddings: List[NDArray[np.float32]],
        model_name: str = "cl-nagoya/ruri-v3-30m",
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Store embedding vectors in the database."""
        if len(chunk_ids) != len(embeddings):
            raise ValueError("Number of chunk IDs must match number of embeddings")

        total_embeddings = len(chunk_ids)

        for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings)):
            embedding_data = EmbeddingData(
                id=str(uuid.uuid4()),
                chunk_id=chunk_id,
                model=model_name,
                vector=embedding.astype(np.float32),
                created_at=datetime.now(),
            )

            sql, params = self.queries.upsert_embedding(embedding_data)
            self.connection.execute(sql, params)

            if progress_callback and (i % 100 == 0 or i == total_embeddings - 1):
                progress_callback("storing_embeddings", i + 1, total_embeddings)

    def get_embedding_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get embedding for a specific chunk."""
        try:
            result = self.connection.execute(
                """
                SELECT id, chunk_id, model, vector, created_at
                FROM embeddings
                WHERE chunk_id = ?
            """,
                [chunk_id],
            ).fetchone()

            if result:
                columns = ["id", "chunk_id", "model", "vector", "created_at"]
                embedding_data = dict(zip(columns, result))

                if embedding_data.get("vector"):
                    embedding_data["vector"] = np.array(embedding_data["vector"], dtype=np.float32)

                return embedding_data

            return None

        except Exception as e:
            logger.error(f"Failed to get embedding by chunk ID: {e}")
            return None

    def get_embeddings_batch(self, chunk_ids: List[str]) -> Dict[str, NDArray[np.float32]]:
        """Get embeddings for multiple chunks."""
        try:
            if not chunk_ids:
                return {}

            placeholders = ",".join(["?" for _ in chunk_ids])
            results = self.connection.execute(
                f"""
                SELECT chunk_id, vector
                FROM embeddings
                WHERE chunk_id IN ({placeholders})
            """,
                chunk_ids,
            ).fetchall()

            embeddings = {}
            for chunk_id, vector in results:
                if vector:
                    embeddings[chunk_id] = np.array(vector, dtype=np.float32)

            return embeddings

        except Exception as e:
            logger.error(f"Failed to get embeddings batch: {e}")
            return {}

    def delete_embeddings_by_chunk_ids(self, chunk_ids: List[str]) -> int:
        """Delete embeddings for specific chunks."""
        try:
            if not chunk_ids:
                return 0

            placeholders = ",".join(["?" for _ in chunk_ids])
            result = self.connection.execute(
                f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )

            return result.rowcount if hasattr(result, "rowcount") else 0

        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            return 0

    def delete_embeddings_by_path(self, path: str) -> int:
        """Delete embeddings for all chunks of a specific file."""
        try:
            result = self.connection.execute(
                """
                DELETE FROM embeddings
                WHERE chunk_id IN (SELECT id FROM chunks WHERE path = ?)
            """,
                [path],
            )

            return result.rowcount if hasattr(result, "rowcount") else 0

        except Exception as e:
            logger.error(f"Failed to delete embeddings by path: {e}")
            return 0

    def get_embedding_count(self) -> int:
        """Get total number of embeddings in the database."""
        try:
            result = self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get embedding count: {e}")
            return 0

    def clear_all_embeddings(self) -> None:
        """Clear all embeddings from the database."""
        try:
            self.connection.execute("DELETE FROM embeddings")
        except Exception as e:
            logger.error(f"Failed to clear embeddings: {e}")

    def get_embeddings_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all embeddings for a specific model."""
        try:
            results = self.connection.execute(
                """
                SELECT id, chunk_id, model, created_at
                FROM embeddings
                WHERE model = ?
            """,
                [model_name],
            ).fetchall()

            embeddings = []
            for result in results:
                columns = ["id", "chunk_id", "model", "created_at"]
                embedding_data = dict(zip(columns, result))
                embeddings.append(embedding_data)

            return embeddings

        except Exception as e:
            logger.error(f"Failed to get embeddings by model: {e}")
            return []

    def update_embedding_model(self, chunk_id: str, new_model_name: str) -> bool:
        """Update the model name for an embedding."""
        try:
            result = self.connection.execute(
                "UPDATE embeddings SET model = ? WHERE chunk_id = ?",
                [new_model_name, chunk_id],
            )

            return result.rowcount > 0 if hasattr(result, "rowcount") else False

        except Exception as e:
            logger.error(f"Failed to update embedding model: {e}")
            return False


class StatisticsRepository:
    """Repository for statistics-related database operations."""

    def __init__(self, connection: DuckDBPyConnection) -> None:
        """Initialize statistics repository."""
        self.connection = connection
        self.queries = StatisticsQueries()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            stats = {}

            result = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
            stats["chunk_count"] = result[0] if result else 0

            result = self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            stats["embedding_count"] = result[0] if result else 0

            result = self.connection.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()
            stats["unique_paths"] = result[0] if result else 0

            language_results = self.connection.execute(
                """
                SELECT language, COUNT(*) as count
                FROM chunks
                GROUP BY language
                ORDER BY count DESC
            """
            ).fetchall()
            stats["language_distribution"] = {lang: count for lang, count in language_results}

            result = self.connection.execute(
                "SELECT AVG(LENGTH(content)) as avg_size FROM chunks"
            ).fetchone()
            stats["avg_chunk_size"] = int(result[0]) if result and result[0] else 0

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
        """Get statistics for a specific file path."""
        try:
            stats = {}
            path_str = str(path)

            result = self.connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE path = ?",
                [path_str],
            ).fetchone()
            stats["chunk_count"] = result[0] if result else 0

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

            result = self.connection.execute(
                """
                SELECT SUM(LENGTH(content)) as total_size
                FROM chunks
                WHERE path = ?
            """,
                [path_str],
            ).fetchone()
            stats["total_content_size"] = int(result[0]) if result and result[0] else 0

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
        """Get detailed chunk statistics."""
        try:
            stats = {}

            result = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
            stats["total"] = result[0] if result else 0

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
        """Get detailed embedding statistics."""
        try:
            stats = {}

            result = self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            stats["total"] = result[0] if result else 0

            model_results = self.connection.execute(
                """
                SELECT model, COUNT(*) as count
                FROM embeddings
                GROUP BY model
            """
            ).fetchall()
            stats["by_model"] = {model: count for model, count in model_results}

            result = self.connection.execute(
                """
                SELECT COUNT(*)
                FROM embeddings e
                LEFT JOIN chunks c ON e.chunk_id = c.id
                WHERE c.id IS NULL
            """
            ).fetchone()
            stats["orphaned"] = result[0] if result else 0

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
        """Store document statistics."""
        try:
            sql, params = self.queries.upsert_document_stats(stats_data)
            self.connection.execute(sql, params)
        except Exception as e:
            logger.error(f"Failed to store document stats: {e}")

    def store_collection_stats(self, stats_data: CollectionStatsData) -> None:
        """Store collection statistics."""
        try:
            sql, params = self.queries.upsert_collection_stats(stats_data)
            self.connection.execute(sql, params)
        except Exception as e:
            logger.error(f"Failed to store collection stats: {e}")

    def get_latest_statistics_summary(self) -> Dict[str, Any]:
        """Get a summary of the latest statistics."""
        try:
            db_stats = self.get_database_stats()
            chunk_stats = self.get_chunk_statistics()
            embedding_stats = self.get_embedding_statistics()

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


# Export all repository classes
__all__ = ["ChunkRepository", "EmbeddingRepository", "StatisticsRepository"]
