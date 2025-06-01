"""Repository for chunk database operations."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from duckdb import DuckDBPyConnection

from oboyu.indexer.core.document_processor import Chunk
from oboyu.indexer.storage.queries.chunk_queries import ChunkQueries
from oboyu.indexer.storage.queries.data_models import ChunkData

logger = logging.getLogger(__name__)


class ChunkRepository:
    """Repository for chunk-related database operations."""

    def __init__(self, connection: DuckDBPyConnection) -> None:
        """Initialize chunk repository.

        Args:
            connection: DuckDB database connection

        """
        self.connection = connection
        self.queries = ChunkQueries()

    def store_chunks(
        self,
        chunks: List[Chunk],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Store document chunks in the database.

        Args:
            chunks: List of document chunks to store
            progress_callback: Optional progress callback

        """
        if not chunks:
            return

        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            # Convert chunk to database format
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

            # Get query and parameters
            sql, params = self.queries.upsert_chunk(chunk_data)

            # Execute query
            self.connection.execute(sql, params)

            # Report progress
            if progress_callback and (i % 100 == 0 or i == total_chunks - 1):
                progress_callback("storing", i + 1, total_chunks)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk data or None if not found

        """
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
                # Convert to dictionary
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

                # Parse metadata if it exists
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
        """Delete all chunks for a specific file path.

        Args:
            path: File path to delete chunks for

        Returns:
            Number of chunks deleted

        """
        try:
            # Delete chunks and get count
            result = self.connection.execute(
                """
                DELETE FROM chunks WHERE path = ?
            """,
                [str(path)],
            )

            return result.rowcount if hasattr(result, "rowcount") else 0

        except Exception as e:
            logger.error(f"Failed to delete chunks by path: {e}")
            return 0

    def get_chunk_count(self) -> int:
        """Get total number of chunks in the database.

        Returns:
            Total chunk count

        """
        try:
            result = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get chunk count: {e}")
            return 0

    def get_paths_with_chunks(self) -> List[str]:
        """Get list of file paths that have chunks in the database.

        Returns:
            List of unique file paths

        """
        try:
            results = self.connection.execute("SELECT DISTINCT path FROM chunks").fetchall()
            return [result[0] for result in results]
        except Exception as e:
            logger.error(f"Failed to get paths with chunks: {e}")
            return []

    def get_chunks_by_path(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Get all chunks for a specific file path.

        Args:
            path: File path to get chunks for

        Returns:
            List of chunk data dictionaries

        """
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
                # Convert to dictionary
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

                # Parse metadata if it exists
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
        """Get chunk IDs that don't have embeddings.

        Args:
            limit: Optional limit on number of IDs to return

        Returns:
            List of chunk IDs without embeddings

        """
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
