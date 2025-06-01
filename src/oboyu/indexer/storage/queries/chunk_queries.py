"""Chunk-related database query builders."""

import json
from typing import Any, Dict, List, Tuple

from oboyu.indexer.core.document_processor import Chunk

from ..utils import DateTimeEncoder
from .data_models import ChunkData


class ChunkQueries:
    """Query builders for chunk database operations."""

    @staticmethod
    def insert_chunk(chunk_data: ChunkData) -> Tuple[str, List[Any]]:
        """Build query to insert a chunk.

        Args:
            chunk_data: Chunk data to insert

        Returns:
            Tuple of (sql, parameters)

        """
        # Convert metadata to JSON string if present
        metadata_json = json.dumps(chunk_data.metadata, cls=DateTimeEncoder) if chunk_data.metadata else None

        sql = """
            INSERT INTO chunks (id, path, title, content, chunk_index, language, created_at, modified_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = [
            chunk_data.id,
            chunk_data.path,
            chunk_data.title,
            chunk_data.content,
            chunk_data.chunk_index,
            chunk_data.language,
            chunk_data.created_at,
            chunk_data.modified_at,
            metadata_json,
        ]

        return sql.strip(), params

    @staticmethod
    def upsert_chunk(chunk_data: ChunkData) -> Tuple[str, List[Any]]:
        """Build query to insert or update a chunk.

        Args:
            chunk_data: Chunk data to upsert

        Returns:
            Tuple of (sql, parameters)

        """
        metadata_json = json.dumps(chunk_data.metadata, cls=DateTimeEncoder) if chunk_data.metadata else None

        sql = """
            INSERT INTO chunks (id, path, title, content, chunk_index, language, created_at, modified_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                path = excluded.path,
                title = excluded.title,
                content = excluded.content,
                chunk_index = excluded.chunk_index,
                language = excluded.language,
                modified_at = excluded.modified_at,
                metadata = excluded.metadata
        """

        params = [
            chunk_data.id,
            chunk_data.path,
            chunk_data.title,
            chunk_data.content,
            chunk_data.chunk_index,
            chunk_data.language,
            chunk_data.created_at,
            chunk_data.modified_at,
            metadata_json,
        ]

        return sql.strip(), params

    @staticmethod
    def select_chunk_by_id(chunk_id: str) -> Tuple[str, List[Any]]:
        """Build query to select a chunk by ID.

        Args:
            chunk_id: ID of chunk to retrieve

        Returns:
            Tuple of (sql, parameters)

        """
        sql = """
            SELECT id, path, title, content, chunk_index, language, created_at, modified_at, metadata
            FROM chunks
            WHERE id = ?
        """

        return sql.strip(), [chunk_id]

    @staticmethod
    def select_chunks_by_path(path: str) -> Tuple[str, List[Any]]:
        """Build query to select chunks by document path.

        Args:
            path: Document path

        Returns:
            Tuple of (sql, parameters)

        """
        sql = """
            SELECT id, path, title, content, chunk_index, language, created_at, modified_at, metadata
            FROM chunks
            WHERE path = ?
            ORDER BY chunk_index
        """

        return sql.strip(), [path]

    @staticmethod
    def delete_chunks_by_path(path: str) -> Tuple[str, List[Any]]:
        """Build query to delete chunks by path.

        Args:
            path: Document path

        Returns:
            Tuple of (sql, parameters)

        """
        sql = "DELETE FROM chunks WHERE path = ?"
        return sql, [path]

    @staticmethod
    def chunk_from_row(row: Tuple[Any, ...]) -> Dict[str, Any]:
        """Convert database row to chunk dictionary.

        Args:
            row: Database row tuple

        Returns:
            Chunk dictionary with proper type conversion

        """
        return {
            "id": str(row[0]),
            "path": str(row[1]),
            "title": str(row[2]),
            "content": str(row[3]),
            "chunk_index": int(row[4]),
            "language": str(row[5]) if row[5] else None,
            "created_at": row[6],
            "modified_at": row[7],
            "metadata": json.loads(row[8]) if row[8] else {},
        }

    @staticmethod
    def search_result_from_row(row: Tuple[Any, ...]) -> Dict[str, Any]:
        """Convert database search result row to result dictionary.

        Args:
            row: Database row tuple from search query

        Returns:
            Search result dictionary with proper type conversion

        """
        return {
            "chunk_id": str(row[0]),
            "path": str(row[1]),
            "title": str(row[2]),
            "content": str(row[3]),
            "chunk_index": int(row[4]),
            "language": str(row[5]) if row[5] else None,
            "metadata": json.loads(row[6]) if row[6] else {},
            "score": float(row[7]),
        }

    @staticmethod
    def from_chunk_to_chunk_data(chunk: Chunk) -> ChunkData:
        """Convert Chunk object to ChunkData.

        Args:
            chunk: Source chunk object

        Returns:
            ChunkData object for database operations

        """
        return ChunkData(
            id=chunk.id,
            path=str(chunk.path),
            title=chunk.title,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            language=chunk.language,
            created_at=chunk.created_at,
            modified_at=chunk.modified_at,
            metadata=chunk.metadata,
        )
