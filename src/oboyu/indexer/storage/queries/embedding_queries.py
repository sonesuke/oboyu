"""Embedding-related database query builders."""

import uuid
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .data_models import EmbeddingData


class EmbeddingQueries:
    """Query builders for embedding database operations."""

    @staticmethod
    def insert_embedding(embedding_data: EmbeddingData) -> Tuple[str, List[Any]]:
        """Build query to insert an embedding.

        Args:
            embedding_data: Embedding data to insert

        Returns:
            Tuple of (sql, parameters)

        """
        # Convert vector to list if it's a numpy array
        if isinstance(embedding_data.vector, np.ndarray):
            vector_list = embedding_data.vector.tolist()
        else:
            vector_list = embedding_data.vector

        sql = """
            INSERT INTO embeddings (id, chunk_id, model, vector, created_at)
            VALUES (?, ?, ?, ?, ?)
        """

        params = [embedding_data.id, embedding_data.chunk_id, embedding_data.model, vector_list, embedding_data.created_at]

        return sql.strip(), params

    @staticmethod
    def upsert_embedding(embedding_data: EmbeddingData) -> Tuple[str, List[Any]]:
        """Build query to insert or update an embedding.

        Args:
            embedding_data: Embedding data to upsert

        Returns:
            Tuple of (sql, parameters)

        """
        if isinstance(embedding_data.vector, np.ndarray):
            vector_list = embedding_data.vector.tolist()
        else:
            vector_list = embedding_data.vector

        sql = """
            INSERT INTO embeddings (id, chunk_id, model, vector, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                chunk_id = excluded.chunk_id,
                model = excluded.model,
                vector = excluded.vector,
                created_at = excluded.created_at
        """

        params = [embedding_data.id, embedding_data.chunk_id, embedding_data.model, vector_list, embedding_data.created_at]

        return sql.strip(), params

    @staticmethod
    def search_by_vector(
        query_vector: Union[List[float], NDArray[np.float32]],
        limit: int = 10,
        language: Optional[str] = None,
        embedding_dimensions: int = 256
    ) -> Tuple[str, List[Any]]:
        """Build query for vector similarity search.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            language: Optional language filter
            embedding_dimensions: Vector dimensions for temporary table

        Returns:
            Tuple of (sql, parameters)

        """
        # Convert vector to list if needed
        if isinstance(query_vector, np.ndarray):
            vector_list = query_vector.tolist()
        else:
            vector_list = query_vector

        # Generate unique temporary table name
        f"temp_query_vector_{uuid.uuid4().hex}"

        # Build query with temporary table approach
        sql = f"""
            WITH query_vector AS (
                SELECT ?::{f"FLOAT[{embedding_dimensions}]"} as vec
            )
            SELECT
                c.id as chunk_id,
                c.path,
                c.title,
                c.content,
                c.chunk_index,
                c.language,
                c.metadata,
                array_distance(e.vector, q.vec) as score
            FROM chunks c
            JOIN embeddings e ON c.id = e.chunk_id
            CROSS JOIN query_vector q
        """

        params = [vector_list]

        # Add language filter if specified
        if language:
            sql += " WHERE c.language = ?"
            params.append(language)

        # Add ordering and limit
        sql += f" ORDER BY score ASC LIMIT {limit}"

        return sql.strip(), params
