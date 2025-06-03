"""Database search operations for vector and text search."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from duckdb import DuckDBPyConnection
from numpy.typing import NDArray

from oboyu.common.types import SearchFilters

logger = logging.getLogger(__name__)


class DatabaseSearchService:
    """Handles database search operations for vector and text queries."""

    def __init__(self, conn: DuckDBPyConnection) -> None:
        """Initialize database search service.
        
        Args:
            conn: Database connection

        """
        self.conn = conn

    def vector_search(
        self,
        query_vector: NDArray[np.float32],
        limit: int = 10,
        language_filter: Optional[str] = None,
        similarity_threshold: float = 0.0,
        filters: Optional[SearchFilters] = None
    ) -> List[Dict[str, Any]]:
        """Execute vector similarity search.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            language_filter: Optional language filter
            similarity_threshold: Minimum similarity threshold
            filters: Optional search filters for date range and path filtering

        Returns:
            List of search results with metadata

        """
        try:
            # Ensure query_vector is float32
            query_vector_f32 = query_vector.astype(np.float32)
            
            # Build query with optional filters
            where_conditions = ["1=1"]
            # Use regular Python floats but cast in SQL
            params = [query_vector_f32.tolist()]

            if language_filter:
                where_conditions.append("c.language = ?")
                params.append(language_filter)

            if similarity_threshold > 0:
                where_conditions.append("array_cosine_similarity(e.vector, CAST(? AS FLOAT[256])) >= ?")
                params.extend([query_vector_f32.tolist(), similarity_threshold])
            
            # Add custom filters
            if filters and filters.has_filters():
                filter_conditions, filter_params = filters.to_sql_conditions()
                where_conditions.extend(filter_conditions)
                params.extend(filter_params)
            
            where_clause = " AND ".join(where_conditions[1:])  # Skip the "1=1"
            if where_clause:
                where_clause = "AND " + where_clause
            
            # Add limit parameter
            params.append(limit)

            query = f"""
                SELECT
                    c.id,
                    c.path,
                    c.title,
                    c.content,
                    c.chunk_index,
                    c.language,
                    c.metadata,
                    array_cosine_similarity(e.vector, CAST(? AS FLOAT[256])) as score
                FROM chunks c
                JOIN embeddings e ON c.id = e.chunk_id
                WHERE 1=1 {where_clause}
                ORDER BY score DESC
                LIMIT ?
            """

            results = self.conn.execute(query, params).fetchall()

            # Convert to list of dicts
            columns = ["id", "path", "title", "content", "chunk_index", "language", "metadata", "score"]
            return [dict(zip(columns, result)) for result in results]

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def bm25_search(
        self,
        terms: List[str],
        limit: int = 10,
        language_filter: Optional[str] = None,
        filters: Optional[SearchFilters] = None
    ) -> List[Dict[str, Any]]:
        """Execute BM25 text search.

        Args:
            terms: List of search terms
            limit: Maximum number of results
            language_filter: Optional language filter
            filters: Optional search filters for date range and path filtering

        Returns:
            List of search results with metadata

        """
        try:
            # Build search query
            search_text = " ".join(terms)
            
            # Build WHERE conditions
            where_conditions = ["content LIKE ?"]
            params: List[Any] = [f"%{search_text}%"]

            if language_filter:
                where_conditions.append("language = ?")
                params.append(language_filter)
            
            # Add custom filters
            if filters and filters.has_filters():
                filter_conditions, filter_params = filters.to_sql_conditions()
                where_conditions.extend(filter_conditions)
                params.extend(filter_params)
            
            # Add limit parameter
            params.append(limit)
            
            where_clause = " AND ".join(where_conditions)

            query = f"""
                SELECT
                    id, path, title, content, chunk_index, language, metadata,
                    1.0 as score
                FROM chunks c
                WHERE {where_clause}
                ORDER BY score DESC
                LIMIT ?
            """

            results = self.conn.execute(query, params).fetchall()

            # Convert to list of dicts
            columns = ["id", "path", "title", "content", "chunk_index", "language", "metadata", "score"]
            return [dict(zip(columns, result)) for result in results]

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk data or None if not found

        """
        try:
            result = self.conn.execute(
                """
                SELECT id, path, title, content, chunk_index, language,
                       created_at, modified_at, metadata
                FROM chunks WHERE id = ?
            """,
                [chunk_id],
            ).fetchone()

            if result:
                columns = ["id", "path", "title", "content", "chunk_index", "language", "created_at", "modified_at", "metadata"]
                return dict(zip(columns, result))
            return None

        except Exception as e:
            logger.error(f"Failed to get chunk by ID: {e}")
            return None
