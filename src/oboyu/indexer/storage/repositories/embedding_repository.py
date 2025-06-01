"""Repository for embedding database operations."""

import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from duckdb import DuckDBPyConnection
from numpy.typing import NDArray

from oboyu.indexer.storage.queries.data_models import EmbeddingData
from oboyu.indexer.storage.queries.embedding_queries import EmbeddingQueries

logger = logging.getLogger(__name__)


class EmbeddingRepository:
    """Repository for embedding-related database operations."""

    def __init__(self, connection: DuckDBPyConnection) -> None:
        """Initialize embedding repository.

        Args:
            connection: DuckDB database connection

        """
        self.connection = connection
        self.queries = EmbeddingQueries()

    def store_embeddings(
        self,
        chunk_ids: List[str],
        embeddings: List[NDArray[np.float32]],
        model_name: str = "cl-nagoya/ruri-v3-30m",
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Store embedding vectors in the database.

        Args:
            chunk_ids: List of chunk IDs
            embeddings: List of embedding vectors
            model_name: Name of the embedding model
            progress_callback: Optional callback for progress updates

        Raises:
            ValueError: If number of chunk IDs doesn't match embeddings

        """
        if len(chunk_ids) != len(embeddings):
            raise ValueError("Number of chunk IDs must match number of embeddings")

        total_embeddings = len(chunk_ids)

        for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings)):
            # Create embedding data
            embedding_data = EmbeddingData(
                id=str(uuid.uuid4()),
                chunk_id=chunk_id,
                model=model_name,
                vector=embedding.astype(np.float32),
                created_at=datetime.now(),
            )

            # Get query and parameters
            sql, params = self.queries.upsert_embedding(embedding_data)

            # Execute query
            self.connection.execute(sql, params)

            # Report progress
            if progress_callback and (i % 100 == 0 or i == total_embeddings - 1):
                progress_callback("storing_embeddings", i + 1, total_embeddings)

    def get_embedding_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get embedding for a specific chunk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Embedding data or None if not found

        """
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
                # Convert to dictionary
                columns = ["id", "chunk_id", "model", "vector", "created_at"]
                embedding_data = dict(zip(columns, result))

                # Convert vector list to numpy array
                if embedding_data.get("vector"):
                    embedding_data["vector"] = np.array(embedding_data["vector"], dtype=np.float32)

                return embedding_data

            return None

        except Exception as e:
            logger.error(f"Failed to get embedding by chunk ID: {e}")
            return None

    def get_embeddings_batch(self, chunk_ids: List[str]) -> Dict[str, NDArray[np.float32]]:
        """Get embeddings for multiple chunks.

        Args:
            chunk_ids: List of chunk identifiers

        Returns:
            Dictionary mapping chunk_id to embedding vector

        """
        try:
            if not chunk_ids:
                return {}

            # Build query with placeholders
            placeholders = ",".join(["?" for _ in chunk_ids])
            results = self.connection.execute(
                f"""
                SELECT chunk_id, vector
                FROM embeddings
                WHERE chunk_id IN ({placeholders})
            """,
                chunk_ids,
            ).fetchall()

            # Convert to dictionary
            embeddings = {}
            for chunk_id, vector in results:
                if vector:
                    embeddings[chunk_id] = np.array(vector, dtype=np.float32)

            return embeddings

        except Exception as e:
            logger.error(f"Failed to get embeddings batch: {e}")
            return {}

    def delete_embeddings_by_chunk_ids(self, chunk_ids: List[str]) -> int:
        """Delete embeddings for specific chunks.

        Args:
            chunk_ids: List of chunk identifiers

        Returns:
            Number of embeddings deleted

        """
        try:
            if not chunk_ids:
                return 0

            # Build query with placeholders
            placeholders = ",".join(["?" for _ in chunk_ids])
            result = self.connection.execute(
                f"""
                DELETE FROM embeddings
                WHERE chunk_id IN ({placeholders})
            """,
                chunk_ids,
            )

            return result.rowcount if hasattr(result, "rowcount") else 0

        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            return 0

    def delete_embeddings_by_path(self, path: str) -> int:
        """Delete embeddings for all chunks of a specific file.

        Args:
            path: File path

        Returns:
            Number of embeddings deleted

        """
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
        """Get total number of embeddings in the database.

        Returns:
            Total embedding count

        """
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
        """Get all embeddings for a specific model.

        Args:
            model_name: Name of the embedding model

        Returns:
            List of embedding data

        """
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
        """Update the model name for an embedding.

        Args:
            chunk_id: Chunk identifier
            new_model_name: New model name

        Returns:
            True if update was successful

        """
        try:
            result = self.connection.execute(
                """
                UPDATE embeddings
                SET model = ?
                WHERE chunk_id = ?
            """,
                [new_model_name, chunk_id],
            )

            return result.rowcount > 0 if hasattr(result, "rowcount") else False

        except Exception as e:
            logger.error(f"Failed to update embedding model: {e}")
            return False
