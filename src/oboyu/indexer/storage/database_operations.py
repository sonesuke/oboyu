"""Database CRUD operations for DatabaseService."""

import json
import uuid
from datetime import datetime
from typing import Callable, ContextManager, List, Optional, Protocol

import numpy as np
from duckdb import DuckDBPyConnection
from numpy.typing import NDArray

from oboyu.common.types import Chunk
from oboyu.indexer.storage.utils import DateTimeEncoder


class DatabaseConnectionProtocol(Protocol):
    """Protocol defining the expected database connection interface."""
    
    conn: Optional[DuckDBPyConnection]
    
    def initialize(self) -> None:
        """Initialize the database connection."""
        ...
    
    def transaction(self) -> ContextManager[DuckDBPyConnection]:
        """Create a database transaction context."""
        ...


class DatabaseOperations:
    """Mixin class providing CRUD operations for DatabaseService."""

    def store_chunks(self: DatabaseConnectionProtocol, chunks: List[Chunk], progress_callback: Optional[Callable[[str, int, int], None]] = None) -> None:
        """Store document chunks in the database.

        Args:
            chunks: List of document chunks to store
            progress_callback: Optional progress callback

        """
        if not chunks:
            return

        if not self.conn:
            self.initialize()

        with self.transaction() as conn:
            total_chunks = len(chunks)

            for i, chunk in enumerate(chunks):
                # Convert chunk to database format
                chunk_data = {
                    "id": chunk.id,
                    "path": str(chunk.path),
                    "title": chunk.title,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "language": chunk.language,
                    "created_at": chunk.created_at or datetime.now(),
                    "modified_at": chunk.modified_at or datetime.now(),
                    "metadata": json.dumps(chunk.metadata or {}, cls=DateTimeEncoder),
                }

                # Insert chunk
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    (id, path, title, content, chunk_index, language, created_at, modified_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        chunk_data["id"],
                        chunk_data["path"],
                        chunk_data["title"],
                        chunk_data["content"],
                        chunk_data["chunk_index"],
                        chunk_data["language"],
                        chunk_data["created_at"],
                        chunk_data["modified_at"],
                        chunk_data["metadata"],
                    ],
                )

                # Report progress
                if progress_callback and (i % 100 == 0 or i == total_chunks - 1):
                    progress_callback("storing", i + 1, total_chunks)

    def store_embeddings(
        self: DatabaseConnectionProtocol,
        chunk_ids: List[str],
        embeddings: List[NDArray[np.float32]],
        model_name: str = "cl-nagoya/ruri-v3-30m",
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> None:
        """Store embedding vectors in the database.

        Args:
            chunk_ids: List of chunk IDs
            embeddings: List of embedding vectors
            model_name: Name of the embedding model
            progress_callback: Optional callback for progress updates

        """
        if len(chunk_ids) != len(embeddings):
            raise ValueError("Number of chunk IDs must match number of embeddings")

        if not self.conn:
            self.initialize()

        total_embeddings = len(chunk_ids)
        
        with self.transaction() as conn:
            for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings)):
                # Generate embedding ID
                embedding_id = str(uuid.uuid4())

                # Store embedding
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings
                    (id, chunk_id, model, vector, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    [embedding_id, chunk_id, model_name, embedding.astype(np.float32).tolist(), datetime.now()],
                )
                
                # Report progress
                if progress_callback and (i % 100 == 0 or i == total_embeddings - 1):
                    progress_callback("storing_embeddings", i + 1, total_embeddings)
