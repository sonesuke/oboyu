"""Database service for pure database operations.

This module provides a clean interface for database operations including
storage and retrieval of chunks, embeddings, and search operations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from oboyu.indexer.core.document_processor import Chunk
from oboyu.indexer.database import Database

logger = logging.getLogger(__name__)


# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""

    def default(self, obj: object) -> object:
        """Convert datetime objects to ISO format strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class DatabaseService:
    """Pure database operations service."""

    def __init__(
        self,
        db_path: Union[str, Path],
        embedding_dimensions: int = 256,
        ef_construction: int = 128,
        ef_search: int = 64,
        m: int = 16,
        m0: Optional[int] = None,
    ) -> None:
        """Initialize database service.

        Args:
            db_path: Path to the database file
            embedding_dimensions: Dimensions of the embedding vectors
            ef_construction: Index construction parameter
            ef_search: Search time parameter
            m: Number of bidirectional links in HNSW graph
            m0: Level-0 connections

        """
        self.db_path = Path(db_path)
        self.embedding_dimensions = embedding_dimensions
        
        # Initialize database using existing Database class
        self.database = Database(
            db_path=db_path,
            embedding_dimensions=embedding_dimensions,
            ef_construction=ef_construction,
            ef_search=ef_search,
            m=m,
            m0=m0,
        )

    def initialize(self) -> None:
        """Initialize the database schema and extensions."""
        self.database.setup()

    def store_chunks(self, chunks: List[Chunk]) -> None:
        """Store document chunks in database.

        Args:
            chunks: List of document chunks to store

        """
        if not chunks:
            return

        # Use existing Database methods to store chunks
        for chunk in chunks:
            # Convert Chunk to the format expected by Database
            chunk_data = {
                'id': chunk.id,
                'path': str(chunk.path),
                'title': chunk.title,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'language': chunk.language,
                'created_at': chunk.created_at,
                'modified_at': chunk.modified_at,
                'metadata': json.dumps(chunk.metadata, cls=DateTimeEncoder),
            }
            
            # Store individual chunk using Database class methods
            if hasattr(self.database, 'insert_chunk'):
                self.database.insert_chunk(chunk_data)
            else:
                # Fallback: use direct SQL execution
                if self.database.conn:
                    self.database.conn.execute("""
                        INSERT INTO chunks (id, path, title, content, chunk_index, language, created_at, modified_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        chunk_data['id'], chunk_data['path'], chunk_data['title'],
                        chunk_data['content'], chunk_data['chunk_index'], chunk_data['language'],
                        chunk_data['created_at'], chunk_data['modified_at'], chunk_data['metadata']
                    ])

    def store_embeddings(self, chunk_ids: List[str], embeddings: List[NDArray[np.float32]]) -> None:
        """Store embedding vectors.

        Args:
            chunk_ids: List of chunk IDs corresponding to embeddings
            embeddings: List of embedding vectors

        """
        if len(chunk_ids) != len(embeddings):
            raise ValueError("Number of chunk IDs must match number of embeddings")

        # Store embeddings using Database class
        if hasattr(self.database, 'insert_embeddings'):
            self.database.insert_embeddings(chunk_ids, embeddings)
        else:
            # Fallback: use direct SQL execution
            if self.database.conn:
                import uuid
                for chunk_id, embedding in zip(chunk_ids, embeddings):
                    self.database.conn.execute("""
                        INSERT INTO embeddings (id, chunk_id, model, vector, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, [str(uuid.uuid4()), chunk_id, "cl-nagoya/ruri-v3-30m", embedding.tolist(), datetime.now()])

    def vector_search(
        self,
        query_vector: NDArray[np.float32],
        limit: int,
        language_filter: Optional[str] = None
    ) -> List[Dict[str, object]]:
        """Execute vector search query.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            language_filter: Optional language filter

        Returns:
            List of search results with metadata

        """
        try:
            # Use existing Database vector search functionality
            if hasattr(self.database, 'vector_search'):
                return self.database.vector_search(query_vector, limit, language_filter)
            else:
                # Basic fallback implementation
                return []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def bm25_search(
        self,
        terms: List[str],
        limit: int,
        language_filter: Optional[str] = None
    ) -> List[Dict[str, object]]:
        """Execute BM25 search query.

        Args:
            terms: List of search terms
            limit: Maximum number of results to return
            language_filter: Optional language filter

        Returns:
            List of search results with metadata

        """
        try:
            # Use existing Database BM25 search functionality
            if hasattr(self.database, 'bm25_search'):
                return self.database.bm25_search(terms, limit, language_filter)
            else:
                # Basic fallback implementation
                return []
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, object]]:
        """Get chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk data or None if not found

        """
        try:
            if self.database.conn:
                result = self.database.conn.execute("""
                    SELECT * FROM chunks WHERE id = ?
                """, [chunk_id]).fetchone()
                
                if result:
                    # Convert row to dict
                    columns = [desc[0] for desc in self.database.conn.description]
                    return dict(zip(columns, result))
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
            if self.database.conn:
                # Delete embeddings first
                self.database.conn.execute("""
                    DELETE FROM embeddings 
                    WHERE chunk_id IN (SELECT id FROM chunks WHERE path = ?)
                """, [str(path)])
                
                # Delete chunks
                result = self.database.conn.execute("""
                    DELETE FROM chunks WHERE path = ?
                """, [str(path)])
                
                return result.rowcount if hasattr(result, 'rowcount') else 0
            return 0
        except Exception as e:
            logger.error(f"Failed to delete chunks by path: {e}")
            return 0

    def get_chunk_count(self) -> int:
        """Get total number of chunks in the database.

        Returns:
            Total chunk count

        """
        try:
            if self.database.conn:
                result = self.database.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
                return result[0] if result else 0
            return 0
        except Exception as e:
            logger.error(f"Failed to get chunk count: {e}")
            return 0

    def get_paths_with_chunks(self) -> List[str]:
        """Get list of file paths that have chunks in the database.

        Returns:
            List of file paths

        """
        try:
            if self.database.conn:
                results = self.database.conn.execute("SELECT DISTINCT path FROM chunks").fetchall()
                return [result[0] for result in results]
            return []
        except Exception as e:
            logger.error(f"Failed to get paths with chunks: {e}")
            return []

    def clear_database(self) -> None:
        """Clear all data from the database."""
        try:
            if self.database.conn:
                self.database.conn.execute("DELETE FROM embeddings")
                self.database.conn.execute("DELETE FROM chunks")
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")

    def close(self) -> None:
        """Close database connection."""
        try:
            if hasattr(self.database, 'close') and callable(self.database.close):
                self.database.close()
            elif hasattr(self.database, 'conn') and self.database.conn:
                self.database.conn.close()
        except Exception as e:
            logger.error(f"Failed to close database: {e}")
