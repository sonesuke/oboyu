"""Database management for Oboyu indexer.

This module handles the DuckDB database operations including schema setup,
VSS (Vector Similarity Search) extension, and HNSW index management.
"""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
from duckdb import DuckDBPyConnection
from numpy.typing import NDArray

from oboyu.indexer.processor import Chunk


# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""

    def default(self, obj: object) -> object:
        """Convert datetime objects to ISO format strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class Database:
    """Manager for the vector database using DuckDB with VSS extension."""

    def __init__(
        self,
        db_path: Union[str, Path],
        embedding_dimensions: int = 256,
        ef_construction: int = 128,
        ef_search: int = 64,
        m: int = 16,
        m0: Optional[int] = None,
    ) -> None:
        """Initialize the database.

        Args:
            db_path: Path to the database file (required, no default)
            embedding_dimensions: Dimensions of the embedding vectors
            ef_construction: Index construction parameter (build-time)
            ef_search: Search time parameter (quality vs. speed)
            m: Number of bidirectional links in HNSW graph
            m0: Level-0 connections (None means use 2*M)

        """
        self.db_path = Path(db_path)
        self.embedding_dimensions = embedding_dimensions
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m = m
        self.m0 = m0 if m0 is not None else 2 * m

        # Connection will be initialized in setup()
        self.conn: Optional[DuckDBPyConnection] = None

    def setup(self) -> None:
        """Set up the database schema and extensions."""
        # Ensure the parent directory exists
        if not str(self.db_path).startswith(":memory:"):
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Always use a file-based database to ensure persistence
        self.conn = duckdb.connect(str(self.db_path))

        # Install and load VSS extension first
        # The VSS extension is a system library that comes with DuckDB
        # See: https://duckdb.org/docs/stable/extensions/vss.html
        self.conn.execute("INSTALL vss")
        self.conn.execute("LOAD vss")

        # Enable experimental persistence for HNSW indexes
        self.conn.execute("SET hnsw_enable_experimental_persistence=true")

        # Create tables if they don't exist
        self._create_schema()

    def _create_schema(self) -> None:
        """Create database schema."""
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        # Create chunks table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id VARCHAR PRIMARY KEY,
                path VARCHAR,             -- Path to file
                title VARCHAR,            -- Chunk title (or original filename)
                content TEXT,             -- Chunk text content
                chunk_index INTEGER,      -- Chunk position in original document
                language VARCHAR,         -- Language code
                created_at TIMESTAMP,     -- Creation timestamp
                modified_at TIMESTAMP,    -- Modification timestamp
                metadata JSON             -- Additional metadata
            )
        """)

        # Create embeddings table
        # DuckDB doesn't support ? in DDL statements, use string formatting carefully
        # This is controlled input from self.embedding_dimensions, not user input
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id VARCHAR PRIMARY KEY,
                chunk_id VARCHAR,         -- Related chunk ID
                model VARCHAR,            -- Embedding model used
                vector FLOAT[{self.embedding_dimensions}],  -- Vector dimensions specific to model
                created_at TIMESTAMP,     -- Embedding generation timestamp
                FOREIGN KEY (chunk_id) REFERENCES chunks (id)
            )
        """)

        # Drop existing index if it exists
        self.conn.execute("DROP INDEX IF EXISTS vector_idx")

        # Create HNSW index with parameters
        # DuckDB doesn't support ? in DDL statements, use string formatting carefully
        # These are controlled inputs, not user input
        self.conn.execute(f"""
            CREATE INDEX vector_idx ON embeddings
            USING HNSW (vector)
            WITH (
                metric = 'cosine',
                ef_construction = {self.ef_construction},
                ef_search = {self.ef_search},
                m = {self.m},
                m0 = {self.m0}
            )
        """)

    def store_chunks(self, chunks: List[Chunk]) -> None:
        """Store document chunks in the database.

        Args:
            chunks: List of document chunks to store

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        # Prepare data for batch insertion
        chunk_data = []
        for chunk in chunks:
            chunk_data.append((
                chunk.id,
                str(chunk.path),
                chunk.title,
                chunk.content,
                chunk.chunk_index,
                chunk.language,
                chunk.created_at,
                chunk.modified_at,
                json.dumps(chunk.metadata, cls=DateTimeEncoder),
            ))

        # Insert chunks
        if chunk_data:
            self.conn.executemany("""
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
            """, chunk_data)

    def store_embeddings(
        self,
        embeddings: List[Tuple[str, str, NDArray[np.float32], datetime]],
        model_name: str,
    ) -> None:
        """Store embeddings in the database.

        Args:
            embeddings: List of (id, chunk_id, embedding, timestamp) tuples
            model_name: Name of the embedding model used

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        # Prepare data for batch insertion
        embedding_data = []
        for embedding_id, chunk_id, vector, timestamp in embeddings:
            embedding_data.append((
                embedding_id,
                chunk_id,
                model_name,
                vector.tolist(),  # Convert numpy array to Python list
                timestamp,
            ))

        # Insert embeddings
        if embedding_data:
            self.conn.executemany("""
                INSERT INTO embeddings (id, chunk_id, model, vector, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    chunk_id = excluded.chunk_id,
                    model = excluded.model,
                    vector = excluded.vector,
                    created_at = excluded.created_at
            """, embedding_data)

    def search(
        self,
        query_vector: NDArray[np.float32],
        limit: int = 10,
        language: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Search for similar documents using vector similarity.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            language: Filter by language (optional)

        Returns:
            List of matching document chunks with similarity scores

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        # First create a temporary table with the query vector
        temp_table_name = f"temp_query_vector_{uuid.uuid4().hex}"

        # We need to ensure query_vector is a numpy float32 array
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        # Create a temporary table with the query vector
        self.conn.execute(f"CREATE TEMPORARY TABLE {temp_table_name}(vec FLOAT[{self.embedding_dimensions}])")

        # Insert the vector as a list
        vector_list = query_vector.tolist()
        self.conn.execute(f"INSERT INTO {temp_table_name} VALUES (?)", [vector_list])

        # Build the query using the temporary table
        sql_query = f"""
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
        CROSS JOIN {temp_table_name} q
        """

        # Add language filter if needed
        params = []
        if language:
            sql_query += " WHERE c.language = ? "
            params.append(language)

        # Add ordering and limit
        sql_query += f" ORDER BY score ASC LIMIT {limit}"

        # Execute search
        results = self.conn.execute(sql_query, params).fetchall()

        # Clean up temporary table
        self.conn.execute(f"DROP TABLE IF EXISTS {temp_table_name}")

        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                "chunk_id": row[0],
                "path": row[1],
                "title": row[2],
                "content": row[3],
                "chunk_index": row[4],
                "language": row[5],
                "metadata": json.loads(row[6]) if row[6] else {},
                "score": float(row[7]),  # Convert to float for JSON serialization
            })

        return formatted_results

    def recompact_index(self) -> None:
        """Recompact the HNSW index to improve search performance."""
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        self.conn.execute("PRAGMA hnsw_compact_index('vector_idx')")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, object]]:
        """Retrieve a chunk by its ID.

        Args:
            chunk_id: ID of the chunk to retrieve

        Returns:
            Chunk data as dictionary or None if not found

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        result = self.conn.execute("""
            SELECT id, path, title, content, chunk_index, language, created_at, modified_at, metadata
            FROM chunks
            WHERE id = ?
        """, [chunk_id]).fetchone()

        if not result:
            return None

        return {
            "id": result[0],
            "path": result[1],
            "title": result[2],
            "content": result[3],
            "chunk_index": result[4],
            "language": result[5],
            "created_at": result[6],
            "modified_at": result[7],
            "metadata": json.loads(result[8]) if result[8] else {},
        }

    def get_chunks_by_path(self, path: Union[str, Path]) -> List[Dict[str, object]]:
        """Retrieve chunks by document path.

        Args:
            path: Path to the document

        Returns:
            List of chunk data dictionaries

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        results = self.conn.execute("""
            SELECT id, path, title, content, chunk_index, language, created_at, modified_at, metadata
            FROM chunks
            WHERE path = ?
            ORDER BY chunk_index
        """, [str(path)]).fetchall()

        chunk_data = []
        for result in results:
            chunk_data.append({
                "id": result[0],
                "path": result[1],
                "title": result[2],
                "content": result[3],
                "chunk_index": result[4],
                "language": result[5],
                "created_at": result[6],
                "modified_at": result[7],
                "metadata": json.loads(result[8]) if result[8] else {},
            })

        return chunk_data

    def delete_chunks_by_path(self, path: Union[str, Path]) -> int:
        """Delete chunks and their embeddings by document path.

        Args:
            path: Path to the document

        Returns:
            Number of chunks deleted

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        # First get the IDs of chunks to delete
        chunk_ids = self.conn.execute("""
            SELECT id FROM chunks WHERE path = ?
        """, [str(path)]).fetchall()

        if not chunk_ids:
            return 0

        # Flatten the list of tuples
        chunk_id_list = [id[0] for id in chunk_ids]

        # Delete related embeddings first (to maintain foreign key integrity)
        for chunk_id in chunk_id_list:
            self.conn.execute("DELETE FROM embeddings WHERE chunk_id = ?", [chunk_id])

        # Then delete the chunks
        result = self.conn.execute("""
            DELETE FROM chunks WHERE path = ?
        """, [str(path)]).fetchone()

        # Handle the case where result might be None
        if result is None:
            return 0

        # Cast the first element of the result to int
        deleted_count: int = int(result[0])
        return deleted_count

    def backup(self, backup_path: Union[str, Path]) -> None:
        """Create a backup of the database.

        Args:
            backup_path: Path to store the backup

        """
        # Close connection to ensure all data is written
        if self.conn:
            self.conn.close()
            self.conn = None

        # Create backup
        shutil.copy2(self.db_path, backup_path)

        # Reconnect
        self.setup()

    def clear(self) -> None:
        """Clear all data from the database.

        This method removes all chunks and embeddings from the database
        while preserving the database schema and structure.
        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        # Delete all data from embeddings first (due to foreign key constraint)
        self.conn.execute("DELETE FROM embeddings")

        # Delete all data from chunks
        self.conn.execute("DELETE FROM chunks")

        # Optionally, we could recompact the index here
        self.recompact_index()

    def get_statistics(self) -> Dict[str, object]:
        """Retrieve statistics about the database.

        Returns:
            Dictionary with database statistics including document count,
            chunk count, languages, embedding model, and last updated timestamp

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        # Get chunk count
        chunk_count_result = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        chunk_count = chunk_count_result[0] if chunk_count_result is not None else 0

        # Get document count (unique paths)
        document_count_result = self.conn.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()
        document_count = document_count_result[0] if document_count_result is not None else 0

        # Get available languages
        languages_result = self.conn.execute(
            "SELECT DISTINCT language FROM chunks WHERE language IS NOT NULL"
        ).fetchall()
        languages = [lang[0] for lang in languages_result if lang[0]]

        # Get embedding model
        model_result = self.conn.execute(
            "SELECT model FROM embeddings LIMIT 1"
        ).fetchone()
        model = model_result[0] if model_result else "unknown"

        # Get last updated timestamp
        last_updated_result = self.conn.execute(
            "SELECT MAX(modified_at) FROM chunks"
        ).fetchone()
        last_updated = last_updated_result[0] if last_updated_result and last_updated_result[0] else "unknown"

        # Return statistics
        return {
            "document_count": document_count,
            "chunk_count": chunk_count,
            "languages": languages,
            "embedding_model": model,
            "db_path": str(self.db_path),
            "last_updated": last_updated
        }

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
