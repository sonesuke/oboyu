"""Unified database service for Oboyu indexer.

This module provides a comprehensive database management interface combining
the functionality of the previous database.py and database_manager.py files.

Key features:
- DuckDB with VSS extension for vector similarity search
- HNSW index for efficient vector search
- Transaction management with context managers
- Connection pooling and lifecycle management
- Schema initialization and migration support
- Batched processing for large document collections
- Type-safe query building with parameter binding
"""

import json
import logging
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Union

import duckdb
import numpy as np
from duckdb import DuckDBPyConnection
from numpy.typing import NDArray

from oboyu.indexer.config import DEFAULT_BATCH_SIZE
from oboyu.indexer.core.document_processor import Chunk
from oboyu.indexer.storage.database_connection import DatabaseConnection
from oboyu.indexer.storage.index_manager import HNSWIndexParams, IndexManager
from oboyu.indexer.storage.migrations import MigrationManager
from oboyu.indexer.storage.schema import DatabaseSchema
from oboyu.indexer.storage.utils import DateTimeEncoder

logger = logging.getLogger(__name__)


class DatabaseService(DatabaseConnection):
    """Unified database service for vector database operations.

    This class combines the functionality of database management, transaction
    handling, and query operations in a single, cohesive interface.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        embedding_dimensions: int = 256,
        hnsw_params: Optional[HNSWIndexParams] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        auto_vacuum: bool = True,
        enable_experimental_features: bool = True,
    ) -> None:
        """Initialize database service.

        Args:
            db_path: Path to the database file
            embedding_dimensions: Dimensions of the embedding vectors
            hnsw_params: HNSW index parameters
            batch_size: Batch size for operations
            auto_vacuum: Enable automatic database maintenance
            enable_experimental_features: Enable experimental DuckDB features

        """
        # Initialize parent class for connection management
        super().__init__(db_path, enable_experimental_features)
        
        self.embedding_dimensions = embedding_dimensions
        self.batch_size = batch_size
        self.auto_vacuum = auto_vacuum

        # Set default HNSW parameters if not provided
        self.hnsw_params = hnsw_params or HNSWIndexParams(
            ef_construction=128,
            ef_search=64,
            m=16,
            m0=None,
        )

        # Initialize schema
        self.schema = DatabaseSchema(embedding_dimensions)

        # Managers will be initialized after connection is established
        self.migration_manager: Optional[MigrationManager] = None
        self.index_manager: Optional[IndexManager] = None

        # Connection state
        self.conn: Optional[DuckDBPyConnection] = None
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize the database schema and extensions."""
        if self._is_initialized:
            return

        try:
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self.conn = duckdb.connect(str(self.db_path))

            # Configure database settings
            self._configure_database()

            # Install and load VSS extension
            self._setup_vss_extension()

            # Initialize managers with connection
            self.migration_manager = MigrationManager(self.conn, self.schema)
            self.index_manager = IndexManager(self.conn, self.schema)

            # Create database schema
            self._create_schema()

            # Run migrations
            self.migration_manager.run_migrations()

            # Initialize HNSW index
            self.index_manager.setup_all_indexes(self.hnsw_params)

            self._is_initialized = True
            logger.info(f"Database initialized successfully at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            if self.conn:
                self.conn.close()
                self.conn = None
            raise

    def _configure_database(self) -> None:
        """Configure database settings for optimal performance."""
        if not self.conn:
            return

        try:
            # Memory and performance settings
            self.conn.execute("SET memory_limit='2GB'")
            self.conn.execute("SET threads=4")
            
            # Enable HNSW experimental persistence for file-based databases
            self.conn.execute("SET hnsw_enable_experimental_persistence=true")

            # Enable experimental features if requested
            if self.enable_experimental_features:
                try:
                    self.conn.execute("SET enable_experimental_features=true")
                except Exception:
                    # Try alternative setting name for newer DuckDB versions
                    try:
                        self.conn.execute("SET enable_external_access=true")
                    except Exception as inner_e:
                        logger.debug(f"Failed to set enable_external_access: {inner_e}")

            # Configure auto-vacuum
            if self.auto_vacuum:
                try:
                    self.conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
                except Exception as e:
                    # Auto-vacuum might not be supported in newer DuckDB versions
                    logger.debug(f"Auto-vacuum configuration failed: {e}")

        except Exception as e:
            logger.warning(f"Failed to configure database settings: {e}")

    def _setup_vss_extension(self) -> None:
        """Install and load the VSS extension for vector operations."""
        if not self.conn:
            return

        try:
            # Install VSS extension
            self.conn.execute("INSTALL vss")
            self.conn.execute("LOAD vss")
            logger.debug("VSS extension loaded successfully")

        except Exception as e:
            logger.error(f"Failed to setup VSS extension: {e}")
            raise

    def _create_schema(self) -> None:
        """Create database schema using schema definitions."""
        if not self.conn:
            return

        try:
            # Get all table definitions
            tables = self.schema.get_all_tables()

            # Create tables in dependency order
            for table in tables:
                self.conn.execute(table.sql)

                # Create indexes for this table
                for index_sql in table.indexes:
                    self.conn.execute(index_sql)

            logger.debug("Database schema created successfully")

        except Exception as e:
            logger.error(f"Failed to create database schema: {e}")
            raise

    def _ensure_connection(self) -> DuckDBPyConnection:
        """Ensure database connection is available."""
        if not self.conn:
            self.initialize()
        assert self.conn is not None, "Database connection should be available"
        return self.conn

    @contextmanager
    def transaction(self) -> Generator[DuckDBPyConnection, None, None]:
        """Context manager for database transactions.

        Yields:
            Database connection within a transaction

        """
        if not self.conn:
            self.initialize()

        if not self.conn:
            raise RuntimeError("Database connection not available")

        try:
            self.conn.execute("BEGIN TRANSACTION")
            yield self.conn
            self.conn.execute("COMMIT")
        except Exception as e:
            try:
                self.conn.execute("ROLLBACK")
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
            raise e

    def store_chunks(self, chunks: List[Chunk], progress_callback: Optional[Callable[[str, int, int], None]] = None) -> None:
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
        self,
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
                import uuid

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

    def vector_search(
        self, query_vector: NDArray[np.float32], limit: int = 10, language_filter: Optional[str] = None, similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Execute vector similarity search.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            language_filter: Optional language filter
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of search results with metadata

        """
        conn = self._ensure_connection()

        try:
            # Ensure query_vector is float32
            query_vector_f32 = query_vector.astype(np.float32)
            
            # Build query with optional language filter
            where_clause = ""
            # Use regular Python floats but cast in SQL
            params = [query_vector_f32.tolist(), limit]

            if language_filter:
                where_clause = "AND c.language = ?"
                params.append(language_filter)

            if similarity_threshold > 0:
                where_clause += " AND array_cosine_similarity(e.vector, CAST(? AS FLOAT[256])) >= ?"
                params.extend([query_vector_f32.tolist(), similarity_threshold])

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

            results = conn.execute(query, params).fetchall()

            # Convert to list of dicts
            columns = ["id", "path", "title", "content", "chunk_index", "language", "metadata", "score"]
            return [dict(zip(columns, result)) for result in results]

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def bm25_search(self, terms: List[str], limit: int = 10, language_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Execute BM25 text search.

        Args:
            terms: List of search terms
            limit: Maximum number of results
            language_filter: Optional language filter

        Returns:
            List of search results with metadata

        """
        if not self.conn:
            self.initialize()

        try:
            # Build search query
            search_text = " ".join(terms)

            where_clause = ""
            params = [f"%{search_text}%", limit]

            if language_filter:
                where_clause = "AND language = ?"
                params.insert(-1, language_filter)

            query = f"""
                SELECT
                    id, path, title, content, chunk_index, language, metadata,
                    1.0 as score
                FROM chunks
                WHERE content LIKE ? {where_clause}
                ORDER BY score DESC
                LIMIT ?
            """

            conn = self._ensure_connection()
            results = conn.execute(query, params).fetchall()

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
        if not self.conn:
            self.initialize()

        try:
            conn = self._ensure_connection()
            result = conn.execute(
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

    def delete_chunks_by_path(self, path: Union[str, Path]) -> int:
        """Delete all chunks for a specific file path.

        Args:
            path: File path to delete chunks for

        Returns:
            Number of chunks deleted

        """
        if not self.conn:
            self.initialize()

        try:
            with self.transaction() as conn:
                # Delete embeddings first
                conn.execute(
                    """
                    DELETE FROM embeddings
                    WHERE chunk_id IN (SELECT id FROM chunks WHERE path = ?)
                """,
                    [str(path)],
                )

                # Delete chunks and get count
                result = conn.execute(
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
        """Get total number of chunks in the database."""
        if not self.conn:
            self.initialize()

        try:
            conn = self._ensure_connection()
            result = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get chunk count: {e}")
            return 0

    def get_paths_with_chunks(self) -> List[str]:
        """Get list of file paths that have chunks in the database."""
        if not self.conn:
            self.initialize()

        try:
            conn = self._ensure_connection()
            results = conn.execute("SELECT DISTINCT path FROM chunks").fetchall()
            return [result[0] for result in results]
        except Exception as e:
            logger.error(f"Failed to get paths with chunks: {e}")
            return []

    def clear_database(self) -> None:
        """Clear all data from the database."""
        if not self.conn:
            self.initialize()

        try:
            with self.transaction() as conn:
                conn.execute("DELETE FROM embeddings")
                conn.execute("DELETE FROM chunks")
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")

    def backup_database(self, backup_path: Union[str, Path]) -> bool:
        """Create a backup of the database.

        Args:
            backup_path: Path for the backup file

        Returns:
            True if backup was successful

        """
        try:
            if self.db_path.exists():
                shutil.copy2(self.db_path, backup_path)
                logger.info(f"Database backed up to {backup_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics

        """
        if not self.conn:
            self.initialize()

        try:
            conn = self._ensure_connection()
            stats = {}

            # Chunk statistics
            result = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            stats["chunk_count"] = result[0] if result else 0

            # Embedding statistics
            result = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            stats["embedding_count"] = result[0] if result else 0

            # Database size
            if self.db_path.exists():
                stats["database_size_bytes"] = self.db_path.stat().st_size
            else:
                stats["database_size_bytes"] = 0

            # Unique paths
            result = conn.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()
            stats["unique_paths"] = result[0] if result else 0

            return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                self._is_initialized = False
                logger.debug("Database connection closed")
            except Exception as e:
                logger.error(f"Failed to close database: {e}")

    def __enter__(self) -> "DatabaseService":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit."""
        self.close()
