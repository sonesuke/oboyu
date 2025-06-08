"""Unified database service for Oboyu indexer.

This module provides a facade interface for database operations using
the repository pattern to separate concerns.

Key features:
- Repository pattern for clean separation of concerns
- Facade pattern for backward compatibility
- Delegated operations to focused repository classes
- Transaction management with context managers
- Type-safe operations with proper error handling
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Union

import numpy as np
from duckdb import DuckDBPyConnection
from numpy.typing import NDArray

from oboyu.common.types import Chunk
from oboyu.indexer.config import DEFAULT_BATCH_SIZE
from oboyu.indexer.storage.consolidated_repositories import ChunkRepository, EmbeddingRepository, StatisticsRepository
from oboyu.indexer.storage.database_lock import DatabaseLock
from oboyu.indexer.storage.database_manager import DatabaseManager
from oboyu.indexer.storage.index_manager import HNSWIndexParams, IndexManager
from oboyu.retriever.storage.database_search_service import DatabaseSearchService

if TYPE_CHECKING:
    from oboyu.common.types import SearchFilters

logger = logging.getLogger(__name__)

class DatabaseService:
    """Unified database service using repository pattern.

    This class acts as a facade for database operations, delegating to
    specialized repository classes while maintaining backward compatibility.
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
        self.db_path = Path(db_path)
        self.embedding_dimensions = embedding_dimensions
        self.batch_size = batch_size
        self.hnsw_params = hnsw_params
        
        # Initialize database manager
        self.db_manager = DatabaseManager(
            db_path=db_path,
            embedding_dimensions=embedding_dimensions,
            hnsw_params=hnsw_params,
            auto_vacuum=auto_vacuum,
            enable_experimental_features=enable_experimental_features,
        )
        
        # Repositories will be initialized after connection is established
        self.chunk_repository: Optional[ChunkRepository] = None
        self.embedding_repository: Optional[EmbeddingRepository] = None
        self.statistics_repository: Optional[StatisticsRepository] = None
        self.search_service: Optional[DatabaseSearchService] = None
        
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize the database and repositories."""
        if self._is_initialized:
            return

        try:
            # Initialize database through manager
            self.db_manager.initialize()
            
            # Get connection from manager
            conn = self.db_manager.get_connection()
            
            # Initialize repositories with connection
            self.chunk_repository = ChunkRepository(conn)
            self.embedding_repository = EmbeddingRepository(conn)
            self.statistics_repository = StatisticsRepository(conn)
            self.search_service = DatabaseSearchService(conn)
            
            self._is_initialized = True
            logger.info("Database service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database service: {e}")
            self.db_manager.close()
            raise

    @property
    def conn(self) -> Optional[DuckDBPyConnection]:
        """Get database connection for backward compatibility."""
        if self._is_initialized:
            return self.db_manager.get_connection()
        return None
    
    @property
    def index_manager(self) -> Optional[IndexManager]:
        """Get index manager for backward compatibility."""
        return self.db_manager.index_manager
    
    @contextmanager
    def transaction(self) -> Generator[DuckDBPyConnection, None, None]:
        """Context manager for database transactions.
        
        Yields:
            Database connection with active transaction
        
        """
        if not self._is_initialized:
            self.initialize()
        
        with self.db_manager.transaction() as conn:
            yield conn

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
        if not self._is_initialized:
            self.initialize()
        
        assert self.chunk_repository is not None
        with self.transaction():
            self.chunk_repository.store_chunks(chunks, progress_callback)
    
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

        """
        if not self._is_initialized:
            self.initialize()
        
        assert self.embedding_repository is not None
        with self.transaction():
            self.embedding_repository.store_embeddings(
                chunk_ids, embeddings, model_name, progress_callback
            )
    
    def vector_search(
        self,
        query_vector: NDArray[np.float32],
        limit: int = 10,
        language_filter: Optional[str] = None,
        similarity_threshold: float = 0.0,
        filters: Optional["SearchFilters"] = None,
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
        if not self._is_initialized:
            self.initialize()
        
        assert self.search_service is not None
        return self.search_service.vector_search(
            query_vector, limit, language_filter, similarity_threshold, filters
        )

    def bm25_search(
        self,
        terms: List[str],
        limit: int = 10,
        language_filter: Optional[str] = None,
        filters: Optional["SearchFilters"] = None,
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
        if not self._is_initialized:
            self.initialize()
        
        assert self.search_service is not None
        return self.search_service.bm25_search(terms, limit, language_filter, filters)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk data or None if not found

        """
        if not self._is_initialized:
            self.initialize()
        
        assert self.chunk_repository is not None
        return self.chunk_repository.get_chunk_by_id(chunk_id)

    def get_chunk_count(self) -> int:
        """Get total number of chunks in the database."""
        if not self._is_initialized:
            self.initialize()
        
        assert self.chunk_repository is not None
        return self.chunk_repository.get_chunk_count()

    def get_paths_with_chunks(self) -> List[str]:
        """Get list of file paths that have chunks in the database."""
        if not self._is_initialized:
            self.initialize()
        
        assert self.chunk_repository is not None
        return self.chunk_repository.get_paths_with_chunks()

    def clear_database(self) -> None:
        """Clear all data from the database with proper index management and locking."""
        if not self._is_initialized:
            self.initialize()

        # Use a lock to prevent concurrent clear operations
        lock = DatabaseLock(self.db_path, "clear")
        
        try:
            with lock.acquire(timeout=30.0):
                logger.info("Acquired lock for clear operation")
                
                # Clear all data from database
                assert self.embedding_repository is not None
                assert self.chunk_repository is not None
                
                conn = self.db_manager.get_connection()
                
                # First, drop the HNSW index if it exists
                # This prevents issues with foreign key constraints and speeds up deletion
                if self.index_manager and self.index_manager.hnsw_index_exists():
                    logger.info("Dropping HNSW index before clearing data")
                    self.index_manager.drop_hnsw_index()
                
                # Clear each table in separate transactions to avoid constraint issues
                try:
                    # Get list of tables to clear
                    result = conn.execute("SHOW TABLES").fetchall()
                    table_names = [row[0] for row in result if row[0] not in ['schema_version']]
                    
                    # Define deletion order to respect foreign key constraints
                    deletion_order = ['embeddings', 'inverted_index', 'document_stats', 'chunks',
                                    'vocabulary', 'collection_stats', 'file_metadata']
                    
                    # Clear tables in order, using individual transactions
                    for table_name in deletion_order:
                        if table_name in table_names:
                            try:
                                with self.transaction():
                                    conn.execute(f"DELETE FROM {table_name}")
                                logger.debug(f"Cleared table {table_name}")
                            except Exception as table_error:
                                logger.warning(f"Failed to clear table {table_name}: {table_error}")
                                # Continue with other tables
                    
                    # Clear any remaining tables not in our order
                    for table_name in table_names:
                        if table_name not in deletion_order:
                            try:
                                with self.transaction():
                                    conn.execute(f"DELETE FROM {table_name}")
                                logger.debug(f"Cleared remaining table {table_name}")
                            except Exception as table_error:
                                logger.warning(f"Failed to clear remaining table {table_name}: {table_error}")
                    
                    logger.info("Database cleared successfully")
                except Exception as e:
                    logger.error(f"Failed to clear database tables: {e}")
                    raise
            
            # Reset database state manager after clearing to ensure fresh state
            # for subsequent operations and cross-process reliability
            if hasattr(self.db_manager, 'state_manager'):
                self.db_manager.state_manager.reset_state()
                logger.info("Database state reset for cross-process reliability")
                    
        except TimeoutError as e:
            logger.error(f"Could not acquire lock for clear operation: {e}")
            raise RuntimeError("Another process is clearing the database. Please try again later.")
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            # Reset state on error as well to ensure clean state
            if hasattr(self.db_manager, 'state_manager'):
                self.db_manager.state_manager.reset_state()
            raise

    def backup_database(self, backup_path: Union[str, Path]) -> bool:
        """Create a backup of the database.

        Args:
            backup_path: Path for the backup file

        Returns:
            True if backup was successful

        """
        return self.db_manager.backup_database(backup_path)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics

        """
        if not self._is_initialized:
            self.initialize()
        
        assert self.statistics_repository is not None
        stats = self.statistics_repository.get_database_stats()
        
        # Add database size
        if self.db_path.exists():
            stats["database_size_bytes"] = self.db_path.stat().st_size
        else:
            stats["database_size_bytes"] = 0
        
        return stats

    def ensure_hnsw_index(self) -> None:
        """Ensure HNSW index exists if there are embeddings."""
        self.db_manager.ensure_hnsw_index()

    def close(self) -> None:
        """Close database connection."""
        if self._is_initialized:
            self.db_manager.close()
            self._is_initialized = False

    def __enter__(self) -> "DatabaseService":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit."""
        self.close()
    
    def store_file_metadata(self, path: Path, file_size: int, file_modified_at: datetime, content_hash: str, chunk_count: int) -> None:
        """Store or update file metadata for change detection.
        
        Args:
            path: Path to the file
            file_size: Size of the file in bytes
            file_modified_at: File modification timestamp
            content_hash: SHA-256 hash of file content
            chunk_count: Number of chunks created from this file
            
        """
        if not self._is_initialized:
            self.initialize()
            
        conn = self._ensure_connection()
        
        # Use UPSERT pattern for file metadata
        now = datetime.now()
        conn.execute("""
            INSERT INTO file_metadata (
                path, last_processed_at, file_modified_at, file_size,
                content_hash, chunk_count, processing_status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'completed', ?)
            ON CONFLICT (path) DO UPDATE SET
                last_processed_at = EXCLUDED.last_processed_at,
                file_modified_at = EXCLUDED.file_modified_at,
                file_size = EXCLUDED.file_size,
                content_hash = EXCLUDED.content_hash,
                chunk_count = EXCLUDED.chunk_count,
                processing_status = 'completed',
                updated_at = ?,
                error_message = NULL
        """, [str(path), now, file_modified_at, file_size, content_hash, chunk_count, now, now])
        
        logger.debug(f"Updated file metadata for {path}")
    
    def delete_chunks_by_path(self, path: Path) -> int:
        """Delete all chunks for a specific file path.
        
        Args:
            path: Path to the file whose chunks should be deleted
            
        Returns:
            Number of chunks deleted
            
        """
        if not self._is_initialized:
            self.initialize()
            
        conn = self._ensure_connection()
        
        # Delete embeddings first (due to foreign key constraint)
        conn.execute("""
            DELETE FROM embeddings
            WHERE chunk_id IN (
                SELECT id FROM chunks WHERE path = ?
            )
        """, [str(path)])
        
        # Delete chunks and get count
        result = conn.execute("SELECT COUNT(*) FROM chunks WHERE path = ?", [str(path)])
        row = result.fetchone()
        deleted_count = row[0] if row else 0
        
        if deleted_count > 0:
            conn.execute("DELETE FROM chunks WHERE path = ?", [str(path)])
        
        # Delete file metadata
        conn.execute("DELETE FROM file_metadata WHERE path = ?", [str(path)])
        
        logger.info(f"Deleted {deleted_count} chunks for path {path}")
        return deleted_count
    
    def _ensure_connection(self) -> DuckDBPyConnection:
        """Ensure database connection is available (backward compatibility)."""
        if not self._is_initialized:
            self.initialize()
        return self.db_manager.get_connection()
