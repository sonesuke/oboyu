"""Event-driven wrapper for the database service."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from oboyu.common.types import Chunk
from oboyu.indexer.storage.database_service import DatabaseService
from oboyu.indexer.storage.index_manager import HNSWIndexParams

from .event_bus import IndexEventBus
from .events import (
    DatabaseClearedEvent,
    DatabaseClearFailedEvent,
    DatabaseConnectionEvent,
    IndexHealthCheckEvent,
)

logger = logging.getLogger(__name__)


class EventDrivenDatabase:
    """Event-driven wrapper around the core database service."""
    
    def __init__(
        self,
        db_path: Union[str, Path],
        embedding_dimensions: int = 256,
        hnsw_params: Optional[HNSWIndexParams] = None,
        batch_size: int = 1000,
        auto_vacuum: bool = True,
        enable_experimental_features: bool = True,
        event_bus: Optional[IndexEventBus] = None
    ) -> None:
        """Initialize the event-driven database service.
        
        Args:
            db_path: Path to the database file
            embedding_dimensions: Dimensions of the embedding vectors
            hnsw_params: HNSW index parameters
            batch_size: Batch size for operations
            auto_vacuum: Enable automatic database maintenance
            enable_experimental_features: Enable experimental DuckDB features
            event_bus: Event bus for publishing events

        """
        self.database_service = DatabaseService(
            db_path=db_path,
            embedding_dimensions=embedding_dimensions,
            hnsw_params=hnsw_params,
            batch_size=batch_size,
            auto_vacuum=auto_vacuum,
            enable_experimental_features=enable_experimental_features
        )
        self.event_bus = event_bus or IndexEventBus()
        self.db_path = Path(db_path)
    
    def initialize(self) -> None:
        """Initialize the database with event publishing."""
        try:
            self.database_service.initialize()
            
            # Publish connection event
            connection_event = DatabaseConnectionEvent(
                connection_status="connected",
                database_path=str(self.db_path)
            )
            self.event_bus.publish(connection_event)
            
        except Exception as e:
            # Publish connection failure event
            connection_event = DatabaseConnectionEvent(
                connection_status="failed",
                database_path=str(self.db_path),
                error=str(e)
            )
            self.event_bus.publish(connection_event)
            raise
    
    def clear_database(self) -> None:
        """Clear all data from the database with event publishing."""
        operation_id = None
        
        try:
            # Get count of records before clearing
            stats = self.get_database_stats()
            records_to_delete = stats.get("total_chunks", 0)
            
            # Create clear event
            clear_event = DatabaseClearedEvent(
                records_deleted=records_to_delete,
                tables_cleared=["chunks", "embeddings", "documents", "file_metadata", "bm25_index", "bm25_statistics"]
            )
            operation_id = clear_event.operation_id
            
            # Perform the clear operation
            self.database_service.clear_database()
            
            # Publish success event
            self.event_bus.publish(clear_event)
            
        except Exception as e:
            # Publish failure event
            failure_event = DatabaseClearFailedEvent(
                operation_id=operation_id or "unknown",
                error=str(e),
                error_type=type(e).__name__
            )
            self.event_bus.publish(failure_event)
            
            raise
    
    def check_health(self) -> Dict[str, Any]:
        """Check the health of the database with event publishing.
        
        Returns:
            Health check results

        """
        try:
            # Perform health check
            health_result = self.database_service.check_health()
            
            # Determine health status based on results
            health_status = "healthy"
            issues_found = []
            checks_performed = ["connection", "schema", "indexes"]
            
            # Check for issues in health result
            if "error" in health_result:
                health_status = "unhealthy"
                issues_found.append(health_result["error"])
            elif health_result.get("total_chunks", 0) == 0:
                health_status = "degraded"
                issues_found.append("No indexed content found")
            
            # Get current statistics
            stats = self.get_database_stats()
            
            # Publish health check event
            health_event = IndexHealthCheckEvent(
                health_status=health_status,
                checks_performed=checks_performed,
                issues_found=issues_found,
                total_documents=stats.get("total_documents", 0),
                total_chunks=stats.get("total_chunks", 0),
                total_embeddings=stats.get("total_embeddings", 0)
            )
            self.event_bus.publish(health_event)
            
            return health_result
            
        except Exception as e:
            # Publish unhealthy status
            health_event = IndexHealthCheckEvent(
                health_status="unhealthy",
                checks_performed=["connection"],
                issues_found=[f"Health check failed: {e}"],
                total_documents=0,
                total_chunks=0,
                total_embeddings=0
            )
            self.event_bus.publish(health_event)
            
            raise
    
    def close(self) -> None:
        """Close the database connection with event publishing."""
        try:
            self.database_service.close()
            
            # Publish disconnection event
            connection_event = DatabaseConnectionEvent(
                connection_status="disconnected",
                database_path=str(self.db_path)
            )
            self.event_bus.publish(connection_event)
            
        except Exception as e:
            logger.error(f"Error during database close: {e}")
    
    # Delegate all other methods to the underlying database service
    def store_chunks(
        self,
        chunks: List[Chunk],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Store document chunks in the database."""
        return self.database_service.store_chunks(chunks, progress_callback)
    
    def store_embeddings(
        self,
        chunk_ids: List[str],
        embeddings: List[NDArray[np.float32]],
        model_name: str = "cl-nagoya/ruri-v3-30m",
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Store embeddings for chunks."""
        return self.database_service.store_embeddings(chunk_ids, embeddings, model_name, progress_callback)
    
    def store_file_metadata(
        self,
        path: Path,
        file_size: int,
        file_modified_at: datetime,
        content_hash: str,
        chunk_count: int,
    ) -> None:
        """Store file metadata for change detection."""
        return self.database_service.store_file_metadata(
            path, file_size, file_modified_at, content_hash, chunk_count
        )
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        return self.database_service.get_chunk_count()
    
    def get_paths_with_chunks(self) -> List[str]:
        """Get all paths that have chunks."""
        return self.database_service.get_paths_with_chunks()
    
    def delete_document(self, path: Path) -> int:
        """Delete a document and all its chunks."""
        return self.database_service.delete_document(path)
    
    def delete_chunks_by_path(self, path: Path) -> int:
        """Delete chunks by file path."""
        return self.database_service.delete_chunks_by_path(path)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.database_service.get_database_stats()
    
    def ensure_hnsw_index(self) -> None:
        """Ensure HNSW index exists."""
        return self.database_service.ensure_hnsw_index()
    
    def get_event_bus(self) -> IndexEventBus:
        """Get the event bus for handler registration."""
        return self.event_bus
    
    # Expose underlying properties for backward compatibility
    @property
    def conn(self) -> Any:  # noqa: ANN401
        """Get database connection."""
        return self.database_service.conn
    
    @property
    def index_manager(self) -> Any:  # noqa: ANN401
        """Get index manager."""
        return self.database_service.index_manager
    
    @property
    def chunk_repository(self) -> Any:  # noqa: ANN401
        """Get chunk repository."""
        return self.database_service.chunk_repository
    
    @property
    def embedding_repository(self) -> Any:  # noqa: ANN401
        """Get embedding repository."""
        return self.database_service.embedding_repository
    
    @property
    def statistics_repository(self) -> Any:  # noqa: ANN401
        """Get statistics repository."""
        return self.database_service.statistics_repository
    
    @property
    def search_service(self) -> Any:  # noqa: ANN401
        """Get search service."""
        return self.database_service.search_service
    
    def transaction(self) -> Any:  # noqa: ANN401
        """Get transaction context manager."""
        return self.database_service.transaction()
