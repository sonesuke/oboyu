"""Event-driven wrapper for the indexer service."""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from oboyu.crawler.crawler import CrawlerResult
from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.indexer import Indexer

from .event_bus import IndexEventBus
from .events import (
    BM25IndexUpdatedEvent,
    DatabaseClearedEvent,
    DatabaseClearFailedEvent,
    DocumentProcessedEvent,
    EmbeddingGeneratedEvent,
    HNSWIndexCreatedEvent,
    IndexingCompletedEvent,
    IndexingFailedEvent,
    IndexingStartedEvent,
)

logger = logging.getLogger(__name__)


class EventDrivenIndexer:
    """Event-driven wrapper around the core indexer."""
    
    def __init__(
        self,
        config: Optional[IndexerConfig] = None,
        event_bus: Optional[IndexEventBus] = None
    ) -> None:
        """Initialize the event-driven indexer.
        
        Args:
            config: Indexer configuration
            event_bus: Event bus for publishing events

        """
        self.indexer = Indexer(config)
        self.event_bus = event_bus or IndexEventBus()
        self.config = config or IndexerConfig()
    
    def index_documents(
        self,
        crawler_results: List[CrawlerResult],
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """Index documents with event publishing.
        
        Args:
            crawler_results: Results from document crawling
            progress_callback: Optional callback for progress updates
            
        Returns:
            Indexing result summary

        """
        operation_id = None
        start_time = time.time()
        
        try:
            # Calculate total size
            total_size_bytes = sum(
                len(result.content.encode('utf-8'))
                for result in crawler_results
                if result.content
            )
            
            # Publish start event
            start_event = IndexingStartedEvent(
                document_count=len(crawler_results),
                total_size_bytes=total_size_bytes,
                source_path=str(crawler_results[0].path.parent) if crawler_results else None
            )
            operation_id = start_event.operation_id
            self.event_bus.publish(start_event)
            
            # Create progress wrapper that publishes document processed events
            def event_progress_callback(stage: str, current: int, total: int) -> None:
                if progress_callback:
                    progress_callback(stage, current, total)
                
                # Publish document processed events for individual documents
                if stage == "processing" and current > 0 and current <= len(crawler_results):
                    doc_result = crawler_results[current - 1]
                    doc_event = DocumentProcessedEvent(
                        operation_id=operation_id,
                        document_path=str(doc_result.path),
                        success=True,
                        processing_time_seconds=time.time() - start_time
                    )
                    self.event_bus.publish(doc_event)
            
            # Delegate to the actual indexer
            result = self.indexer.index_documents(crawler_results, event_progress_callback)
            
            duration = time.time() - start_time
            
            # Publish completion event
            completion_event = IndexingCompletedEvent(
                operation_id=operation_id,
                document_count=len(crawler_results),
                chunks_created=result.get("indexed_chunks", 0),
                embeddings_generated=result.get("indexed_chunks", 0),  # Assuming 1:1 ratio
                duration_seconds=duration,
                success=True
            )
            self.event_bus.publish(completion_event)
            
            # Publish sub-operation events
            self._publish_sub_operation_events(operation_id, result)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Publish failure event
            failure_event = IndexingFailedEvent(
                operation_id=operation_id or "unknown",
                error=str(e),
                error_type=type(e).__name__,
                document_count_processed=0,  # Could track this more precisely
                duration_seconds=duration
            )
            self.event_bus.publish(failure_event)
            
            raise
    
    def _publish_sub_operation_events(self, operation_id: str, result: Dict[str, Any]) -> None:
        """Publish events for sub-operations within indexing.
        
        Args:
            operation_id: The main operation ID
            result: Indexing result with statistics

        """
        chunks_created = result.get("indexed_chunks", 0)
        
        if chunks_created > 0:
            # Publish embedding generation event
            embedding_event = EmbeddingGeneratedEvent(
                operation_id=operation_id,
                chunk_count=chunks_created,
                embedding_model=self.config.model.embedding_model if self.config.model else "unknown",
                success=True
            )
            self.event_bus.publish(embedding_event)
            
            # Publish BM25 index update event
            bm25_event = BM25IndexUpdatedEvent(
                operation_id=operation_id,
                documents_indexed=result.get("total_documents", 0),
                terms_indexed=chunks_created,  # Approximation
                success=True
            )
            self.event_bus.publish(bm25_event)
            
            # Publish HNSW index creation event
            hnsw_event = HNSWIndexCreatedEvent(
                operation_id=operation_id,
                vector_count=chunks_created,
                success=True
            )
            self.event_bus.publish(hnsw_event)
    
    def clear_index(self) -> None:
        """Clear all data from the index with event publishing."""
        operation_id = None
        
        try:
            # Get count of records before clearing
            stats = self.indexer.get_database_stats()
            records_to_delete = stats.get("total_chunks", 0)
            
            # Create clear event
            clear_event = DatabaseClearedEvent(
                records_deleted=records_to_delete,
                tables_cleared=["chunks", "embeddings", "documents", "file_metadata"]
            )
            operation_id = clear_event.operation_id
            
            # Perform the clear operation
            self.indexer.clear_index()
            
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
    
    def delete_document(self, path: Path) -> int:
        """Delete a document and all its chunks with event publishing.
        
        Args:
            path: Path to the document to delete
            
        Returns:
            Number of chunks deleted

        """
        try:
            chunks_deleted = self.indexer.delete_document(path)
            
            # Could publish a document deletion event here if needed
            logger.info(f"Deleted {chunks_deleted} chunks for document {path}")
            
            return chunks_deleted
            
        except Exception as e:
            logger.error(f"Failed to delete document {path}: {e}")
            raise
    
    def check_index_health(self) -> Dict[str, Any]:
        """Check the health of the index with event publishing.
        
        Returns:
            Health check results

        """
        try:
            health_result = self.indexer.check_index_health()
            
            # Could publish index health check event here
            # This would be handled by the health monitoring system
            
            return health_result
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def get_paths_with_chunks(self) -> List[str]:
        """Get all paths that have chunks in the database."""
        return self.indexer.get_paths_with_chunks()
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return self.indexer.get_index_stats()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.indexer.get_database_stats()
    
    def close(self) -> None:
        """Close all resources properly."""
        self.indexer.close()
    
    def __enter__(self) -> "EventDrivenIndexer":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ANN401
        """Context manager exit."""
        self.close()
    
    # Expose the event bus for external configuration
    def get_event_bus(self) -> IndexEventBus:
        """Get the event bus for handler registration."""
        return self.event_bus
    
    # Expose underlying services for backward compatibility
    @property
    def database_service(self) -> Any:  # noqa: ANN401
        """Access to underlying database service."""
        return self.indexer.database_service
    
    @property
    def embedding_service(self) -> Any:  # noqa: ANN401
        """Access to underlying embedding service."""
        return self.indexer.embedding_service
    
    @property
    def document_processor(self) -> Any:  # noqa: ANN401
        """Access to underlying document processor."""
        return self.indexer.document_processor
    
    @property
    def bm25_indexer(self) -> Any:  # noqa: ANN401
        """Access to underlying BM25 indexer."""
        return self.indexer.bm25_indexer
    
    @property
    def tokenizer_service(self) -> Any:  # noqa: ANN401
        """Access to underlying tokenizer service."""
        return self.indexer.tokenizer_service
