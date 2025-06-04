"""Audit logging event handler."""

import logging
from typing import Optional

from ..events import IndexEvent
from ..store import EventStore
from .base import EventHandler

logger = logging.getLogger(__name__)


class AuditLogger(EventHandler):
    """Creates audit trail of all index operations."""
    
    def __init__(self, event_store: Optional[EventStore] = None) -> None:
        """Initialize the audit logger.
        
        Args:
            event_store: Optional event store for persistence

        """
        self.event_store = event_store
    
    def handle(self, event: IndexEvent) -> None:
        """Handle an event by logging it for audit purposes.
        
        Args:
            event: The event to log

        """
        # Log the event with structured data
        logger.info(
            "index_operation",
            extra={
                "event_type": event.event_type,
                "operation_id": event.operation_id,
                "timestamp": event.timestamp.isoformat(),
                "event_data": event.to_dict(),
            }
        )
        
        # Store in persistent storage if available
        if self.event_store:
            try:
                self.event_store.store_event(event)
            except Exception as e:
                logger.error(f"Failed to store event in audit log: {e}")
        
        # Log specific events with additional context
        self._log_event_details(event)
    
    def _log_event_details(self, event: IndexEvent) -> None:
        """Log event-specific details for audit trail.
        
        Args:
            event: The event to log details for

        """
        if event.event_type == "indexing_started":
            logger.info(
                f"Indexing started: {getattr(event, 'document_count', 0)} documents, "
                f"{getattr(event, 'total_size_bytes', 0)} bytes",
                extra={"operation_id": event.operation_id}
            )
        
        elif event.event_type == "indexing_completed":
            logger.info(
                f"Indexing completed: {getattr(event, 'chunks_created', 0)} chunks, "
                f"{getattr(event, 'embeddings_generated', 0)} embeddings, "
                f"{getattr(event, 'duration_seconds', 0):.2f}s",
                extra={"operation_id": event.operation_id}
            )
        
        elif event.event_type == "indexing_failed":
            logger.error(
                f"Indexing failed: {getattr(event, 'error', 'Unknown error')}",
                extra={"operation_id": event.operation_id}
            )
        
        elif event.event_type == "database_cleared":
            logger.warning(
                f"Database cleared: {getattr(event, 'records_deleted', 0)} records deleted",
                extra={"operation_id": event.operation_id}
            )
        
        elif event.event_type == "database_clear_failed":
            logger.error(
                f"Database clear failed: {getattr(event, 'error', 'Unknown error')}",
                extra={"operation_id": event.operation_id}
            )
        
        elif event.event_type == "index_corruption_detected":
            logger.critical(
                f"Index corruption detected: {getattr(event, 'corruption_type', 'Unknown')} "
                f"(severity: {getattr(event, 'severity', 'unknown')})",
                extra={"operation_id": event.operation_id}
            )
        
        elif event.event_type == "index_health_check":
            health_status = getattr(event, 'health_status', 'unknown')
            issues = getattr(event, 'issues_found', [])
            if issues:
                logger.warning(
                    f"Health check completed: {health_status}, issues found: {', '.join(issues)}",
                    extra={"operation_id": event.operation_id}
                )
            else:
                logger.info(
                    f"Health check completed: {health_status}",
                    extra={"operation_id": event.operation_id}
                )
