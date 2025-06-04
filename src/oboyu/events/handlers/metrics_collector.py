"""Metrics collection event handler."""

import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from ..events import IndexEvent
from .base import EventHandler

logger = logging.getLogger(__name__)


class IndexMetrics:
    """Collects and stores index performance metrics."""
    
    def __init__(self) -> None:
        """Initialize metrics storage."""
        self.indexing_durations: List[float] = []
        self.documents_processed: List[int] = []
        self.chunks_created: List[int] = []
        self.embeddings_generated: List[int] = []
        self.failures_count: int = 0
        self.corruption_counts: Dict[str, int] = defaultdict(int)
        self.last_indexing_time: datetime = datetime.min
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.health_status_history: List[tuple] = []  # (timestamp, status)
    
    def record_indexing_duration(self, duration: float) -> None:
        """Record indexing operation duration."""
        self.indexing_durations.append(duration)
        self.last_indexing_time = datetime.now()
    
    def record_documents_processed(self, count: int) -> None:
        """Record number of documents processed."""
        self.documents_processed.append(count)
    
    def record_chunks_created(self, count: int) -> None:
        """Record number of chunks created."""
        self.chunks_created.append(count)
    
    def record_embeddings_generated(self, count: int) -> None:
        """Record number of embeddings generated."""
        self.embeddings_generated.append(count)
    
    def increment_failures(self) -> None:
        """Increment failure count."""
        self.failures_count += 1
    
    def increment_corruption_count(self, severity: str) -> None:
        """Increment corruption count by severity."""
        self.corruption_counts[severity] += 1
    
    def record_operation(self, operation_type: str) -> None:
        """Record an operation occurrence."""
        self.operation_counts[operation_type] += 1
    
    def record_health_status(self, status: str) -> None:
        """Record health check status."""
        self.health_status_history.append((datetime.now(), status))
        # Keep only recent history (last 100 checks)
        if len(self.health_status_history) > 100:
            self.health_status_history = self.health_status_history[-100:]
    
    def get_average_indexing_duration(self) -> float:
        """Get average indexing duration."""
        if not self.indexing_durations:
            return 0.0
        return sum(self.indexing_durations) / len(self.indexing_durations)
    
    def get_total_documents_processed(self) -> int:
        """Get total number of documents processed."""
        return sum(self.documents_processed)
    
    def get_total_chunks_created(self) -> int:
        """Get total number of chunks created."""
        return sum(self.chunks_created)
    
    def get_total_embeddings_generated(self) -> int:
        """Get total number of embeddings generated."""
        return sum(self.embeddings_generated)
    
    def get_success_rate(self) -> float:
        """Calculate success rate of operations."""
        total_operations = self.operation_counts.get("indexing_started", 0)
        successful_operations = self.operation_counts.get("indexing_completed", 0)
        
        if total_operations == 0:
            return 1.0
        
        return successful_operations / total_operations
    
    def get_recent_health_status(self) -> str:
        """Get the most recent health status."""
        if not self.health_status_history:
            return "unknown"
        return self.health_status_history[-1][1]
    
    def get_metrics_summary(self) -> Dict:
        """Get a summary of all metrics."""
        return {
            "total_operations": sum(self.operation_counts.values()),
            "operation_counts": dict(self.operation_counts),
            "average_indexing_duration": self.get_average_indexing_duration(),
            "total_documents_processed": self.get_total_documents_processed(),
            "total_chunks_created": self.get_total_chunks_created(),
            "total_embeddings_generated": self.get_total_embeddings_generated(),
            "failures_count": self.failures_count,
            "corruption_counts": dict(self.corruption_counts),
            "success_rate": self.get_success_rate(),
            "last_indexing_time": self.last_indexing_time.isoformat() if self.last_indexing_time != datetime.min else None,
            "recent_health_status": self.get_recent_health_status(),
        }


class IndexMetricsCollector(EventHandler):
    """Collects performance and health metrics from events."""
    
    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self.metrics = IndexMetrics()
        self._operation_start_times: Dict[str, float] = {}
    
    def handle(self, event: IndexEvent) -> None:
        """Handle an event by updating metrics.
        
        Args:
            event: The event to process for metrics

        """
        try:
            self._update_metrics(event)
        except Exception as e:
            logger.error(f"Failed to update metrics for event {event.event_type}: {e}")
    
    def _update_metrics(self, event: IndexEvent) -> None:
        """Update metrics based on event type and data.
        
        Args:
            event: The event to process

        """
        # Record operation count
        self.metrics.record_operation(event.event_type)
        
        if event.event_type == "indexing_started":
            # Track operation start time
            self._operation_start_times[event.operation_id] = time.time()
            
            # Record documents being processed
            document_count = getattr(event, 'document_count', 0)
            if document_count > 0:
                self.metrics.record_documents_processed(document_count)
        
        elif event.event_type == "indexing_completed":
            # Record completion metrics
            duration = getattr(event, 'duration_seconds', 0.0)
            chunks_created = getattr(event, 'chunks_created', 0)
            embeddings_generated = getattr(event, 'embeddings_generated', 0)
            
            if duration > 0:
                self.metrics.record_indexing_duration(duration)
            
            if chunks_created > 0:
                self.metrics.record_chunks_created(chunks_created)
            
            if embeddings_generated > 0:
                self.metrics.record_embeddings_generated(embeddings_generated)
            
            # Clean up operation tracking
            self._operation_start_times.pop(event.operation_id, None)
        
        elif event.event_type == "indexing_failed":
            self.metrics.increment_failures()
            
            # Clean up operation tracking
            self._operation_start_times.pop(event.operation_id, None)
        
        elif event.event_type == "index_corruption_detected":
            severity = getattr(event, 'severity', 'unknown')
            self.metrics.increment_corruption_count(severity)
        
        elif event.event_type == "index_health_check":
            health_status = getattr(event, 'health_status', 'unknown')
            self.metrics.record_health_status(health_status)
        
        elif event.event_type == "embedding_generated":
            chunk_count = getattr(event, 'chunk_count', 0)
            if chunk_count > 0:
                self.metrics.record_embeddings_generated(chunk_count)
        
        elif event.event_type == "bm25_index_updated":
            documents_indexed = getattr(event, 'documents_indexed', 0)
            if documents_indexed > 0:
                self.metrics.record_documents_processed(documents_indexed)
    
    def get_metrics(self) -> IndexMetrics:
        """Get the current metrics object.
        
        Returns:
            The metrics object with current data

        """
        return self.metrics
    
    def get_metrics_summary(self) -> Dict:
        """Get a summary of current metrics.
        
        Returns:
            Dictionary with metrics summary

        """
        return self.metrics.get_metrics_summary()
    
    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        self.metrics = IndexMetrics()
        self._operation_start_times.clear()
        logger.info("Metrics reset to initial state")
