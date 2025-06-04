"""Tests for the metrics collector event handler."""

import pytest
from datetime import datetime

from oboyu.events.events import (
    IndexingStartedEvent,
    IndexingCompletedEvent,
    IndexingFailedEvent,
    IndexCorruptionDetectedEvent,
    IndexHealthCheckEvent,
    EmbeddingGeneratedEvent,
    BM25IndexUpdatedEvent,
)
from oboyu.events.handlers.metrics_collector import IndexMetricsCollector, IndexMetrics


class TestIndexMetrics:
    """Test cases for IndexMetrics."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = IndexMetrics()
        
        assert len(metrics.indexing_durations) == 0
        assert len(metrics.documents_processed) == 0
        assert len(metrics.chunks_created) == 0
        assert len(metrics.embeddings_generated) == 0
        assert metrics.failures_count == 0
        assert len(metrics.corruption_counts) == 0
        assert metrics.last_indexing_time == datetime.min
        assert len(metrics.operation_counts) == 0
    
    def test_record_indexing_duration(self):
        """Test recording indexing duration."""
        metrics = IndexMetrics()
        
        metrics.record_indexing_duration(5.5)
        metrics.record_indexing_duration(3.2)
        
        assert len(metrics.indexing_durations) == 2
        assert 5.5 in metrics.indexing_durations
        assert 3.2 in metrics.indexing_durations
        assert metrics.last_indexing_time > datetime.min
    
    def test_record_documents_processed(self):
        """Test recording documents processed."""
        metrics = IndexMetrics()
        
        metrics.record_documents_processed(10)
        metrics.record_documents_processed(5)
        
        assert len(metrics.documents_processed) == 2
        assert 10 in metrics.documents_processed
        assert 5 in metrics.documents_processed
    
    def test_increment_failures(self):
        """Test incrementing failure count."""
        metrics = IndexMetrics()
        
        metrics.increment_failures()
        metrics.increment_failures()
        
        assert metrics.failures_count == 2
    
    def test_increment_corruption_count(self):
        """Test incrementing corruption count by severity."""
        metrics = IndexMetrics()
        
        metrics.increment_corruption_count("high")
        metrics.increment_corruption_count("medium")
        metrics.increment_corruption_count("high")
        
        assert metrics.corruption_counts["high"] == 2
        assert metrics.corruption_counts["medium"] == 1
    
    def test_get_average_indexing_duration(self):
        """Test calculating average indexing duration."""
        metrics = IndexMetrics()
        
        # No durations recorded
        assert metrics.get_average_indexing_duration() == 0.0
        
        # Record some durations
        metrics.record_indexing_duration(4.0)
        metrics.record_indexing_duration(6.0)
        
        assert metrics.get_average_indexing_duration() == 5.0
    
    def test_get_total_documents_processed(self):
        """Test getting total documents processed."""
        metrics = IndexMetrics()
        
        assert metrics.get_total_documents_processed() == 0
        
        metrics.record_documents_processed(10)
        metrics.record_documents_processed(5)
        
        assert metrics.get_total_documents_processed() == 15
    
    def test_get_success_rate(self):
        """Test calculating success rate."""
        metrics = IndexMetrics()
        
        # No operations
        assert metrics.get_success_rate() == 1.0
        
        # Record operations
        metrics.record_operation("indexing_started")
        metrics.record_operation("indexing_started")
        metrics.record_operation("indexing_completed")
        
        assert metrics.get_success_rate() == 0.5  # 1 completed out of 2 started
    
    def test_record_health_status(self):
        """Test recording health status."""
        metrics = IndexMetrics()
        
        metrics.record_health_status("healthy")
        metrics.record_health_status("degraded")
        
        assert len(metrics.health_status_history) == 2
        assert metrics.get_recent_health_status() == "degraded"
    
    def test_health_status_history_limit(self):
        """Test health status history is limited."""
        metrics = IndexMetrics()
        
        # Add more than 100 entries
        for i in range(150):
            metrics.record_health_status("healthy")
        
        assert len(metrics.health_status_history) == 100
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        metrics = IndexMetrics()
        
        metrics.record_indexing_duration(5.0)
        metrics.record_documents_processed(10)
        metrics.record_chunks_created(50)
        metrics.record_embeddings_generated(50)
        metrics.increment_failures()
        metrics.increment_corruption_count("high")
        metrics.record_operation("indexing_started")
        metrics.record_operation("indexing_completed")
        
        summary = metrics.get_metrics_summary()
        
        assert summary["average_indexing_duration"] == 5.0
        assert summary["total_documents_processed"] == 10
        assert summary["total_chunks_created"] == 50
        assert summary["total_embeddings_generated"] == 50
        assert summary["failures_count"] == 1
        assert summary["corruption_counts"]["high"] == 1
        assert summary["success_rate"] == 1.0
        assert summary["total_operations"] == 2


class TestIndexMetricsCollector:
    """Test cases for IndexMetricsCollector."""
    
    def test_initialization(self):
        """Test metrics collector initialization."""
        collector = IndexMetricsCollector()
        
        assert isinstance(collector.metrics, IndexMetrics)
        assert len(collector._operation_start_times) == 0
    
    def test_handle_indexing_started_event(self):
        """Test handling indexing started event."""
        collector = IndexMetricsCollector()
        event = IndexingStartedEvent(
            document_count=10,
            operation_id="test-op-1"
        )
        
        collector.handle(event)
        
        assert collector.metrics.operation_counts["indexing_started"] == 1
        assert collector.metrics.get_total_documents_processed() == 10
        assert "test-op-1" in collector._operation_start_times
    
    def test_handle_indexing_completed_event(self):
        """Test handling indexing completed event."""
        collector = IndexMetricsCollector()
        event = IndexingCompletedEvent(
            duration_seconds=5.5,
            chunks_created=25,
            embeddings_generated=25,
            operation_id="test-op-1"
        )
        
        collector.handle(event)
        
        assert collector.metrics.operation_counts["indexing_completed"] == 1
        assert collector.metrics.get_average_indexing_duration() == 5.5
        assert collector.metrics.get_total_chunks_created() == 25
        assert collector.metrics.get_total_embeddings_generated() == 25
    
    def test_handle_indexing_failed_event(self):
        """Test handling indexing failed event."""
        collector = IndexMetricsCollector()
        event = IndexingFailedEvent(
            error="Test error",
            operation_id="test-op-1"
        )
        
        collector.handle(event)
        
        assert collector.metrics.operation_counts["indexing_failed"] == 1
        assert collector.metrics.failures_count == 1
    
    def test_handle_corruption_detected_event(self):
        """Test handling corruption detected event."""
        collector = IndexMetricsCollector()
        event = IndexCorruptionDetectedEvent(
            severity="high",
            corruption_type="data_mismatch"
        )
        
        collector.handle(event)
        
        assert collector.metrics.corruption_counts["high"] == 1
    
    def test_handle_health_check_event(self):
        """Test handling health check event."""
        collector = IndexMetricsCollector()
        event = IndexHealthCheckEvent(
            health_status="healthy"
        )
        
        collector.handle(event)
        
        assert collector.metrics.get_recent_health_status() == "healthy"
    
    def test_handle_embedding_generated_event(self):
        """Test handling embedding generated event."""
        collector = IndexMetricsCollector()
        event = EmbeddingGeneratedEvent(
            chunk_count=30
        )
        
        collector.handle(event)
        
        assert collector.metrics.get_total_embeddings_generated() == 30
    
    def test_handle_bm25_index_updated_event(self):
        """Test handling BM25 index updated event."""
        collector = IndexMetricsCollector()
        event = BM25IndexUpdatedEvent(
            documents_indexed=15
        )
        
        collector.handle(event)
        
        assert collector.metrics.get_total_documents_processed() == 15
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary from collector."""
        collector = IndexMetricsCollector()
        
        # Handle some events
        collector.handle(IndexingStartedEvent(document_count=10))
        collector.handle(IndexingCompletedEvent(duration_seconds=5.0, chunks_created=50))
        
        summary = collector.get_metrics_summary()
        
        assert summary["total_operations"] == 2
        assert summary["average_indexing_duration"] == 5.0
        assert summary["total_chunks_created"] == 50
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        collector = IndexMetricsCollector()
        
        # Add some data
        collector.handle(IndexingStartedEvent(document_count=10))
        collector._operation_start_times["test"] = 123.0
        
        # Reset
        collector.reset_metrics()
        
        assert collector.metrics.get_total_documents_processed() == 0
        assert len(collector._operation_start_times) == 0
    
    def test_handle_exception_handling(self):
        """Test that exceptions in handle method are caught."""
        collector = IndexMetricsCollector()
        
        # Create a malformed event (this should not happen in practice)
        class BadEvent:
            event_type = "bad_event"
            operation_id = "test"
            timestamp = datetime.now()
            
            def to_dict(self):
                raise Exception("Bad event")
        
        bad_event = BadEvent()
        
        # Should not raise exception
        collector.handle(bad_event)
        
        # Should still record the operation
        assert collector.metrics.operation_counts["bad_event"] == 1