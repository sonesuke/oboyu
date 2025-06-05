"""Tests for the health monitoring system."""

import pytest
from datetime import datetime, timedelta

from oboyu.events.events import (
    IndexingCompletedEvent,
    IndexingFailedEvent,
    DatabaseClearedEvent,
    IndexCorruptionDetectedEvent,
    IndexHealthCheckEvent,
)
from oboyu.monitoring.health import (
    HealthStatus,
    SystemHealth,
    IndexHealthMonitor,
    OperationRecord,
    HealthReport,
)


class TestSystemHealth:
    """Test cases for SystemHealth."""
    
    def test_initialization(self):
        """Test system health initialization."""
        health = SystemHealth()
        
        assert len(health.recent_operations) == 0
        assert health.corruption_detected is False
        assert health.corruption_count == 0
        assert health.last_successful_index is None
    
    def test_record_successful_operation(self):
        """Test recording successful operation."""
        health = SystemHealth()
        
        health.record_successful_operation("op-1", "indexing_completed", 5.5)
        
        assert len(health.recent_operations) == 1
        operation = health.recent_operations[0]
        assert operation.operation_id == "op-1"
        assert operation.operation_type == "indexing_completed"
        assert operation.success is True
        assert operation.duration_seconds == 5.5
        assert health.last_successful_index is not None
    
    def test_record_failed_operation(self):
        """Test recording failed operation."""
        health = SystemHealth()
        
        health.record_failed_operation("op-1", "indexing_failed", "ValueError", 2.0)
        
        assert len(health.recent_operations) == 1
        operation = health.recent_operations[0]
        assert operation.operation_id == "op-1"
        assert operation.operation_type == "indexing_failed"
        assert operation.success is False
        assert operation.error_type == "ValueError"
        assert operation.duration_seconds == 2.0
    
    def test_set_corruption_detected(self):
        """Test setting corruption detected."""
        health = SystemHealth()
        
        health.set_corruption_detected("high")
        
        assert health.corruption_detected is True
        assert health.corruption_count == 1
        assert health._corruption_severity_counts["high"] == 1
    
    def test_clear_corruption_status(self):
        """Test clearing corruption status."""
        health = SystemHealth()
        
        health.set_corruption_detected("high")
        assert health.corruption_detected is True
        
        health.clear_corruption_status()
        assert health.corruption_detected is False
    
    def test_recent_operations_limit(self):
        """Test that recent operations are limited."""
        health = SystemHealth(max_recent_operations=3)
        
        for i in range(5):
            health.record_successful_operation(f"op-{i}", "test", 1.0)
        
        assert len(health.recent_operations) == 3
        # Should keep the most recent ones
        assert health.recent_operations[0].operation_id == "op-2"
        assert health.recent_operations[2].operation_id == "op-4"
    
    def test_overall_status_healthy(self):
        """Test overall status calculation - healthy."""
        health = SystemHealth()
        
        # Add some successful recent operations
        for i in range(5):
            health.record_successful_operation(f"op-{i}", "indexing_completed", 1.0)
        
        assert health.overall_status == HealthStatus.HEALTHY
    
    def test_overall_status_corruption_critical(self):
        """Test overall status calculation - critical due to corruption."""
        health = SystemHealth()
        
        health.set_corruption_detected("critical")
        
        assert health.overall_status == HealthStatus.CRITICAL
    
    def test_overall_status_corruption_high(self):
        """Test overall status calculation - unhealthy due to high corruption."""
        health = SystemHealth()
        
        health.set_corruption_detected("high")
        
        assert health.overall_status == HealthStatus.UNHEALTHY
    
    def test_overall_status_corruption_medium(self):
        """Test overall status calculation - degraded due to medium corruption."""
        health = SystemHealth()
        
        health.set_corruption_detected("medium")
        
        assert health.overall_status == HealthStatus.DEGRADED
    
    def test_overall_status_low_success_rate(self):
        """Test overall status calculation - low success rate."""
        health = SystemHealth()
        
        # Add mix of successful and failed operations
        health.record_successful_operation("op-1", "indexing_completed", 1.0)
        health.record_failed_operation("op-2", "indexing_failed", "Error", 1.0)
        health.record_failed_operation("op-3", "indexing_failed", "Error", 1.0)
        health.record_failed_operation("op-4", "indexing_failed", "Error", 1.0)
        
        # Success rate is 25% (1 out of 4)
        assert health.overall_status == HealthStatus.CRITICAL
    
    def test_get_success_rate(self):
        """Test success rate calculation."""
        health = SystemHealth()
        
        # No operations
        assert health.get_success_rate() == 1.0
        
        # Add operations
        health.record_successful_operation("op-1", "test", 1.0)
        health.record_successful_operation("op-2", "test", 1.0)
        health.record_failed_operation("op-3", "test", "Error", 1.0)
        
        assert health.get_success_rate() == 2/3
    
    def test_get_success_rate_with_time_window(self):
        """Test success rate calculation with time window."""
        health = SystemHealth()
        
        # Add old operation
        old_op = OperationRecord("op-1", "test", datetime.now() - timedelta(hours=2), True)
        health.recent_operations.append(old_op)
        
        # Add recent operation
        health.record_successful_operation("op-2", "test", 1.0)
        
        # Success rate for last hour should only include recent operation
        rate = health.get_success_rate(timedelta(hours=1))
        assert rate == 1.0
    
    def test_get_average_duration(self):
        """Test average duration calculation."""
        health = SystemHealth()
        
        # No operations
        assert health.get_average_duration() == 0.0
        
        # Add operations
        health.record_successful_operation("op-1", "indexing_completed", 5.0)
        health.record_successful_operation("op-2", "indexing_completed", 3.0)
        
        assert health.get_average_duration("indexing_completed") == 4.0
    
    def test_get_average_duration_filter_by_type(self):
        """Test average duration calculation filtered by operation type."""
        health = SystemHealth()
        
        health.record_successful_operation("op-1", "indexing_completed", 5.0)
        health.record_successful_operation("op-2", "database_cleared", 1.0)
        
        assert health.get_average_duration("indexing_completed") == 5.0
        assert health.get_average_duration("database_cleared") == 1.0


class TestIndexHealthMonitor:
    """Test cases for IndexHealthMonitor."""
    
    def test_initialization(self):
        """Test health monitor initialization."""
        monitor = IndexHealthMonitor()
        
        assert isinstance(monitor.health_status, SystemHealth)
    
    def test_handle_indexing_completed_event(self):
        """Test handling indexing completed event."""
        monitor = IndexHealthMonitor()
        event = IndexingCompletedEvent(
            operation_id="op-1",
            duration_seconds=5.5
        )
        
        monitor.handle(event)
        
        assert len(monitor.health_status.recent_operations) == 1
        operation = monitor.health_status.recent_operations[0]
        assert operation.operation_id == "op-1"
        assert operation.success is True
        assert operation.duration_seconds == 5.5
    
    def test_handle_indexing_failed_event(self):
        """Test handling indexing failed event."""
        monitor = IndexHealthMonitor()
        event = IndexingFailedEvent(
            operation_id="op-1",
            error_type="ValueError",
            duration_seconds=2.0
        )
        
        monitor.handle(event)
        
        assert len(monitor.health_status.recent_operations) == 1
        operation = monitor.health_status.recent_operations[0]
        assert operation.operation_id == "op-1"
        assert operation.success is False
        assert operation.error_type == "ValueError"
    
    def test_handle_database_cleared_event(self):
        """Test handling database cleared event."""
        monitor = IndexHealthMonitor()
        event = DatabaseClearedEvent(operation_id="op-1")
        
        monitor.handle(event)
        
        assert len(monitor.health_status.recent_operations) == 1
        operation = monitor.health_status.recent_operations[0]
        assert operation.operation_id == "op-1"
        assert operation.success is True
    
    def test_handle_corruption_detected_event(self):
        """Test handling corruption detected event."""
        monitor = IndexHealthMonitor()
        event = IndexCorruptionDetectedEvent(
            severity="high",
            corruption_type="data_mismatch"
        )
        
        monitor.handle(event)
        
        assert monitor.health_status.corruption_detected is True
        assert monitor.health_status.corruption_count == 1
    
    def test_handle_health_check_event_healthy(self):
        """Test handling healthy health check event."""
        monitor = IndexHealthMonitor()
        
        # First set corruption
        monitor.health_status.set_corruption_detected("medium")
        assert monitor.health_status.corruption_detected is True
        
        # Then handle healthy check
        event = IndexHealthCheckEvent(health_status="healthy")
        monitor.handle(event)
        
        # Should clear corruption status
        assert monitor.health_status.corruption_detected is False
    
    def test_get_health_report(self):
        """Test getting health report."""
        monitor = IndexHealthMonitor()
        
        # Add some operations
        monitor.handle(IndexingCompletedEvent(operation_id="op-1", duration_seconds=5.0))
        monitor.handle(IndexingFailedEvent(operation_id="op-2", error_type="Error"))
        monitor.handle(IndexCorruptionDetectedEvent(severity="medium"))
        
        report = monitor.get_health_report()
        
        assert isinstance(report, HealthReport)
        assert report.total_operations == 2
        assert report.successful_operations == 1
        assert report.failed_operations == 1
        assert report.corruption_detected is True
        assert report.corruption_count == 1
        assert len(report.recent_operations) == 2
        assert report.success_rate == 0.5
        assert report.average_operation_duration == 5.0  # Only successful operations with duration > 0 are counted
        assert "Index corruption detected (1 times)" in report.issues
    
    def test_get_health_report_issues_detection(self):
        """Test health report issue detection."""
        monitor = IndexHealthMonitor()
        
        # Add multiple failed operations to trigger low success rate
        for i in range(5):
            monitor.handle(IndexingFailedEvent(operation_id=f"op-{i}", error_type="Error"))
        
        # Set last successful index to old time
        monitor.health_status.last_successful_index = datetime.now() - timedelta(hours=25)
        
        report = monitor.get_health_report()
        
        assert "Low success rate in recent operations" in report.issues
        assert "No successful indexing in over 24 hours" in report.issues