"""System health tracking and monitoring."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from ..events.events import IndexEvent
from ..events.handlers.base import EventHandler

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class OperationRecord:
    """Record of an operation for health tracking."""

    operation_id: str
    operation_type: str
    timestamp: datetime
    success: bool
    duration_seconds: float = 0.0
    error_type: Optional[str] = None


@dataclass
class HealthReport:
    """Comprehensive health report."""

    overall_status: HealthStatus
    recent_operations: List[OperationRecord]
    corruption_detected: bool
    last_successful_index: Optional[datetime]
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_operation_duration: float
    corruption_count: int
    issues: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations


class SystemHealth:
    """Tracks system health state."""
    
    def __init__(self, max_recent_operations: int = 50) -> None:
        """Initialize system health tracking.
        
        Args:
            max_recent_operations: Maximum number of recent operations to keep

        """
        self.max_recent_operations = max_recent_operations
        self.recent_operations: List[OperationRecord] = []
        self.corruption_detected = False
        self.corruption_count = 0
        self.last_successful_index: Optional[datetime] = None
        self._corruption_severity_counts: Dict[str, int] = {}
    
    def record_successful_operation(
        self,
        operation_id: str,
        operation_type: str,
        duration: float = 0.0
    ) -> None:
        """Record a successful operation.
        
        Args:
            operation_id: Unique operation identifier
            operation_type: Type of operation
            duration: Operation duration in seconds

        """
        record = OperationRecord(
            operation_id=operation_id,
            operation_type=operation_type,
            timestamp=datetime.now(),
            success=True,
            duration_seconds=duration
        )
        
        self._add_operation_record(record)
        
        if operation_type == "indexing_completed":
            self.last_successful_index = record.timestamp
    
    def record_failed_operation(
        self,
        operation_id: str,
        operation_type: str,
        error_type: str,
        duration: float = 0.0
    ) -> None:
        """Record a failed operation.
        
        Args:
            operation_id: Unique operation identifier
            operation_type: Type of operation
            error_type: Type of error that occurred
            duration: Operation duration in seconds

        """
        record = OperationRecord(
            operation_id=operation_id,
            operation_type=operation_type,
            timestamp=datetime.now(),
            success=False,
            duration_seconds=duration,
            error_type=error_type
        )
        
        self._add_operation_record(record)
    
    def set_corruption_detected(self, severity: str = "medium") -> None:
        """Mark that corruption has been detected.
        
        Args:
            severity: Severity of the corruption

        """
        self.corruption_detected = True
        self.corruption_count += 1
        self._corruption_severity_counts[severity] = (
            self._corruption_severity_counts.get(severity, 0) + 1
        )
    
    def clear_corruption_status(self) -> None:
        """Clear corruption status after resolution."""
        self.corruption_detected = False
    
    def _add_operation_record(self, record: OperationRecord) -> None:
        """Add an operation record and maintain size limit.
        
        Args:
            record: The operation record to add

        """
        self.recent_operations.append(record)
        
        # Keep only recent operations
        if len(self.recent_operations) > self.max_recent_operations:
            self.recent_operations = self.recent_operations[-self.max_recent_operations:]
    
    @property
    def overall_status(self) -> HealthStatus:
        """Calculate overall system health status."""
        if self.corruption_detected:
            # Check corruption severity
            critical_corruptions = self._corruption_severity_counts.get("critical", 0)
            high_corruptions = self._corruption_severity_counts.get("high", 0)
            
            if critical_corruptions > 0:
                return HealthStatus.CRITICAL
            elif high_corruptions > 0:
                return HealthStatus.UNHEALTHY
            else:
                return HealthStatus.DEGRADED
        
        # Check recent operation success rate
        recent_window = datetime.now() - timedelta(hours=1)
        recent_ops = [op for op in self.recent_operations if op.timestamp >= recent_window]
        
        if not recent_ops:
            return HealthStatus.HEALTHY
        
        success_rate = sum(1 for op in recent_ops if op.success) / len(recent_ops)
        
        if success_rate >= 0.95:
            return HealthStatus.HEALTHY
        elif success_rate >= 0.8:
            return HealthStatus.DEGRADED
        elif success_rate >= 0.5:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL
    
    def get_success_rate(self, time_window: Optional[timedelta] = None) -> float:
        """Get success rate for operations within a time window.
        
        Args:
            time_window: Time window to consider (default: all operations)
            
        Returns:
            Success rate as a float between 0 and 1

        """
        operations = self.recent_operations
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            operations = [op for op in operations if op.timestamp >= cutoff_time]
        
        if not operations:
            return 1.0
        
        successful = sum(1 for op in operations if op.success)
        return successful / len(operations)
    
    def get_average_duration(self, operation_type: Optional[str] = None) -> float:
        """Get average operation duration.
        
        Args:
            operation_type: Filter by operation type (optional)
            
        Returns:
            Average duration in seconds

        """
        operations = self.recent_operations
        
        if operation_type:
            operations = [op for op in operations if op.operation_type == operation_type]
        
        if not operations:
            return 0.0
        
        durations = [op.duration_seconds for op in operations if op.duration_seconds > 0]
        
        if not durations:
            return 0.0
        
        return sum(durations) / len(durations)


class IndexHealthMonitor(EventHandler):
    """Monitors system health via events."""
    
    def __init__(self) -> None:
        """Initialize the health monitor."""
        self.health_status = SystemHealth()
    
    def handle(self, event: IndexEvent) -> None:
        """Handle an event to update health status.
        
        Args:
            event: The event to process

        """
        try:
            self._update_health_from_event(event)
        except Exception as e:
            logger.error(f"Failed to update health from event {event.event_type}: {e}")
    
    def _update_health_from_event(self, event: IndexEvent) -> None:
        """Update health status based on event.
        
        Args:
            event: The event to process

        """
        if event.event_type == "indexing_completed":
            duration = getattr(event, 'duration_seconds', 0.0)
            self.health_status.record_successful_operation(
                operation_id=event.operation_id,
                operation_type=event.event_type,
                duration=duration
            )
        
        elif event.event_type == "indexing_failed":
            duration = getattr(event, 'duration_seconds', 0.0)
            error_type = getattr(event, 'error_type', 'unknown')
            self.health_status.record_failed_operation(
                operation_id=event.operation_id,
                operation_type=event.event_type,
                error_type=error_type,
                duration=duration
            )
        
        elif event.event_type == "database_cleared":
            self.health_status.record_successful_operation(
                operation_id=event.operation_id,
                operation_type=event.event_type
            )
        
        elif event.event_type == "database_clear_failed":
            error_type = getattr(event, 'error_type', 'unknown')
            self.health_status.record_failed_operation(
                operation_id=event.operation_id,
                operation_type=event.event_type,
                error_type=error_type
            )
        
        elif event.event_type == "index_corruption_detected":
            severity = getattr(event, 'severity', 'medium')
            self.health_status.set_corruption_detected(severity)
        
        elif event.event_type == "index_health_check":
            health_status = getattr(event, 'health_status', 'healthy')
            if health_status == "healthy":
                # Clear corruption status if health check passes
                self.health_status.clear_corruption_status()
    
    def get_health_report(self) -> HealthReport:
        """Get a comprehensive health report.
        
        Returns:
            Current health report

        """
        total_ops = len(self.health_status.recent_operations)
        successful_ops = sum(1 for op in self.health_status.recent_operations if op.success)
        failed_ops = total_ops - successful_ops
        
        issues = []
        if self.health_status.corruption_detected:
            issues.append(f"Index corruption detected ({self.health_status.corruption_count} times)")
        
        if self.health_status.get_success_rate(timedelta(hours=1)) < 0.8:
            issues.append("Low success rate in recent operations")
        
        if self.health_status.last_successful_index:
            time_since_last_index = datetime.now() - self.health_status.last_successful_index
            if time_since_last_index > timedelta(hours=24):
                issues.append("No successful indexing in over 24 hours")
        
        return HealthReport(
            overall_status=self.health_status.overall_status,
            recent_operations=list(self.health_status.recent_operations),
            corruption_detected=self.health_status.corruption_detected,
            last_successful_index=self.health_status.last_successful_index,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            average_operation_duration=self.health_status.get_average_duration(),
            corruption_count=self.health_status.corruption_count,
            issues=issues
        )
