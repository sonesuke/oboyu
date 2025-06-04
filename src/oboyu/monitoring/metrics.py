"""Performance metrics tracking."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PerformanceMetrics:
    """Performance metrics for index operations."""
    
    operation_durations: Dict[str, List[float]] = field(default_factory=dict)
    throughput_metrics: Dict[str, List[float]] = field(default_factory=dict)
    resource_usage: Dict[str, List[float]] = field(default_factory=dict)
    
    def record_operation_duration(self, operation_type: str, duration: float) -> None:
        """Record duration for an operation type.
        
        Args:
            operation_type: Type of operation
            duration: Duration in seconds

        """
        if operation_type not in self.operation_durations:
            self.operation_durations[operation_type] = []
        
        self.operation_durations[operation_type].append(duration)
        
        # Keep only recent measurements (last 100)
        if len(self.operation_durations[operation_type]) > 100:
            self.operation_durations[operation_type] = self.operation_durations[operation_type][-100:]
    
    def record_throughput(self, operation_type: str, items_per_second: float) -> None:
        """Record throughput for an operation type.
        
        Args:
            operation_type: Type of operation
            items_per_second: Items processed per second

        """
        if operation_type not in self.throughput_metrics:
            self.throughput_metrics[operation_type] = []
        
        self.throughput_metrics[operation_type].append(items_per_second)
        
        # Keep only recent measurements
        if len(self.throughput_metrics[operation_type]) > 100:
            self.throughput_metrics[operation_type] = self.throughput_metrics[operation_type][-100:]
    
    def get_average_duration(self, operation_type: str) -> float:
        """Get average duration for an operation type.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            Average duration in seconds

        """
        durations = self.operation_durations.get(operation_type, [])
        if not durations:
            return 0.0
        
        return sum(durations) / len(durations)
    
    def get_average_throughput(self, operation_type: str) -> float:
        """Get average throughput for an operation type.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            Average throughput in items per second

        """
        throughputs = self.throughput_metrics.get(operation_type, [])
        if not throughputs:
            return 0.0
        
        return sum(throughputs) / len(throughputs)
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of all performance metrics.
        
        Returns:
            Dictionary with metrics summary

        """
        summary = {}
        
        for operation_type in self.operation_durations:
            durations = self.operation_durations[operation_type]
            throughputs = self.throughput_metrics.get(operation_type, [])
            
            summary[operation_type] = {
                "average_duration": sum(durations) / len(durations) if durations else 0.0,
                "min_duration": min(durations) if durations else 0.0,
                "max_duration": max(durations) if durations else 0.0,
                "average_throughput": sum(throughputs) / len(throughputs) if throughputs else 0.0,
                "sample_count": len(durations),
            }
        
        return summary
