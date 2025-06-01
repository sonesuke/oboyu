"""Base class for all benchmark implementations."""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logger import BenchmarkLogger
from ..utils import SystemMonitor, Timer, get_timestamp, save_json


@dataclass
class BenchmarkResult:
    """Base class for benchmark results."""
    
    benchmark_type: str
    timestamp: str
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)


class BenchmarkBase(ABC):
    """Base class for all benchmark implementations."""
    
    def __init__(
        self,
        name: str,
        output_dir: Optional[Path] = None,
        logger: Optional[BenchmarkLogger] = None,
        monitor_system: bool = True,
    ) -> None:
        """Initialize benchmark base.
        
        Args:
            name: Name of the benchmark
            output_dir: Directory to save results
            logger: Logger instance
            monitor_system: Whether to monitor system resources

        """
        self.name = name
        self.output_dir = output_dir or Path("bench/results")
        self.logger = logger or BenchmarkLogger()
        self.monitor_system = monitor_system
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize monitoring
        self.system_monitor = SystemMonitor() if monitor_system else None
        
    @abstractmethod
    def setup(self) -> bool:
        """Set up the benchmark environment.
        
        Returns:
            True if setup successful, False otherwise

        """
        pass
    
    @abstractmethod
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the actual benchmark.
        
        Returns:
            Dictionary containing benchmark metrics

        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up after benchmark completion."""
        pass
    
    def run(self) -> BenchmarkResult:
        """Run the complete benchmark workflow.
        
        Returns:
            BenchmarkResult containing all results and metadata

        """
        timestamp = get_timestamp()
        start_time = time.time()
        
        self.logger.section(f"Running {self.name} Benchmark")
        
        # Initialize result
        result = BenchmarkResult(
            benchmark_type=self.name,
            timestamp=timestamp,
            duration_seconds=0.0,
            success=False,
        )
        
        try:
            # Setup phase
            self.logger.info("Setting up benchmark environment...")
            if not self.setup():
                result.error_message = "Benchmark setup failed"
                return result
            
            # Start system monitoring
            if self.system_monitor:
                self.system_monitor.start()
            
            # Run benchmark with timing
            self.logger.info("Running benchmark...")
            with Timer("Benchmark execution") as timer:
                metrics = self.run_benchmark()
            
            # Stop system monitoring
            if self.system_monitor:
                self.system_monitor.stop()
                result.system_info = self.system_monitor.get_summary()
            
            # Record results
            result.duration_seconds = time.time() - start_time
            result.metrics = metrics
            result.success = True
            
            self.logger.success(f"Benchmark completed in {timer.elapsed:.2f}s")
            
        except Exception as e:
            result.error_message = str(e)
            result.duration_seconds = time.time() - start_time
            self.logger.error(f"Benchmark failed: {e}")
            
        finally:
            # Always cleanup
            try:
                self.cleanup()
            except Exception as e:
                self.logger.warning(f"Cleanup failed: {e}")
        
        # Save results
        self._save_results(result)
        
        return result
    
    def _save_results(self, result: BenchmarkResult) -> None:
        """Save benchmark results to files.
        
        Args:
            result: BenchmarkResult to save

        """
        # Save JSON results
        json_path = self.output_dir / f"{self.name}_{result.timestamp}.json"
        save_json(result.__dict__, json_path)
        
        # Save human-readable report
        report_path = self.output_dir / f"{self.name}_report_{result.timestamp}.txt"
        self._save_report(result, report_path)
        
        self.logger.info(f"Results saved to {json_path}")
        self.logger.info(f"Report saved to {report_path}")
    
    def _save_report(self, result: BenchmarkResult, report_path: Path) -> None:
        """Save human-readable benchmark report.
        
        Args:
            result: BenchmarkResult to save
            report_path: Path to save report

        """
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Oboyu {self.name} Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {result.timestamp}\n")
            f.write(f"Duration: {result.duration_seconds:.2f} seconds\n")
            f.write(f"Success: {result.success}\n")
            
            if result.error_message:
                f.write(f"Error: {result.error_message}\n")
            
            f.write("\n")
            
            # Write metrics
            if result.metrics:
                f.write("Metrics:\n")
                f.write("-" * 20 + "\n")
                for key, value in result.metrics.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Write system info
            if result.system_info:
                f.write("System Information:\n")
                f.write("-" * 20 + "\n")
                for key, value in result.system_info.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Write metadata
            if result.metadata:
                f.write("Metadata:\n")
                f.write("-" * 20 + "\n")
                for key, value in result.metadata.items():
                    f.write(f"{key}: {value}\n")
    
    def get_latest_results(self, count: int = 5) -> List[BenchmarkResult]:
        """Get the latest benchmark results.
        
        Args:
            count: Number of latest results to return
            
        Returns:
            List of BenchmarkResult objects

        """
        # Find all result files for this benchmark
        pattern = f"{self.name}_*.json"
        result_files = sorted(
            self.output_dir.glob(pattern),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        results = []
        for file_path in result_files[:count]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results.append(BenchmarkResult(**data))
            except Exception as e:
                self.logger.warning(f"Failed to load result from {file_path}: {e}")
        
        return results
