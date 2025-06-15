#!/usr/bin/env python3
"""Main benchmark orchestrator for Oboyu.

This module provides the core benchmark orchestration functionality, coordinating
between different benchmark types and managing the overall execution pipeline.
It serves as the main backend for the unified benchmark runner.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .core.benchmark_base import BenchmarkBase, BenchmarkResult
from .logger import BenchmarkLogger
from .utils import SystemMonitor, Timer, get_timestamp, save_json


@dataclass
class BenchmarkSuite:
    """Configuration for a complete benchmark suite."""
    
    name: str
    benchmarks: List[str]
    mode: str = "quick"  # quick, comprehensive, custom
    output_dir: Optional[Path] = None
    parallel: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuiteResult:
    """Results from running a complete benchmark suite."""
    
    suite_name: str
    timestamp: str
    total_duration_seconds: float
    benchmarks_run: int
    benchmarks_successful: int
    benchmarks_failed: int
    individual_results: List[BenchmarkResult]
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Main benchmark orchestrator that coordinates all benchmark execution."""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        logger: Optional[BenchmarkLogger] = None,
        monitor_system: bool = True,
    ) -> None:
        """Initialize the benchmark runner.
        
        Args:
            output_dir: Directory to save results
            logger: Logger instance for output
            monitor_system: Whether to monitor system resources
        """
        self.output_dir = output_dir or Path("bench/results")
        self.logger = logger or BenchmarkLogger()
        self.monitor_system = monitor_system
        self.system_monitor = SystemMonitor() if monitor_system else None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Available benchmark types
        self._benchmark_registry = {
            "speed": self._run_speed_benchmarks,
            "accuracy": self._run_accuracy_benchmarks,
            "reranker": self._run_reranker_benchmarks,
        }
    
    def run_suite(
        self,
        suite: Union[BenchmarkSuite, str],
        **kwargs: Any,
    ) -> SuiteResult:
        """Run a complete benchmark suite.
        
        Args:
            suite: Either a BenchmarkSuite object or suite name string
            **kwargs: Additional arguments passed to individual benchmarks
            
        Returns:
            Complete suite results
        """
        if isinstance(suite, str):
            suite = self._create_predefined_suite(suite)
            
        self.logger.section(f"Running Benchmark Suite: {suite.name}")
        
        start_time = time.time()
        results: List[BenchmarkResult] = []
        
        # Start system monitoring if enabled
        if self.system_monitor:
            self.system_monitor.start()
            
        try:
            # Run each benchmark in the suite
            for benchmark_type in suite.benchmarks:
                if benchmark_type not in self._benchmark_registry:
                    self.logger.warning(f"Unknown benchmark type: {benchmark_type}")
                    continue
                    
                self.logger.info(f"Running {benchmark_type} benchmark...")
                
                try:
                    benchmark_func = self._benchmark_registry[benchmark_type]
                    result = benchmark_func(suite=suite, **kwargs)
                    results.append(result)
                    
                    if result.success:
                        self.logger.success(f"✅ {benchmark_type} benchmark completed")
                    else:
                        self.logger.error(f"❌ {benchmark_type} benchmark failed: {result.error_message}")
                        
                except Exception as e:
                    error_result = BenchmarkResult(
                        benchmark_type=benchmark_type,
                        timestamp=get_timestamp(),
                        duration_seconds=0.0,
                        success=False,
                        error_message=str(e),
                    )
                    results.append(error_result)
                    self.logger.error(f"❌ {benchmark_type} benchmark failed with exception: {e}")
                    
        finally:
            # Stop system monitoring
            if self.system_monitor:
                self.system_monitor.stop()
                
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Create suite result
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        suite_result = SuiteResult(
            suite_name=suite.name,
            timestamp=get_timestamp(),
            total_duration_seconds=total_duration,
            benchmarks_run=len(results),
            benchmarks_successful=successful,
            benchmarks_failed=failed,
            individual_results=results,
            summary_metrics=self._calculate_suite_metrics(results),
            system_info=self.system_monitor.get_summary() if self.system_monitor else {},
        )
        
        # Save results
        self._save_suite_results(suite_result)
        
        # Print summary
        self._print_suite_summary(suite_result)
        
        return suite_result
    
    def run_single_benchmark(
        self,
        benchmark_type: str,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Run a single benchmark type.
        
        Args:
            benchmark_type: Type of benchmark to run
            **kwargs: Arguments for the benchmark
            
        Returns:
            Benchmark result
        """
        if benchmark_type not in self._benchmark_registry:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
            
        self.logger.section(f"Running {benchmark_type} benchmark")
        
        # Create a minimal suite for single benchmark
        suite = BenchmarkSuite(
            name=f"single_{benchmark_type}",
            benchmarks=[benchmark_type],
        )
        
        benchmark_func = self._benchmark_registry[benchmark_type]
        return benchmark_func(suite=suite, **kwargs)
    
    def _create_predefined_suite(self, suite_name: str) -> BenchmarkSuite:
        """Create a predefined benchmark suite by name."""
        predefined_suites = {
            "quick": BenchmarkSuite(
                name="Quick Benchmark Suite",
                benchmarks=["speed", "accuracy"],
                mode="quick",
            ),
            "comprehensive": BenchmarkSuite(
                name="Comprehensive Benchmark Suite", 
                benchmarks=["speed", "accuracy", "reranker"],
                mode="comprehensive",
            ),
            "speed_only": BenchmarkSuite(
                name="Speed Benchmarks Only",
                benchmarks=["speed"],
                mode="comprehensive",
            ),
            "accuracy_only": BenchmarkSuite(
                name="Accuracy Benchmarks Only",
                benchmarks=["accuracy"],
                mode="comprehensive",
            ),
            "all": BenchmarkSuite(
                name="All Benchmarks",
                benchmarks=["speed", "accuracy", "reranker"],
                mode="comprehensive",
            ),
        }
        
        if suite_name not in predefined_suites:
            raise ValueError(f"Unknown predefined suite: {suite_name}. Available: {list(predefined_suites.keys())}")
            
        return predefined_suites[suite_name]
    
    def _run_speed_benchmarks(self, suite: BenchmarkSuite, **kwargs: Any) -> BenchmarkResult:
        """Run speed benchmarks."""
        try:
            from .speed.run_speed_benchmark import main as run_speed_main
            
            # Prepare arguments based on suite mode
            args = []
            if suite.mode == "quick":
                args.extend(["--datasets", "small"])
            elif suite.mode == "comprehensive":
                args.extend(["--datasets", "small", "medium", "large"])
                
            # Add any additional arguments from kwargs
            if "datasets" in kwargs:
                args = ["--datasets"] + kwargs["datasets"]
            if "languages" in kwargs:
                args.extend(["--languages"] + kwargs["languages"])
            if kwargs.get("verbose", False):
                args.append("--verbose")
                
            start_time = time.time()
            success = run_speed_main(args)
            end_time = time.time()
            
            return BenchmarkResult(
                benchmark_type="speed",
                timestamp=get_timestamp(),
                duration_seconds=end_time - start_time,
                success=success,
                metadata={"args": args, "mode": suite.mode},
            )
            
        except Exception as e:
            return BenchmarkResult(
                benchmark_type="speed",
                timestamp=get_timestamp(),
                duration_seconds=0.0,
                success=False,  
                error_message=str(e),
            )
    
    def _run_accuracy_benchmarks(self, suite: BenchmarkSuite, **kwargs: Any) -> BenchmarkResult:
        """Run accuracy benchmarks."""
        try:
            from .accuracy.benchmark_rag_accuracy import main as run_accuracy_main
            
            # Prepare arguments based on suite mode
            args = []
            if suite.mode == "quick":
                args.extend(["--datasets", "synthetic"])
            elif suite.mode == "comprehensive":
                args.extend(["--datasets", "synthetic", "miracl-ja", "mldr-ja"])
                
            # Add any additional arguments from kwargs
            if "datasets" in kwargs:
                args = ["--datasets"] + kwargs["datasets"]
            if "search_modes" in kwargs:
                args.extend(["--search-modes"] + kwargs["search_modes"])
            if kwargs.get("verbose", False):
                args.append("--verbose")
                
            start_time = time.time()
            success = run_accuracy_main(args)
            end_time = time.time()
            
            return BenchmarkResult(
                benchmark_type="accuracy",
                timestamp=get_timestamp(),
                duration_seconds=end_time - start_time,
                success=success,
                metadata={"args": args, "mode": suite.mode},
            )
            
        except Exception as e:
            return BenchmarkResult(
                benchmark_type="accuracy",
                timestamp=get_timestamp(),
                duration_seconds=0.0,
                success=False,
                error_message=str(e),
            )
    
    def _run_reranker_benchmarks(self, suite: BenchmarkSuite, **kwargs: Any) -> BenchmarkResult:
        """Run reranker benchmarks."""
        try:
            from .reranker.benchmark_reranking import main as run_reranker_main
            
            # Prepare arguments based on suite mode
            args = []
            if suite.mode == "quick":
                args.extend(["--models", "cl-nagoya/ruri-reranker-small"])
            elif suite.mode == "comprehensive":
                args.extend(["--models", "cl-nagoya/ruri-reranker-small", "cl-nagoya/ruri-v3-reranker-310m"])
                
            # Add any additional arguments from kwargs
            if "models" in kwargs:
                args = ["--models"] + kwargs["models"]
            if kwargs.get("verbose", False):
                args.append("--verbose")
                
            start_time = time.time()
            success = run_reranker_main(args)
            end_time = time.time()
            
            return BenchmarkResult(
                benchmark_type="reranker",
                timestamp=get_timestamp(),
                duration_seconds=end_time - start_time,
                success=success,
                metadata={"args": args, "mode": suite.mode},
            )
            
        except Exception as e:
            return BenchmarkResult(
                benchmark_type="reranker",
                timestamp=get_timestamp(),
                duration_seconds=0.0,
                success=False,
                error_message=str(e),
            )
    
    def _calculate_suite_metrics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary metrics for the suite."""
        if not results:
            return {}
            
        total_duration = sum(r.duration_seconds for r in results)
        success_rate = sum(1 for r in results if r.success) / len(results)
        
        return {
            "total_duration_seconds": total_duration,
            "success_rate": success_rate,
            "benchmark_count": len(results),
            "successful_benchmarks": sum(1 for r in results if r.success),
            "failed_benchmarks": sum(1 for r in results if not r.success),
        }
    
    def _save_suite_results(self, suite_result: SuiteResult) -> None:
        """Save suite results to file."""
        timestamp = suite_result.timestamp.replace(":", "-").replace(" ", "_")
        filename = f"benchmark_suite_{suite_result.suite_name.lower().replace(' ', '_')}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert to dict for JSON serialization
        result_dict = {
            "suite_name": suite_result.suite_name,
            "timestamp": suite_result.timestamp,
            "total_duration_seconds": suite_result.total_duration_seconds,
            "benchmarks_run": suite_result.benchmarks_run,
            "benchmarks_successful": suite_result.benchmarks_successful,
            "benchmarks_failed": suite_result.benchmarks_failed,
            "summary_metrics": suite_result.summary_metrics,
            "system_info": suite_result.system_info,
            "individual_results": [
                {
                    "benchmark_type": r.benchmark_type,
                    "timestamp": r.timestamp,
                    "duration_seconds": r.duration_seconds,
                    "success": r.success,
                    "error_message": r.error_message,
                    "metadata": r.metadata,
                    "metrics": r.metrics,
                    "system_info": r.system_info,
                }
                for r in suite_result.individual_results
            ],
        }
        
        save_json(result_dict, filepath)
        self.logger.info(f"Suite results saved to: {filepath}")
    
    def _print_suite_summary(self, suite_result: SuiteResult) -> None:
        """Print a summary of the suite results."""
        self.logger.section("Benchmark Suite Summary")
        
        print(f"Suite: {suite_result.suite_name}")
        print(f"Duration: {suite_result.total_duration_seconds:.1f}s")
        print(f"Benchmarks: {suite_result.benchmarks_successful}/{suite_result.benchmarks_run} successful")
        
        if suite_result.benchmarks_failed > 0:
            print(f"❌ {suite_result.benchmarks_failed} benchmarks failed")
            
        # Print individual results
        print("\nIndividual Results:")
        for result in suite_result.individual_results:
            status = "✅" if result.success else "❌"
            duration = f"{result.duration_seconds:.1f}s"
            print(f"  {status} {result.benchmark_type}: {duration}")
            if not result.success and result.error_message:
                print(f"     Error: {result.error_message}")
                
        # Print summary metrics
        if suite_result.summary_metrics:
            print(f"\nSummary:")
            print(f"  Success rate: {suite_result.summary_metrics.get('success_rate', 0):.1%}")


def main() -> int:
    """Main entry point for direct execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark orchestrator for Oboyu")
    parser.add_argument(
        "suite",
        nargs="?",
        default="quick",
        choices=["quick", "comprehensive", "speed_only", "accuracy_only", "all"],
        help="Benchmark suite to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create runner
    logger = BenchmarkLogger(verbose=args.verbose)
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        logger=logger,
    )
    
    # Run suite
    try:
        result = runner.run_suite(args.suite)
        return 0 if result.benchmarks_failed == 0 else 1
    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())