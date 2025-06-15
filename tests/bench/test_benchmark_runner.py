"""Tests for benchmark runner."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Import benchmark runner modules
import sys
from pathlib import Path

# Add bench directory to path
bench_path = Path(__file__).parent.parent.parent / "bench"
sys.path.insert(0, str(bench_path))

from bench.benchmark_runner import (
    BenchmarkRunner,
    BenchmarkSuite,
    SuiteResult,
)
from bench.core.benchmark_base import BenchmarkResult
from bench.logger import BenchmarkLogger


class TestBenchmarkSuite:
    """Test BenchmarkSuite class."""
    
    def test_suite_creation(self):
        """Test creating a benchmark suite."""
        suite = BenchmarkSuite(
            name="Test Suite",
            benchmarks=["speed", "accuracy"],
            mode="quick",
        )
        
        assert suite.name == "Test Suite"
        assert suite.benchmarks == ["speed", "accuracy"]
        assert suite.mode == "quick"
        assert suite.parallel is False  # default


class TestBenchmarkRunner:
    """Test BenchmarkRunner class."""
    
    def test_runner_creation(self):
        """Test creating a benchmark runner."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BenchmarkRunner(
                output_dir=Path(temp_dir),
                logger=BenchmarkLogger(verbose=False),
                monitor_system=False,
            )
            
            assert runner.output_dir.exists()
            assert runner.logger is not None
            assert runner.system_monitor is None  # disabled
    
    def test_predefined_suites(self):
        """Test predefined benchmark suites."""
        runner = BenchmarkRunner(monitor_system=False)
        
        # Test valid suite names
        valid_suites = ["quick", "comprehensive", "speed_only", "accuracy_only", "all"]
        
        for suite_name in valid_suites:
            suite = runner._create_predefined_suite(suite_name)
            assert isinstance(suite, BenchmarkSuite)
            assert suite.name
            assert len(suite.benchmarks) > 0
    
    def test_invalid_suite(self):
        """Test invalid suite name."""
        runner = BenchmarkRunner(monitor_system=False)
        
        with pytest.raises(ValueError, match="Unknown predefined suite"):
            runner._create_predefined_suite("invalid_suite")
    
    @patch('bench.benchmark_runner.BenchmarkRunner._run_speed_benchmarks')
    @patch('bench.benchmark_runner.BenchmarkRunner._run_accuracy_benchmarks')
    def test_run_suite_success(self, mock_accuracy, mock_speed):
        """Test successful suite execution."""
        # Mock successful benchmark results
        speed_result = BenchmarkResult(
            benchmark_type="speed",
            timestamp="2024-01-01 00:00:00",
            duration_seconds=10.0,
            success=True,
        )
        accuracy_result = BenchmarkResult(
            benchmark_type="accuracy",
            timestamp="2024-01-01 00:00:01",
            duration_seconds=20.0,
            success=True,
        )
        
        mock_speed.return_value = speed_result
        mock_accuracy.return_value = accuracy_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BenchmarkRunner(
                output_dir=Path(temp_dir),
                monitor_system=False,
            )
            
            suite = BenchmarkSuite(
                name="Test Suite",
                benchmarks=["speed", "accuracy"],
            )
            
            result = runner.run_suite(suite)
            
            assert isinstance(result, SuiteResult)
            assert result.suite_name == "Test Suite"
            assert result.benchmarks_run == 2
            assert result.benchmarks_successful == 2
            assert result.benchmarks_failed == 0
            assert len(result.individual_results) == 2
    
    @patch('bench.benchmark_runner.BenchmarkRunner._run_speed_benchmarks')
    def test_run_suite_failure(self, mock_speed):
        """Test suite execution with failures."""
        # Mock failed benchmark result
        speed_result = BenchmarkResult(
            benchmark_type="speed",
            timestamp="2024-01-01 00:00:00",
            duration_seconds=10.0,
            success=False,
            error_message="Test error",
        )
        
        mock_speed.return_value = speed_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BenchmarkRunner(
                output_dir=Path(temp_dir),
                monitor_system=False,
            )
            
            suite = BenchmarkSuite(
                name="Test Suite",
                benchmarks=["speed"],
            )
            
            result = runner.run_suite(suite)
            
            assert result.benchmarks_run == 1
            assert result.benchmarks_successful == 0
            assert result.benchmarks_failed == 1
    
    def test_run_single_benchmark(self):
        """Test running a single benchmark."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BenchmarkRunner(
                output_dir=Path(temp_dir),
                monitor_system=False,
            )
            
            # Mock the benchmark function
            with patch.object(runner, '_run_speed_benchmarks') as mock_speed:
                mock_result = BenchmarkResult(
                    benchmark_type="speed",
                    timestamp="2024-01-01 00:00:00",
                    duration_seconds=10.0,
                    success=True,
                )
                mock_speed.return_value = mock_result
                
                result = runner.run_single_benchmark("speed")
                
                assert result.benchmark_type == "speed"
                assert result.success is True
    
    def test_run_invalid_benchmark(self):
        """Test running invalid benchmark type."""
        runner = BenchmarkRunner(monitor_system=False)
        
        with pytest.raises(ValueError, match="Unknown benchmark type"):
            runner.run_single_benchmark("invalid_type")
    
    def test_suite_metrics_calculation(self):
        """Test suite summary metrics calculation."""
        runner = BenchmarkRunner(monitor_system=False)
        
        results = [
            BenchmarkResult(
                benchmark_type="speed",
                timestamp="2024-01-01 00:00:00",
                duration_seconds=10.0,
                success=True,
            ),
            BenchmarkResult(
                benchmark_type="accuracy",
                timestamp="2024-01-01 00:00:01",
                duration_seconds=20.0,
                success=False,
                error_message="Test error",
            ),
        ]
        
        metrics = runner._calculate_suite_metrics(results)
        
        assert metrics["total_duration_seconds"] == 30.0
        assert metrics["success_rate"] == 0.5
        assert metrics["benchmark_count"] == 2
        assert metrics["successful_benchmarks"] == 1
        assert metrics["failed_benchmarks"] == 1
    
    def test_empty_results_metrics(self):
        """Test metrics calculation with empty results."""
        runner = BenchmarkRunner(monitor_system=False)
        
        metrics = runner._calculate_suite_metrics([])
        assert metrics == {}
    
    def test_result_comparison(self):
        """Test benchmark result comparison."""
        runner = BenchmarkRunner(monitor_system=False)
        
        current = BenchmarkResult(
            benchmark_type="speed",
            timestamp="2024-01-01 00:00:01",
            duration_seconds=15.0,
            success=True,
            metrics={"throughput": 100.0, "latency": 0.1},
        )
        
        previous = BenchmarkResult(
            benchmark_type="speed",
            timestamp="2024-01-01 00:00:00",
            duration_seconds=10.0,
            success=True,
            metrics={"throughput": 80.0, "latency": 0.2},
        )
        
        comparison = runner._compare_results(current, previous)
        
        assert comparison["duration_change"] == 5.0
        assert comparison["duration_change_percent"] == 50.0
        
        # Check metrics comparison
        metrics_comp = comparison["metrics_comparison"]
        assert "throughput" in metrics_comp
        assert "latency" in metrics_comp
        
        # Throughput improved (higher is better)
        assert metrics_comp["throughput"]["improved"] is True
        # Latency improved (lower is better)
        assert metrics_comp["latency"]["improved"] is True
    
    def test_improvement_detection(self):
        """Test metric improvement detection."""
        runner = BenchmarkRunner(monitor_system=False)
        
        # Test time metrics (lower is better)
        assert runner._is_improvement("duration", -1.0) is True
        assert runner._is_improvement("time", 1.0) is False
        assert runner._is_improvement("latency", -0.5) is True
        
        # Test non-time metrics (higher is better)
        assert runner._is_improvement("throughput", 1.0) is True
        assert runner._is_improvement("accuracy", -0.1) is False
        assert runner._is_improvement("precision", 0.5) is True


class TestSuiteResult:
    """Test SuiteResult class."""
    
    def test_suite_result_creation(self):
        """Test creating a suite result."""
        individual_results = [
            BenchmarkResult(
                benchmark_type="speed",
                timestamp="2024-01-01 00:00:00",
                duration_seconds=10.0,
                success=True,
            )
        ]
        
        result = SuiteResult(
            suite_name="Test Suite",
            timestamp="2024-01-01 00:00:00",
            total_duration_seconds=10.0,
            benchmarks_run=1,
            benchmarks_successful=1,
            benchmarks_failed=0,
            individual_results=individual_results,
        )
        
        assert result.suite_name == "Test Suite"
        assert result.benchmarks_run == 1
        assert result.benchmarks_successful == 1
        assert result.benchmarks_failed == 0
        assert len(result.individual_results) == 1


class TestBenchmarkIntegration:
    """Integration tests for benchmark system."""
    
    def test_end_to_end_suite_execution(self):
        """Test complete suite execution workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            runner = BenchmarkRunner(
                output_dir=output_dir,
                monitor_system=False,
            )
            
            # Mock all benchmark functions to avoid dependencies
            with patch.object(runner, '_run_speed_benchmarks') as mock_speed, \
                 patch.object(runner, '_run_accuracy_benchmarks') as mock_accuracy:
                
                # Create mock results
                speed_result = BenchmarkResult(
                    benchmark_type="speed",
                    timestamp="2024-01-01 00:00:00",
                    duration_seconds=10.0,
                    success=True,
                    metrics={"files_per_second": 100},
                )
                
                accuracy_result = BenchmarkResult(
                    benchmark_type="accuracy",
                    timestamp="2024-01-01 00:00:01",
                    duration_seconds=20.0,
                    success=True,
                    metrics={"ndcg_at_10": 0.8},
                )
                
                mock_speed.return_value = speed_result
                mock_accuracy.return_value = accuracy_result
                
                # Run quick suite
                result = runner.run_suite("quick")
                
                # Verify results
                assert result.benchmarks_run >= 1
                assert result.benchmarks_successful > 0
                assert result.benchmarks_failed == 0
                
                # Check that results were saved
                result_files = list(output_dir.glob("*.json"))
                assert len(result_files) > 0


if __name__ == "__main__":
    pytest.main([__file__])