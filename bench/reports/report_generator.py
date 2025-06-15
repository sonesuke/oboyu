"""Main report generator for benchmark results.

This module provides the core report generation functionality that coordinates
between different formatters and creates comprehensive benchmark reports.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.benchmark_base import BenchmarkResult
from ..utils import get_timestamp


logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    formats: List[str] = field(default_factory=lambda: ["json", "txt", "html"])
    include_visualizations: bool = True
    include_raw_data: bool = False
    include_system_info: bool = True
    output_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_formats = {"json", "txt", "html", "md", "csv"}
        invalid_formats = set(self.formats) - valid_formats
        if invalid_formats:
            raise ValueError(f"Invalid report formats: {invalid_formats}")


@dataclass
class BenchmarkReport:
    """Complete benchmark report data."""
    
    title: str
    timestamp: str
    summary: Dict[str, Any]
    results: List[BenchmarkResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
    comparisons: Dict[str, Any] = field(default_factory=dict)
    visualizations: Dict[str, str] = field(default_factory=dict)  # name -> file path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "title": self.title,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "results": [
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
                for r in self.results
            ],
            "metadata": self.metadata,
            "comparisons": self.comparisons,
            "visualizations": self.visualizations,
        }


class ReportGenerator:
    """Main report generator that coordinates different output formats."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize report generator.
        
        Args:
            config: Report generation configuration
        """
        self.config = config or ReportConfig()
        self.output_dir = self.config.output_dir or Path("bench/results/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize formatters
        self._formatters = {}
        self._load_formatters()
    
    def _load_formatters(self):
        """Load available report formatters."""
        try:
            from .formatters import (
                JSONFormatter,
                TextFormatter,
                HTMLFormatter,
                MarkdownFormatter,
            )
            
            self._formatters = {
                "json": JSONFormatter(),
                "txt": TextFormatter(),
                "html": HTMLFormatter(template_dir=self.config.template_dir),
                "md": MarkdownFormatter(),
            }
        except ImportError as e:
            logger.warning(f"Some formatters not available: {e}")
    
    def generate_report(
        self,
        results: List[BenchmarkResult],
        title: str = None,
        metadata: Dict[str, Any] = None,
        previous_results: List[BenchmarkResult] = None,
    ) -> BenchmarkReport:
        """Generate a comprehensive benchmark report.
        
        Args:
            results: List of benchmark results to include in report
            title: Report title (auto-generated if not provided)
            metadata: Additional metadata to include
            previous_results: Previous results for comparison
            
        Returns:
            Generated benchmark report
        """
        timestamp = get_timestamp()
        
        if title is None:
            title = f"Benchmark Report - {timestamp}"
        
        # Generate summary
        summary = self._generate_summary(results)
        
        # Generate comparisons if previous results provided
        comparisons = {}
        if previous_results:
            comparisons = self._generate_comparisons(results, previous_results)
        
        # Generate visualizations if enabled
        visualizations = {}
        if self.config.include_visualizations:
            visualizations = self._generate_visualizations(results, previous_results)
        
        # Create report object
        report = BenchmarkReport(
            title=title,
            timestamp=timestamp,
            summary=summary,
            results=results,
            metadata=metadata or {},
            comparisons=comparisons,
            visualizations=visualizations,
        )
        
        return report
    
    def save_report(
        self,
        report: BenchmarkReport,
        filename_prefix: str = None,
    ) -> Dict[str, Path]:
        """Save report in configured formats.
        
        Args:
            report: Report to save
            filename_prefix: Optional prefix for output filenames
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        if filename_prefix is None:
            timestamp = report.timestamp.replace(":", "-").replace(" ", "_")
            filename_prefix = f"benchmark_report_{timestamp}"
        
        saved_files = {}
        
        for format_name in self.config.formats:
            if format_name not in self._formatters:
                logger.warning(f"Formatter not available for {format_name}")
                continue
            
            try:
                formatter = self._formatters[format_name]
                
                # Generate content
                content = formatter.format(report)
                
                # Determine file extension
                extensions = {
                    "json": "json",
                    "txt": "txt",
                    "html": "html",
                    "md": "md",
                    "csv": "csv",
                }
                ext = extensions.get(format_name, format_name)
                
                # Save file
                output_path = self.output_dir / f"{filename_prefix}.{ext}"
                
                if format_name == "json":
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(content, f, ensure_ascii=False, indent=2)
                else:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(content)
                
                saved_files[format_name] = output_path
                logger.info(f"Saved {format_name} report to {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to save {format_name} report: {e}")
        
        return saved_files
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if not results:
            return {}
        
        total_benchmarks = len(results)
        successful_benchmarks = sum(1 for r in results if r.success)
        failed_benchmarks = total_benchmarks - successful_benchmarks
        
        total_duration = sum(r.duration_seconds for r in results)
        avg_duration = total_duration / total_benchmarks if total_benchmarks > 0 else 0
        
        # Aggregate metrics by type
        metrics_by_type = {}
        for result in results:
            if result.success and result.metrics:
                metrics_by_type[result.benchmark_type] = result.metrics
        
        summary = {
            "total_benchmarks": total_benchmarks,
            "successful_benchmarks": successful_benchmarks,
            "failed_benchmarks": failed_benchmarks,
            "success_rate": successful_benchmarks / total_benchmarks if total_benchmarks > 0 else 0,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": avg_duration,
            "benchmark_types": list(set(r.benchmark_type for r in results)),
            "metrics_by_type": metrics_by_type,
        }
        
        return summary
    
    def _generate_comparisons(
        self,
        current_results: List[BenchmarkResult],
        previous_results: List[BenchmarkResult],
    ) -> Dict[str, Any]:
        """Generate comparison analysis between current and previous results."""
        comparisons = {}
        
        # Group results by benchmark type
        current_by_type = {r.benchmark_type: r for r in current_results if r.success}
        previous_by_type = {r.benchmark_type: r for r in previous_results if r.success}
        
        for benchmark_type in current_by_type:
            if benchmark_type not in previous_by_type:
                continue
            
            current = current_by_type[benchmark_type]
            previous = previous_by_type[benchmark_type]
            
            comparison = self._compare_results(current, previous)
            comparisons[benchmark_type] = comparison
        
        return comparisons
    
    def _compare_results(
        self,
        current: BenchmarkResult,
        previous: BenchmarkResult,
    ) -> Dict[str, Any]:
        """Compare two benchmark results."""
        comparison = {
            "duration_change": current.duration_seconds - previous.duration_seconds,
            "duration_change_percent": (
                (current.duration_seconds - previous.duration_seconds) / previous.duration_seconds * 100
                if previous.duration_seconds > 0 else 0
            ),
            "metrics_comparison": {},
        }
        
        # Compare metrics if available
        if current.metrics and previous.metrics:
            for metric_name in current.metrics:
                if metric_name in previous.metrics:
                    current_value = current.metrics[metric_name]
                    previous_value = previous.metrics[metric_name]
                    
                    if isinstance(current_value, (int, float)) and isinstance(previous_value, (int, float)):
                        change = current_value - previous_value
                        change_percent = (change / previous_value * 100) if previous_value != 0 else 0
                        
                        comparison["metrics_comparison"][metric_name] = {
                            "current": current_value,
                            "previous": previous_value,
                            "change": change,
                            "change_percent": change_percent,
                            "improved": self._is_improvement(metric_name, change),
                        }
        
        return comparison
    
    def _is_improvement(self, metric_name: str, change: float) -> bool:
        """Determine if a metric change represents an improvement."""
        # For most metrics, higher is better
        # For duration/time metrics, lower is better
        time_metrics = {
            "duration", "time", "latency", "response_time",
            "indexing_time", "search_time", "processing_time"
        }
        
        metric_lower = metric_name.lower()
        is_time_metric = any(term in metric_lower for term in time_metrics)
        
        if is_time_metric:
            return change < 0  # Lower time is better
        else:
            return change > 0  # Higher value is better for most metrics
    
    def _generate_visualizations(
        self,
        current_results: List[BenchmarkResult],
        previous_results: Optional[List[BenchmarkResult]] = None,
    ) -> Dict[str, str]:
        """Generate visualization files for the report."""
        visualizations = {}
        
        try:
            from .visualizations import (
                create_performance_charts,
                create_comparison_plots,
                save_visualization,
            )
            
            # Create performance charts
            if current_results:
                performance_chart = create_performance_charts(current_results)
                if performance_chart:
                    chart_path = self.output_dir / "performance_chart.png"
                    save_visualization(performance_chart, chart_path)
                    visualizations["performance_chart"] = str(chart_path)
            
            # Create comparison plots if previous results available
            if previous_results:
                comparison_plot = create_comparison_plots(current_results, previous_results)
                if comparison_plot:
                    plot_path = self.output_dir / "comparison_plot.png"
                    save_visualization(comparison_plot, plot_path)
                    visualizations["comparison_plot"] = str(plot_path)
            
        except ImportError:
            logger.warning("Visualization libraries not available - skipping charts")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return visualizations


# Convenience functions
def generate_report(
    results: List[BenchmarkResult],
    title: str = None,
    formats: List[str] = None,
    output_dir: Path = None,
    include_visualizations: bool = True,
    previous_results: List[BenchmarkResult] = None,
) -> BenchmarkReport:
    """Generate a benchmark report with specified options.
    
    Args:
        results: Benchmark results to include
        title: Report title
        formats: Output formats to generate
        output_dir: Output directory
        include_visualizations: Whether to include charts
        previous_results: Previous results for comparison
        
    Returns:
        Generated benchmark report
    """
    config = ReportConfig(
        formats=formats or ["json", "txt", "html"],
        include_visualizations=include_visualizations,
        output_dir=output_dir,
    )
    
    generator = ReportGenerator(config)
    report = generator.generate_report(
        results=results,
        title=title,
        previous_results=previous_results,
    )
    
    # Save in all configured formats
    generator.save_report(report)
    
    return report


def generate_all_reports(
    results_dir: Path,
    output_dir: Path = None,
    include_comparisons: bool = True,
) -> List[BenchmarkReport]:
    """Generate reports for all benchmark results in a directory.
    
    Args:
        results_dir: Directory containing benchmark result files
        output_dir: Output directory for reports
        include_comparisons: Whether to include time-series comparisons
        
    Returns:
        List of generated reports
    """
    if not results_dir.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    
    # Find all result files
    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        logger.warning(f"No result files found in {results_dir}")
        return []
    
    # Sort by timestamp
    result_files.sort()
    
    reports = []
    previous_results = None
    
    config = ReportConfig(output_dir=output_dir)
    generator = ReportGenerator(config)
    
    for result_file in result_files:
        try:
            # Load results
            with open(result_file, "r") as f:
                data = json.load(f)
            
            # Convert to BenchmarkResult objects
            results = []
            if "individual_results" in data:  # Suite results
                for result_data in data["individual_results"]:
                    result = BenchmarkResult(**result_data)
                    results.append(result)
            else:  # Single result
                result = BenchmarkResult(**data)
                results.append(result)
            
            # Generate report
            report = generator.generate_report(
                results=results,
                title=f"Report for {result_file.stem}",
                previous_results=previous_results if include_comparisons else None,
            )
            
            # Save report
            generator.save_report(report, filename_prefix=result_file.stem)
            reports.append(report)
            
            # Update previous results for next iteration
            if include_comparisons:
                previous_results = results
                
        except Exception as e:
            logger.error(f"Failed to process {result_file}: {e}")
    
    logger.info(f"Generated {len(reports)} reports in {output_dir or config.output_dir}")
    return reports