"""Report generation for benchmark results."""

import json
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

from bench.speed.results import BenchmarkRun
from bench.utils import format_time, print_header, print_metric, print_section

console = Console()


class BenchmarkReporter:
    """Generate reports from benchmark results."""
    
    def __init__(self, run: BenchmarkRun) -> None:
        """Initialize reporter with a benchmark run."""
        self.run = run
    
    def generate_text_report(self) -> str:
        """Generate a human-readable text report."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("OBOYU PERFORMANCE BENCHMARK REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Run information
        lines.append("Run Information:")
        lines.append(f"  Run ID: {self.run.run_id}")
        lines.append(f"  Timestamp: {self.run.timestamp}")
        lines.append("")
        
        # System information
        lines.append("System Information:")
        lines.append(f"  Platform: {self.run.system_info.platform}")
        lines.append(f"  Python: {self.run.system_info.python['version'].split()[0]}")
        lines.append(f"  Oboyu Version: {self.run.system_info.oboyu_version}")
        lines.append(f"  CPU Count: {self.run.system_info.cpu_count}")
        lines.append(f"  Memory: {self.run.system_info.memory_gb:.1f} GB")
        lines.append("")
        
        # Indexing results
        if self.run.indexing_results:
            lines.append("-" * 80)
            lines.append("INDEXING PERFORMANCE")
            lines.append("-" * 80)
            lines.append("")
            
            for size, result in sorted(self.run.indexing_results.items()):
                lines.append(f"{size.upper()} Dataset:")
                lines.append(f"  Total Files: {result.total_files:,}")
                lines.append(f"  Total Chunks: {result.total_chunks:,}")
                lines.append(f"  Total Time: {format_time(result.total_time)}")
                lines.append("")
                lines.append("  Time Breakdown:")
                lines.append(f"    File Discovery: {format_time(result.file_discovery_time)} ({result.file_discovery_time/result.total_time*100:.1f}%)")
                lines.append(f"    Content Extraction: {format_time(result.content_extraction_time)} ({result.content_extraction_time/result.total_time*100:.1f}%)")
                lines.append(f"    Embedding Generation: {format_time(result.embedding_generation_time)} ({result.embedding_generation_time/result.total_time*100:.1f}%)")
                lines.append(f"    Database Storage: {format_time(result.database_storage_time)} ({result.database_storage_time/result.total_time*100:.1f}%)")
                lines.append("")
                lines.append("  Performance Metrics:")
                lines.append(f"    Files/second: {result.files_per_second:.2f}")
                lines.append(f"    Chunks/second: {result.chunks_per_second:.2f}")
                
                if result.system_metrics:
                    lines.append("")
                    lines.append("  System Usage:")
                    lines.append(f"    CPU Average: {result.system_metrics.get('cpu_percent_avg', 0):.1f}%")
                    lines.append(f"    CPU Peak: {result.system_metrics.get('cpu_percent_max', 0):.1f}%")
                    lines.append(f"    Memory Average: {result.system_metrics.get('memory_usage_mb_avg', 0):.1f} MB")
                    lines.append(f"    Memory Peak: {result.system_metrics.get('memory_usage_mb_max', 0):.1f} MB")
                
                lines.append("")
        
        # Search results
        if self.run.search_results:
            lines.append("-" * 80)
            lines.append("SEARCH PERFORMANCE")
            lines.append("-" * 80)
            lines.append("")
            
            for key, result in sorted(self.run.search_results.items()):
                dataset, language = key.rsplit("_", 1)
                lines.append(f"{dataset.upper()} Dataset - {language.capitalize()} Queries:")
                lines.append(f"  Total Queries: {result.total_queries}")
                lines.append("")
                
                # Results by top_k
                lines.append("  Results by top_k:")
                for top_k in sorted(result.top_k_values):
                    stats = result.statistics.get(top_k, {})
                    qps = result.queries_per_second.get(top_k, 0)
                    
                    lines.append(f"    k={top_k}:")
                    lines.append(f"      Queries/second: {qps:.2f}")
                    lines.append(f"      Mean response: {format_time(stats.get('mean', 0))}")
                    lines.append(f"      Median response: {format_time(stats.get('median', 0))}")
                    lines.append(f"      95th percentile: {format_time(stats.get('p95', 0))}")
                    lines.append(f"      99th percentile: {format_time(stats.get('p99', 0))}")
                
                if result.system_metrics:
                    lines.append("")
                    lines.append("  System Usage:")
                    lines.append(f"    CPU Average: {result.system_metrics.get('cpu_percent_avg', 0):.1f}%")
                    lines.append(f"    Memory Average: {result.system_metrics.get('memory_usage_mb_avg', 0):.1f} MB")
                
                lines.append("")
        
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def generate_json_report(self) -> Dict[str, Any]:
        """Generate a JSON report with all data."""
        from dataclasses import asdict
        return asdict(self.run)
    
    def print_summary(self) -> None:
        """Print a summary of results to console."""
        print_header("Benchmark Results Summary")
        
        # Run info
        print_section("Run Information")
        print_metric("Run ID", self.run.run_id)
        print_metric("Timestamp", self.run.timestamp)
        print_metric("System", f"{self.run.system_info.platform} (Python {self.run.system_info.python['version'].split()[0]})")
        
        # Indexing summary
        if self.run.indexing_results:
            print_section("Indexing Performance")
            
            table = Table(title="Indexing Results")
            table.add_column("Dataset", style="cyan")
            table.add_column("Files", justify="right")
            table.add_column("Chunks", justify="right")
            table.add_column("Time", justify="right")
            table.add_column("Files/s", justify="right")
            table.add_column("Chunks/s", justify="right")
            
            for size, result in sorted(self.run.indexing_results.items()):
                table.add_row(
                    size.upper(),
                    f"{result.total_files:,}",
                    f"{result.total_chunks:,}",
                    format_time(result.total_time),
                    f"{result.files_per_second:.1f}",
                    f"{result.chunks_per_second:.1f}"
                )
            
            console.print(table)
        
        # Search summary
        if self.run.search_results:
            print_section("Search Performance")
            
            table = Table(title="Search Results (Queries/Second)")
            table.add_column("Dataset", style="cyan")
            table.add_column("Language", style="cyan")
            table.add_column("Queries", justify="right")
            
            # Add columns for each top_k value
            top_k_values = set()
            for result in self.run.search_results.values():
                top_k_values.update(result.top_k_values)
            
            for k in sorted(top_k_values):
                table.add_column(f"k={k}", justify="right")
            
            for key, result in sorted(self.run.search_results.items()):
                dataset, language = key.rsplit("_", 1)
                row = [
                    dataset.upper(),
                    language.capitalize(),
                    str(result.total_queries)
                ]
                
                for k in sorted(top_k_values):
                    qps = result.queries_per_second.get(k, 0)
                    row.append(f"{qps:.1f}")
                
                table.add_row(*row)
            
            console.print(table)
    
    def save_reports(self, output_dir: Path) -> Dict[str, Path]:
        """Save all report formats to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save text report
        text_file = output_dir / f"report_{self.run.run_id}.txt"
        text_file.write_text(self.generate_text_report(), encoding="utf-8")
        saved_files["text"] = text_file
        
        # Save JSON report
        json_file = output_dir / f"report_{self.run.run_id}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.generate_json_report(), f, indent=2, ensure_ascii=False)
        saved_files["json"] = json_file
        
        return saved_files


def generate_comparison_report(runs: List[BenchmarkRun]) -> str:
    """Generate a comparison report between multiple runs."""
    lines = []
    
    lines.append("=" * 80)
    lines.append("BENCHMARK COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Run information
    lines.append("Runs Compared:")
    for i, run in enumerate(runs):
        lines.append(f"  {i+1}. {run.run_id} ({run.timestamp})")
        lines.append(f"     System: {run.system_info.platform}")
    lines.append("")
    
    # Indexing comparison
    if any(run.indexing_results for run in runs):
        lines.append("-" * 80)
        lines.append("INDEXING PERFORMANCE COMPARISON")
        lines.append("-" * 80)
        lines.append("")
        
        # Get all dataset sizes
        all_sizes = set()
        for run in runs:
            all_sizes.update(run.indexing_results.keys())
        
        for size in sorted(all_sizes):
            lines.append(f"{size.upper()} Dataset:")
            lines.append("  Files/second:")
            
            for i, run in enumerate(runs):
                if size in run.indexing_results:
                    result = run.indexing_results[size]
                    lines.append(f"    Run {i+1}: {result.files_per_second:.2f}")
                else:
                    lines.append(f"    Run {i+1}: N/A")
            
            # Calculate improvement
            values = []
            for run in runs:
                if size in run.indexing_results:
                    values.append(run.indexing_results[size].files_per_second)
            
            if len(values) >= 2:
                improvement = ((values[-1] - values[0]) / values[0]) * 100
                lines.append(f"    Change: {improvement:+.1f}%")
            
            lines.append("")
    
    # Search comparison
    if any(run.search_results for run in runs):
        lines.append("-" * 80)
        lines.append("SEARCH PERFORMANCE COMPARISON")
        lines.append("-" * 80)
        lines.append("")
        
        # Get all search configurations
        all_configs = set()
        for run in runs:
            all_configs.update(run.search_results.keys())
        
        for config in sorted(all_configs):
            lines.append(f"{config}:")
            
            # Get common top_k values
            common_k = None
            for run in runs:
                if config in run.search_results:
                    if common_k is None:
                        common_k = set(run.search_results[config].top_k_values)
                    else:
                        common_k &= set(run.search_results[config].top_k_values)
            
            if common_k:
                for k in sorted(common_k):
                    lines.append(f"  Queries/second (k={k}):")
                    
                    values = []
                    for i, run in enumerate(runs):
                        if config in run.search_results:
                            qps = run.search_results[config].queries_per_second.get(k, 0)
                            lines.append(f"    Run {i+1}: {qps:.2f}")
                            values.append(qps)
                        else:
                            lines.append(f"    Run {i+1}: N/A")
                    
                    if len(values) >= 2:
                        improvement = ((values[-1] - values[0]) / values[0]) * 100
                        lines.append(f"    Change: {improvement:+.1f}%")
            
            lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF COMPARISON")
    lines.append("=" * 80)
    
    return "\n".join(lines)
