#!/usr/bin/env python3
"""Analyze and compare benchmark results."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

from bench.config import RESULTS_DIR
from bench.speed.reporter import generate_comparison_report
from bench.speed.results import BenchmarkRun, ResultsManager
from bench.utils import print_header, print_metric, print_section

console = Console()


class BenchmarkAnalyzer:
    """Analyze benchmark results for trends and regressions."""
    
    def __init__(self, results_dir: Path = RESULTS_DIR) -> None:
        """Initialize analyzer."""
        self.results_manager = ResultsManager(results_dir)
    
    def find_regressions(
        self,
        baseline_run: BenchmarkRun,
        current_run: BenchmarkRun,
        threshold: float = 0.1
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find performance regressions between runs."""
        regressions = {
            "indexing": [],
            "search": []
        }
        
        # Check indexing regressions
        for size in baseline_run.indexing_results:
            if size in current_run.indexing_results:
                baseline = baseline_run.indexing_results[size]
                current = current_run.indexing_results[size]
                
                # Check files/second regression
                baseline_fps = baseline.files_per_second
                current_fps = current.files_per_second
                
                if baseline_fps > 0:
                    change = (current_fps - baseline_fps) / baseline_fps
                    
                    if change < -threshold:
                        regressions["indexing"].append({
                            "dataset": size,
                            "metric": "files_per_second",
                            "baseline": baseline_fps,
                            "current": current_fps,
                            "change_percent": change * 100
                        })
        
        # Check search regressions
        for config in baseline_run.search_results:
            if config in current_run.search_results:
                baseline = baseline_run.search_results[config]
                current = current_run.search_results[config]
                
                # Check queries/second for each top_k
                for top_k in baseline.queries_per_second:
                    if top_k in current.queries_per_second:
                        baseline_qps = baseline.queries_per_second[top_k]
                        current_qps = current.queries_per_second[top_k]
                        
                        if baseline_qps > 0:
                            change = (current_qps - baseline_qps) / baseline_qps
                            
                            if change < -threshold:
                                regressions["search"].append({
                                    "config": config,
                                    "top_k": top_k,
                                    "metric": "queries_per_second",
                                    "baseline": baseline_qps,
                                    "current": current_qps,
                                    "change_percent": change * 100
                                })
        
        return regressions
    
    def calculate_trends(self, runs: List[BenchmarkRun]) -> Dict[str, Dict[str, List[float]]]:
        """Calculate performance trends across multiple runs."""
        trends = {
            "indexing": {},
            "search": {}
        }
        
        # Indexing trends
        all_sizes = set()
        for run in runs:
            all_sizes.update(run.indexing_results.keys())
        
        for size in all_sizes:
            trends["indexing"][f"{size}_files_per_second"] = []
            trends["indexing"][f"{size}_chunks_per_second"] = []
            
            for run in runs:
                if size in run.indexing_results:
                    result = run.indexing_results[size]
                    trends["indexing"][f"{size}_files_per_second"].append(result.files_per_second)
                    trends["indexing"][f"{size}_chunks_per_second"].append(result.chunks_per_second)
        
        # Search trends
        all_configs = set()
        for run in runs:
            all_configs.update(run.search_results.keys())
        
        for config in all_configs:
            # Get common top_k values
            common_k = None
            for run in runs:
                if config in run.search_results:
                    if common_k is None:
                        common_k = set(run.search_results[config].top_k_values)
                    else:
                        common_k &= set(run.search_results[config].top_k_values)
            
            if common_k:
                for k in common_k:
                    key = f"{config}_qps_k{k}"
                    trends["search"][key] = []
                    
                    for run in runs:
                        if config in run.search_results:
                            qps = run.search_results[config].queries_per_second.get(k, 0)
                            trends["search"][key].append(qps)
        
        return trends
    
    def print_analysis(self, runs: List[BenchmarkRun]) -> None:
        """Print analysis of benchmark runs."""
        if not runs:
            console.print("[yellow]No runs to analyze[/yellow]")
            return
        
        print_header("Benchmark Analysis")
        
        # Latest run summary
        latest = runs[-1]
        print_section("Latest Run Summary")
        print_metric("Run ID", latest.run_id)
        print_metric("Timestamp", latest.timestamp)
        print_metric("System", latest.system_info.platform)
        
        # Performance trends
        if len(runs) > 1:
            print_section("Performance Trends")
            trends = self.calculate_trends(runs)
            
            # Indexing trends
            if trends["indexing"]:
                console.print("\n[bold]Indexing Performance:[/bold]")
                table = Table()
                table.add_column("Metric", style="cyan")
                table.add_column("First", justify="right")
                table.add_column("Latest", justify="right")
                table.add_column("Change", justify="right")
                table.add_column("Trend", justify="center")
                
                for metric, values in sorted(trends["indexing"].items()):
                    if len(values) >= 2:
                        first = values[0]
                        latest = values[-1]
                        change = ((latest - first) / first * 100) if first > 0 else 0
                        trend = "📈" if change > 5 else "📉" if change < -5 else "➡️"
                        
                        table.add_row(
                            metric,
                            f"{first:.2f}",
                            f"{latest:.2f}",
                            f"{change:+.1f}%",
                            trend
                        )
                
                console.print(table)
            
            # Search trends
            if trends["search"]:
                console.print("\n[bold]Search Performance:[/bold]")
                table = Table()
                table.add_column("Metric", style="cyan")
                table.add_column("First", justify="right")
                table.add_column("Latest", justify="right")
                table.add_column("Change", justify="right")
                table.add_column("Trend", justify="center")
                
                for metric, values in sorted(trends["search"].items()):
                    if len(values) >= 2:
                        first = values[0]
                        latest = values[-1]
                        change = ((latest - first) / first * 100) if first > 0 else 0
                        trend = "📈" if change > 5 else "📉" if change < -5 else "➡️"
                        
                        table.add_row(
                            metric,
                            f"{first:.2f}",
                            f"{latest:.2f}",
                            f"{change:+.1f}%",
                            trend
                        )
                
                console.print(table)
        
        # Regression detection
        if len(runs) >= 2:
            print_section("Regression Detection")
            baseline = runs[-2]
            current = runs[-1]
            
            regressions = self.find_regressions(baseline, current, threshold=0.05)
            
            if regressions["indexing"] or regressions["search"]:
                console.print("[red]⚠️  Performance regressions detected![/red]\n")
                
                if regressions["indexing"]:
                    console.print("[bold]Indexing Regressions:[/bold]")
                    for reg in regressions["indexing"]:
                        console.print(
                            f"  • {reg['dataset']} dataset: "
                            f"{reg['metric']} decreased by {abs(reg['change_percent']):.1f}% "
                            f"({reg['baseline']:.2f} → {reg['current']:.2f})"
                        )
                
                if regressions["search"]:
                    console.print("\n[bold]Search Regressions:[/bold]")
                    for reg in regressions["search"]:
                        console.print(
                            f"  • {reg['config']} (k={reg['top_k']}): "
                            f"{reg['metric']} decreased by {abs(reg['change_percent']):.1f}% "
                            f"({reg['baseline']:.2f} → {reg['current']:.2f})"
                        )
            else:
                console.print("[green]✓ No significant regressions detected[/green]")


def main() -> None:
    """Analyze benchmark results."""
    parser = argparse.ArgumentParser(description="Analyze Oboyu benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=5,
        help="Number of latest runs to analyze"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        help="Compare two specific runs by ID"
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.1,
        help="Threshold for regression detection (default: 10%%)"
    )
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer(args.results_dir)
    
    if args.compare:
        # Compare specific runs
        run1 = analyzer.results_manager.get_run(args.compare[0])
        run2 = analyzer.results_manager.get_run(args.compare[1])
        
        if not run1 or not run2:
            console.print("[red]Error: Could not load specified runs[/red]")
            return
        
        print_header(f"Comparing {args.compare[0]} vs {args.compare[1]}")
        
        # Generate comparison report
        report = generate_comparison_report([run1, run2])
        console.print(report)
        
        # Check for regressions
        regressions = analyzer.find_regressions(run1, run2, args.regression_threshold)
        if regressions["indexing"] or regressions["search"]:
            console.print("\n[red]⚠️  Regressions detected![/red]")
    
    else:
        # Analyze latest runs
        runs_info = analyzer.results_manager.list_runs()
        
        if not runs_info:
            console.print("[yellow]No benchmark runs found[/yellow]")
            return
        
        # Load latest runs
        runs = []
        for info in runs_info[:args.latest]:
            run = analyzer.results_manager.get_run(info["run_id"])
            if run:
                runs.append(run)
        
        # Reverse to get chronological order
        runs.reverse()
        
        # Analyze
        analyzer.print_analysis(runs)


if __name__ == "__main__":
    main()
