"""Benchmark command for Oboyu CLI.

This module provides the benchmark command that integrates with the comprehensive
benchmark infrastructure for performance evaluation and regression detection.
"""

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from oboyu.cli.common_options import VerboseOption

# Add bench directory to path for imports
bench_path = Path(__file__).parent.parent.parent.parent / "bench"
if bench_path.exists():
    sys.path.insert(0, str(bench_path))

# Import benchmark modules
try:
    from bench.benchmark_runner import BenchmarkRunner
    from bench.logger import BenchmarkLogger
    from bench.utils import check_oboyu_installation

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# Create Typer app for benchmark commands
app = typer.Typer(
    name="benchmark",
    help="Run performance benchmarks and generate reports",
    add_completion=False,
)

console = Console()


@app.command("run")
def run_benchmark(
    ctx: typer.Context,
    suite: str = typer.Argument(
        default="quick",
        help="Benchmark suite to run (quick, comprehensive, speed_only, accuracy_only, all)",
    ),
    datasets: Optional[List[str]] = typer.Option(
        None,
        "--datasets",
        help="Dataset sizes for speed benchmarks (small, medium, large)",
    ),
    accuracy_datasets: Optional[List[str]] = typer.Option(
        None,
        "--accuracy-datasets",
        help="Datasets for accuracy evaluation (synthetic, miracl-ja, mldr-ja, jagovfaqs-22k, jacwir)",
    ),
    search_modes: Optional[List[str]] = typer.Option(
        None,
        "--search-modes",
        help="Search modes to evaluate (vector, bm25, hybrid)",
    ),
    models: Optional[List[str]] = typer.Option(
        None,
        "--models",
        help="Reranker models to evaluate (small, large)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Custom output directory for results",
    ),
    formats: Optional[List[str]] = typer.Option(
        ["json", "txt", "html"],
        "--formats",
        help="Report formats to generate (json, txt, html, md)",
    ),
    include_visualizations: bool = typer.Option(
        True,
        "--visualizations/--no-visualizations",
        help="Include visualizations in reports",
    ),
    verbose: VerboseOption = False,
) -> None:
    """Run benchmark suite and generate comprehensive reports."""
    if not BENCHMARK_AVAILABLE:
        console.print("❌ Benchmark infrastructure not available", style="red")
        console.print("Make sure you're running from the project root directory", style="yellow")
        raise typer.Exit(1)

    # Check if Oboyu is properly installed
    if not check_oboyu_installation():
        console.print("❌ Oboyu not properly installed for benchmarking", style="red")
        raise typer.Exit(1)

    # Create benchmark logger
    logger = BenchmarkLogger(verbose=verbose)

    # Create benchmark runner
    runner = BenchmarkRunner(
        output_dir=output_dir,
        logger=logger,
    )

    # Prepare keyword arguments for benchmark execution
    kwargs = {}
    if datasets:
        kwargs["datasets"] = datasets
    if accuracy_datasets:
        kwargs["datasets"] = accuracy_datasets  # Use for accuracy benchmarks
    if search_modes:
        kwargs["search_modes"] = search_modes
    if models:
        kwargs["models"] = models
    if verbose:
        kwargs["verbose"] = verbose

    try:
        console.print(f"🚀 Running benchmark suite: {suite}", style="bold blue")

        # Run the benchmark suite
        result = runner.run_suite(suite, **kwargs)

        # Generate reports if requested
        if formats:
            console.print("📊 Generating reports...", style="blue")
            try:
                from bench.reports import generate_report

                generate_report(
                    results=result.individual_results,
                    title=f"Benchmark Report - {suite}",
                    formats=formats,
                    output_dir=output_dir,
                    include_visualizations=include_visualizations,
                )

                console.print("✅ Reports generated successfully", style="green")

            except ImportError:
                console.print("⚠️ Report generation not available - results saved as JSON only", style="yellow")

        # Print final status
        if result.benchmarks_failed == 0:
            console.print("🎉 All benchmarks completed successfully!", style="bold green")
        else:
            console.print(f"⚠️ {result.benchmarks_failed} benchmark(s) failed", style="bold yellow")

        # Exit with appropriate code
        raise typer.Exit(0 if result.benchmarks_failed == 0 else 1)

    except KeyboardInterrupt:
        console.print("\n❌ Benchmark interrupted by user", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Benchmark failed: {e}", style="red")
        if verbose:
            import traceback

            console.print(traceback.format_exc(), style="dim red")
        raise typer.Exit(1)


@app.command("list")
def list_suites() -> None:
    """List available benchmark suites and their descriptions."""
    if not BENCHMARK_AVAILABLE:
        console.print("❌ Benchmark infrastructure not available", style="red")
        raise typer.Exit(1)

    table = Table(title="Available Benchmark Suites")
    table.add_column("Suite", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Benchmarks", style="yellow")

    suites = {
        "quick": ("Quick evaluation with reduced scope", "speed, accuracy"),
        "comprehensive": ("Full evaluation with all benchmarks", "speed, accuracy, reranker"),
        "speed_only": ("Speed benchmarks only", "speed"),
        "accuracy_only": ("Accuracy benchmarks only", "accuracy"),
        "all": ("All available benchmarks", "speed, accuracy, reranker"),
    }

    for suite_name, (description, benchmarks) in suites.items():
        table.add_row(suite_name, description, benchmarks)

    console.print(table)


@app.command("status")
def benchmark_status(
    results_dir: Optional[Path] = typer.Option(
        None,
        "--results-dir",
        help="Directory containing benchmark results",
    ),
    latest: int = typer.Option(
        5,
        "--latest",
        help="Number of latest results to show",
    ),
) -> None:
    """Show status of recent benchmark runs."""
    if not BENCHMARK_AVAILABLE:
        console.print("❌ Benchmark infrastructure not available", style="red")
        raise typer.Exit(1)

    if results_dir is None:
        results_dir = Path("bench/results")

    if not results_dir.exists():
        console.print(f"❌ Results directory not found: {results_dir}", style="red")
        raise typer.Exit(1)

    # Find recent result files
    result_files = list(results_dir.glob("*.json"))
    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    if not result_files:
        console.print("No benchmark results found", style="yellow")
        return

    # Show latest results
    table = Table(title=f"Latest {min(latest, len(result_files))} Benchmark Results")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Suite/Type", style="white")
    table.add_column("Status", style="white")
    table.add_column("Duration", style="yellow")
    table.add_column("File", style="dim")

    import json

    for result_file in result_files[:latest]:
        try:
            with open(result_file, "r") as f:
                data = json.load(f)

            # Extract information based on file type
            if "suite_name" in data:  # Suite result
                timestamp = data.get("timestamp", "Unknown")
                suite_name = data.get("suite_name", "Unknown")
                successful = data.get("benchmarks_successful", 0)
                failed = data.get("benchmarks_failed", 0)
                duration = data.get("total_duration_seconds", 0)

                status = f"✅ {successful}" if failed == 0 else f"⚠️ {successful}/{successful + failed}"

            else:  # Individual result
                timestamp = data.get("timestamp", "Unknown")
                suite_name = data.get("benchmark_type", "Unknown")
                success = data.get("success", False)
                duration = data.get("duration_seconds", 0)

                status = "✅ PASS" if success else "❌ FAIL"

            table.add_row(
                timestamp,
                suite_name,
                status,
                f"{duration:.1f}s",
                result_file.name,
            )

        except Exception as e:
            table.add_row(
                "Unknown",
                "Invalid",
                f"❌ Error: {e}",
                "0s",
                result_file.name,
            )

    console.print(table)


@app.command("compare")
def compare_results(
    file1: Path = typer.Argument(help="First result file"),
    file2: Path = typer.Argument(help="Second result file"),
    regression_threshold: float = typer.Option(
        0.1,
        "--threshold",
        help="Regression threshold (0.1 = 10%)",
    ),
) -> None:
    """Compare two benchmark result files."""
    if not BENCHMARK_AVAILABLE:
        console.print("❌ Benchmark infrastructure not available", style="red")
        raise typer.Exit(1)

    if not file1.exists():
        console.print(f"❌ File not found: {file1}", style="red")
        raise typer.Exit(1)

    if not file2.exists():
        console.print(f"❌ File not found: {file2}", style="red")
        raise typer.Exit(1)

    try:
        import json

        # Load result files
        with open(file1, "r") as f:
            data1 = json.load(f)
        with open(file2, "r") as f:
            data2 = json.load(f)

        console.print(f"📊 Comparing {file1.name} vs {file2.name}", style="bold blue")

        # Simple comparison logic
        table = Table(title="Benchmark Comparison")
        table.add_column("Benchmark", style="cyan")
        table.add_column("File 1", style="white")
        table.add_column("File 2", style="white")
        table.add_column("Change", style="yellow")
        table.add_column("Status", style="white")

        # Extract comparable metrics
        def get_duration(data: dict) -> Optional[float]:
            if "total_duration_seconds" in data:
                return data["total_duration_seconds"]
            elif "duration_seconds" in data:
                return data["duration_seconds"]
            return None

        duration1 = get_duration(data1)
        duration2 = get_duration(data2)

        if duration1 is not None and duration2 is not None:
            change = duration2 - duration1
            change_percent = (change / duration1 * 100) if duration1 > 0 else 0

            if abs(change_percent) > regression_threshold * 100:
                if change > 0:
                    status = "⚠️ Regression"
                else:
                    status = "✅ Improvement"
            else:
                status = "➡️ Stable"

            table.add_row(
                "Duration",
                f"{duration1:.1f}s",
                f"{duration2:.1f}s",
                f"{change:+.1f}s ({change_percent:+.1f}%)",
                status,
            )

        console.print(table)

    except Exception as e:
        console.print(f"❌ Comparison failed: {e}", style="red")
        raise typer.Exit(1)


@app.command("report")
def generate_reports(
    results_dir: Path = typer.Argument(help="Directory containing benchmark results"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for reports",
    ),
    formats: List[str] = typer.Option(
        ["html"],
        "--formats",
        help="Report formats to generate",
    ),
    include_comparisons: bool = typer.Option(
        True,
        "--comparisons/--no-comparisons",
        help="Include time-series comparisons",
    ),
) -> None:
    """Generate reports from existing benchmark results."""
    if not BENCHMARK_AVAILABLE:
        console.print("❌ Benchmark infrastructure not available", style="red")
        raise typer.Exit(1)

    if not results_dir.exists():
        console.print(f"❌ Results directory not found: {results_dir}", style="red")
        raise typer.Exit(1)

    try:
        from bench.reports import generate_all_reports

        console.print(f"📊 Generating reports from {results_dir}", style="blue")

        reports = generate_all_reports(
            results_dir=results_dir,
            output_dir=output_dir,
            include_comparisons=include_comparisons,
        )

        console.print(f"✅ Generated {len(reports)} reports", style="green")

        if output_dir:
            console.print(f"📁 Reports saved to: {output_dir}", style="blue")

    except Exception as e:
        console.print(f"❌ Report generation failed: {e}", style="red")
        raise typer.Exit(1)


@app.command("clean")
def clean_results(
    results_dir: Optional[Path] = typer.Option(
        None,
        "--results-dir",
        help="Directory containing benchmark results",
    ),
    keep_latest: int = typer.Option(
        10,
        "--keep-latest",
        help="Number of latest results to keep",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be deleted without actually deleting",
    ),
) -> None:
    """Clean up old benchmark result files."""
    if results_dir is None:
        results_dir = Path("bench/results")

    if not results_dir.exists():
        console.print(f"❌ Results directory not found: {results_dir}", style="red")
        raise typer.Exit(1)

    # Find all result files
    result_files = list(results_dir.glob("*.json"))
    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    if len(result_files) <= keep_latest:
        console.print(f"No files to clean (found {len(result_files)}, keeping {keep_latest})", style="green")
        return

    files_to_delete = result_files[keep_latest:]

    if dry_run:
        console.print(f"Would delete {len(files_to_delete)} files:", style="yellow")
        for file in files_to_delete:
            console.print(f"  - {file.name}", style="dim")
    else:
        console.print(f"Deleting {len(files_to_delete)} old result files...", style="blue")

        deleted = 0
        for file in files_to_delete:
            try:
                file.unlink()
                deleted += 1
            except Exception as e:
                console.print(f"Failed to delete {file.name}: {e}", style="red")

        console.print(f"✅ Deleted {deleted} files", style="green")


if __name__ == "__main__":
    app()
