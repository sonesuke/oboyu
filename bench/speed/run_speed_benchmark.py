#!/usr/bin/env python3
"""Main entry point for running Oboyu speed benchmarks."""

import argparse
import sys
from pathlib import Path

# Add bench parent directory to path for importing bench modules
bench_parent = Path(__file__).parent.parent
sys.path.insert(0, str(bench_parent.parent))
sys.path.insert(0, str(bench_parent))

# Add src directory to path for importing oboyu modules
src_path = Path(__file__).parent.parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from rich.console import Console

from bench.config import DATA_DIR, DATASET_SIZES, QUERIES_DIR, RESULTS_DIR, get_query_languages
from bench.speed.analyze import BenchmarkAnalyzer
from bench.speed.runner import BenchmarkRunner
from bench.utils import check_oboyu_installation, print_header

console = Console()


def main() -> None:
    """Run speed benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run Oboyu performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with small dataset
  python bench/run_speed_benchmark.py --datasets small
  
  # Run benchmark with multiple datasets
  python bench/run_speed_benchmark.py --datasets small medium
  
  # Skip indexing benchmarks
  python bench/run_speed_benchmark.py --datasets small --skip-indexing
  
  # Analyze results after running
  python bench/run_speed_benchmark.py --analyze-only
        """
    )
    
    # Benchmark mode
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing results"
    )
    
    # Dataset selection
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_SIZES.keys()),
        help="Dataset sizes to benchmark"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=get_query_languages(),
        help="Query languages to test"
    )
    
    # Benchmark options
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip indexing benchmarks"
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip search benchmarks"
    )
    parser.add_argument(
        "--use-existing-indexes",
        action="store_true",
        help="Use existing test indexes instead of creating new ones"
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of test data and queries"
    )
    
    # Directory options
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory for test datasets"
    )
    parser.add_argument(
        "--queries-dir",
        type=Path,
        default=QUERIES_DIR,
        help="Directory for query datasets"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for benchmark results"
    )
    
    args = parser.parse_args()
    
    # Check Oboyu installation
    if not check_oboyu_installation():
        console.print("[red]Error: Oboyu is not installed[/red]")
        console.print("Please install Oboyu first: pip install -e .")
        sys.exit(1)
    
    # Analyze only mode
    if args.analyze_only:
        print_header("Benchmark Analysis")
        analyzer = BenchmarkAnalyzer(args.results_dir)
        
        runs_info = analyzer.results_manager.list_runs()
        if not runs_info:
            console.print("[yellow]No benchmark results found[/yellow]")
            sys.exit(0)
        
        # Load and analyze latest runs
        runs = []
        for info in runs_info[:5]:  # Latest 5 runs
            run = analyzer.results_manager.get_run(info["run_id"])
            if run:
                runs.append(run)
        
        runs.reverse()  # Chronological order
        analyzer.print_analysis(runs)
        sys.exit(0)
    
    # Determine datasets and languages
    datasets = args.datasets or ["small"]
    languages = args.languages or ["english", "japanese"]
    
    # Create and run benchmark
    try:
        runner = BenchmarkRunner(
            dataset_sizes=datasets,
            query_languages=languages,
            data_dir=args.data_dir,
            queries_dir=args.queries_dir,
            results_dir=args.results_dir
        )
        
        run = runner.run(
            force_regenerate=args.force_regenerate,
            skip_indexing=args.skip_indexing,
            skip_search=args.skip_search,
            use_existing_indexes=args.use_existing_indexes
        )
        
        console.print("\n[green]✨ Benchmark completed successfully![/green]")
        console.print(f"Run ID: {run.run_id}")
        
        # Offer to analyze results
        console.print("\nTo analyze results, run:")
        console.print("  python bench/analyze.py --latest 5")
        console.print(f"  python bench/analyze.py --compare {run.run_id} <previous_run_id>")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
