#!/usr/bin/env python3
"""Unified benchmark runner for Oboyu.

This is the main entry point for all Oboyu benchmarks, providing a clean
interface to run speed benchmarks, accuracy evaluations, and reranker tests.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add bench directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bench.logger import BenchmarkLogger
from bench.utils import check_oboyu_installation, print_header


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for Oboyu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks (quick mode)
  python bench/run_benchmarks.py all --quick

  # Run only speed benchmarks
  python bench/run_benchmarks.py speed --datasets small medium

  # Run only accuracy evaluation
  python bench/run_benchmarks.py accuracy --datasets synthetic

  # Run reranker-specific benchmarks
  python bench/run_benchmarks.py reranker --models small large

  # Run comprehensive evaluation
  python bench/run_benchmarks.py all --comprehensive
        """,
    )

    # Benchmark type selection
    parser.add_argument(
        "benchmark_type",
        choices=["all", "speed", "accuracy", "reranker"],
        help="Type of benchmark to run",
    )

    # Common options
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks with reduced scope")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive benchmarks with full scope")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--output-dir", type=Path, help="Custom output directory for results")

    # Speed benchmark options
    speed_group = parser.add_argument_group("Speed benchmark options")
    speed_group.add_argument(
        "--datasets",
        nargs="+",
        choices=["small", "medium", "large"],
        help="Dataset sizes for speed benchmarks",
    )
    speed_group.add_argument(
        "--languages",
        nargs="+",
        choices=["japanese", "english", "mixed"],
        help="Query languages for speed benchmarks",
    )

    # Accuracy benchmark options
    accuracy_group = parser.add_argument_group("Accuracy benchmark options")
    accuracy_group.add_argument(
        "--accuracy-datasets",
        nargs="+",
        choices=["synthetic", "miracl-ja", "mldr-ja", "jagovfaqs-22k", "jacwir"],
        help="Datasets for accuracy evaluation",
    )
    accuracy_group.add_argument(
        "--search-modes",
        nargs="+",
        choices=["vector", "bm25", "hybrid"],
        help="Search modes to evaluate",
    )

    # Reranker benchmark options
    reranker_group = parser.add_argument_group("Reranker benchmark options")
    reranker_group.add_argument(
        "--models",
        nargs="+",
        choices=["small", "large"],
        help="Reranker models to evaluate (small=ruri-reranker-small, large=ruri-v3-reranker-310m)",
    )

    return parser.parse_args()


def run_speed_benchmarks(args: argparse.Namespace, logger: BenchmarkLogger) -> bool:
    """Run speed benchmarks."""
    try:
        from bench.speed.run_speed_benchmark import main as run_speed_main

        logger.section("Running Speed Benchmarks")

        # Prepare arguments for speed benchmark
        speed_args = []

        # Dataset selection
        if args.quick:
            speed_args.extend(["--datasets", "small"])
        elif args.comprehensive:
            speed_args.extend(["--datasets", "small", "medium", "large"])
        elif args.datasets:
            speed_args.extend(["--datasets"] + args.datasets)

        # Language selection
        if args.languages:
            speed_args.extend(["--languages"] + args.languages)

        if args.verbose:
            speed_args.append("--verbose")

        if args.output_dir:
            speed_args.extend(["--output-dir", str(args.output_dir)])

        # Run speed benchmark with prepared arguments
        logger.info(f"Running speed benchmark with args: {' '.join(speed_args)}")
        return run_speed_main(speed_args)

    except Exception as e:
        logger.error(f"Speed benchmark failed: {e}")
        return False


def run_accuracy_benchmarks(args: argparse.Namespace, logger: BenchmarkLogger) -> bool:
    """Run accuracy benchmarks."""
    try:
        from bench.accuracy.benchmark_rag_accuracy import main as run_accuracy_main

        logger.section("Running Accuracy Benchmarks")

        # Prepare arguments for accuracy benchmark
        accuracy_args = []

        # Dataset selection
        if args.quick:
            accuracy_args.extend(["--datasets", "synthetic"])
        elif args.comprehensive:
            accuracy_args.extend(["--datasets", "synthetic", "miracl-ja", "mldr-ja"])
        elif args.accuracy_datasets:
            accuracy_args.extend(["--datasets"] + args.accuracy_datasets)

        # Search mode selection
        if args.search_modes:
            accuracy_args.extend(["--search-modes"] + args.search_modes)

        if args.verbose:
            accuracy_args.append("--verbose")

        # Run accuracy benchmark with prepared arguments
        logger.info(f"Running accuracy benchmark with args: {' '.join(accuracy_args)}")
        return run_accuracy_main(accuracy_args)

    except Exception as e:
        logger.error(f"Accuracy benchmark failed: {e}")
        return False


def run_reranker_benchmarks(args: argparse.Namespace, logger: BenchmarkLogger) -> bool:
    """Run reranker benchmarks."""
    try:
        from bench.reranker.benchmark_reranking import main as run_reranker_main

        logger.section("Running Reranker Benchmarks")

        # Prepare arguments for reranker benchmark
        reranker_args = []

        # Model selection
        model_mapping = {
            "small": "cl-nagoya/ruri-reranker-small",
            "large": "cl-nagoya/ruri-v3-reranker-310m",
        }

        if args.quick:
            reranker_args.extend(["--models", "cl-nagoya/ruri-reranker-small"])
        elif args.comprehensive:
            reranker_args.extend(["--models", "cl-nagoya/ruri-reranker-small", "cl-nagoya/ruri-v3-reranker-310m"])
        elif args.models:
            model_names = [model_mapping.get(m, m) for m in args.models]
            reranker_args.extend(["--models"] + model_names)

        if args.verbose:
            reranker_args.append("--verbose")

        # Run reranker benchmark with prepared arguments
        logger.info(f"Running reranker benchmark with args: {' '.join(reranker_args)}")
        return run_reranker_main(reranker_args)

    except Exception as e:
        logger.error(f"Reranker benchmark failed: {e}")
        return False


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    args = parse_args() if argv is None else parse_args(argv)
    logger = BenchmarkLogger(verbose=args.verbose)

    print_header("Oboyu Benchmark Suite")

    # Check if Oboyu is properly installed
    if not check_oboyu_installation():
        return 1

    # Track overall success
    success = True

    # Run benchmarks based on type
    if args.benchmark_type in ["all", "speed"]:
        if not run_speed_benchmarks(args, logger):
            success = False

    if args.benchmark_type in ["all", "accuracy"]:
        if not run_accuracy_benchmarks(args, logger):
            success = False

    if args.benchmark_type in ["all", "reranker"]:
        if not run_reranker_benchmarks(args, logger):
            success = False

    # Print final status
    if success:
        logger.success("All benchmarks completed successfully!")
        return 0
    else:
        logger.error("Some benchmarks failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
