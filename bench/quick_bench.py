#!/usr/bin/env python3
"""Quick benchmark runner for common operations.

This script provides shortcuts for the most common benchmark operations.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and handle errors."""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
        print(f"✅ {description} - Complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed with exit code {e.returncode}")
        return False

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quick benchmark operations")
    parser.add_argument(
        "operation",
        choices=["quick", "setup", "speed", "accuracy", "all", "analyze"],
        help="Operation to perform",
    )

    args = parser.parse_args()

    print("🚀 Oboyu Quick Benchmark Runner")
    print("=" * 40)

    success = True

    if args.operation == "setup":
        # Setup test data and queries
        success &= run_command(
            ["uv", "run", "python", "bench/speed/generate_test_data.py", "small"],
            "Generating small test dataset"
        )
        success &= run_command(
            ["uv", "run", "python", "bench/speed/generate_queries.py"],
            "Generating query datasets"
        )

    elif args.operation == "quick":
        # Run quick benchmark suite
        success &= run_command(
            ["uv", "run", "python", "bench/run_benchmarks.py", "all", "--quick"],
            "Running quick benchmark suite"
        )

    elif args.operation == "speed":
        # Run only speed benchmarks
        success &= run_command(
            ["uv", "run", "python", "bench/run_benchmarks.py", "speed", "--datasets", "small"],
            "Running speed benchmarks"
        )

    elif args.operation == "accuracy":
        # Run only accuracy benchmarks
        success &= run_command(
            ["uv", "run", "python", "bench/run_benchmarks.py", "accuracy", "--datasets", "synthetic"],
            "Running accuracy benchmarks"
        )

    elif args.operation == "all":
        # Run comprehensive benchmarks
        success &= run_command(
            ["uv", "run", "python", "bench/run_benchmarks.py", "all", "--comprehensive"],
            "Running comprehensive benchmark suite"
        )

    elif args.operation == "analyze":
        # Analyze latest results
        success &= run_command(
            ["uv", "run", "python", "bench/speed/analyze.py", "--latest", "3"],
            "Analyzing latest speed benchmark results"
        )

    if success:
        print("\n🎉 All operations completed successfully!")
        return 0
    else:
        print("\n💥 Some operations failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
