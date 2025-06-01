"""Speed benchmarking module for Oboyu.

This module contains tools for measuring and analyzing the speed performance
of Oboyu's indexing and search operations.
"""

from bench.speed.results import BenchmarkRun, ResultsManager
from bench.speed.runner import BenchmarkRunner

__all__ = ["BenchmarkRunner", "BenchmarkRun", "ResultsManager"]
