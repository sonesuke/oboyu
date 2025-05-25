"""Speed benchmarking module for Oboyu.

This module contains tools for measuring and analyzing the speed performance
of Oboyu's indexing and search operations.
"""

from .runner import BenchmarkRunner
from .results import BenchmarkRun, ResultsManager

__all__ = ["BenchmarkRunner", "BenchmarkRun", "ResultsManager"]