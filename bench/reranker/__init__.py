"""Reranker evaluation benchmarks for Oboyu.

This module provides benchmarks specifically for evaluating reranker performance
and effectiveness in improving search result quality.
"""

from .benchmark_reranking import main as run_reranker_benchmark

__all__ = [
    "run_reranker_benchmark",
]