"""Accuracy evaluation benchmarks for Oboyu.

This module provides comprehensive accuracy evaluation for Oboyu as a RAG system,
with special focus on Japanese document search performance.
"""

from .benchmark_rag_accuracy import main as run_accuracy_benchmark
from .rag_accuracy import (
    DatasetManager,
    RAGEvaluationConfig, 
    RAGEvaluator,
    ResultsAnalyzer,
)

__all__ = [
    "run_accuracy_benchmark",
    "DatasetManager",
    "RAGEvaluationConfig",
    "RAGEvaluator", 
    "ResultsAnalyzer",
]