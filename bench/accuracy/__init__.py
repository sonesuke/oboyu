"""Accuracy evaluation benchmarks for Oboyu.

This module provides comprehensive accuracy evaluation for Oboyu as a RAG system,
with special focus on Japanese document search performance.
"""

# Defer imports to avoid circular dependency issues
__all__ = [
    "run_accuracy_benchmark",
    "DatasetManager",
    "RAGEvaluationConfig",
    "RAGEvaluator",
    "ResultsAnalyzer",
]
