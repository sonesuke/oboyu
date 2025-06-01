"""Performance benchmark suite for Oboyu.

This package provides comprehensive benchmarking capabilities for Oboyu:
- Speed benchmarks for indexing and search performance
- Accuracy evaluation for RAG performance 
- Reranker-specific evaluation
- Unified benchmark runner

Examples:
    # Run all benchmarks quickly
    python bench/run_benchmarks.py all --quick
    
    # Run only speed benchmarks  
    python bench/run_benchmarks.py speed --datasets small
    
    # Run accuracy evaluation
    python bench/run_benchmarks.py accuracy --datasets synthetic

"""

# Defer imports to avoid circular dependency issues
__all__ = [
    "run_accuracy_benchmark",
    "run_reranker_benchmark",
    "run_speed_benchmark",
]
