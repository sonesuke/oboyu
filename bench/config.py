"""Configuration for Oboyu performance benchmarks."""

from pathlib import Path
from typing import Any, Dict, List

# Base directory for all benchmark-related files
BENCH_DIR = Path(__file__).parent
DATA_DIR = BENCH_DIR / "data"
QUERIES_DIR = BENCH_DIR / "queries"
RESULTS_DIR = BENCH_DIR / "results"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
QUERIES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset size configurations
DATASET_SIZES = {
    "small": {
        "name": "Small Dataset",
        "file_count": 50,
        "files_per_type": {
            ".txt": 20,
            ".md": 15,
            ".py": 10,
            ".html": 5
        },
        "content_size_range": (1000, 5000)  # Characters per file
    },
    "medium": {
        "name": "Medium Dataset",
        "file_count": 1000,
        "files_per_type": {
            ".txt": 400,
            ".md": 300,
            ".py": 200,
            ".html": 100
        },
        "content_size_range": (2000, 10000)
    },
    "large": {
        "name": "Large Dataset",
        "file_count": 10000,
        "files_per_type": {
            ".txt": 4000,
            ".md": 3000,
            ".py": 2000,
            ".html": 1000
        },
        "content_size_range": (3000, 15000)
    }
}

# Query configurations
QUERY_CONFIG = {
    "japanese": {
        "count": 50,
        "types": {
            "technical": 15,    # Technical documentation queries
            "business": 15,     # Business document queries
            "general": 10,      # General knowledge queries
            "code": 10          # Code-related queries
        }
    },
    "english": {
        "count": 50,
        "types": {
            "technical": 15,
            "business": 15,
            "general": 10,
            "code": 10
        }
    },
    "mixed": {
        "count": 20,  # Mixed Japanese-English queries
        "types": {
            "technical": 10,
            "general": 10
        }
    }
}

# Benchmark configurations
BENCHMARK_CONFIG = {
    "indexing": {
        "warmup_runs": 1,
        "test_runs": 3,
        "metrics": [
            "total_time",
            "file_discovery_time",
            "content_extraction_time",
            "embedding_generation_time",
            "database_storage_time",
            "files_per_second",
            "chunks_per_second"
        ]
    },
    "search": {
        "warmup_runs": 5,
        "test_runs": 100,
        "top_k_values": [1, 5, 10, 20, 50],
        "metrics": [
            "response_time",
            "queries_per_second",
            "first_result_time",
            "total_results_time"
        ]
    }
}

# System monitoring configuration
MONITORING_CONFIG = {
    "enabled": True,
    "sample_interval": 0.1,  # seconds
    "metrics": [
        "cpu_percent",
        "memory_usage_mb",
        "disk_io_read_mb",
        "disk_io_write_mb"
    ]
}

# Output configuration
OUTPUT_CONFIG = {
    "formats": ["json", "text"],
    "pretty_print": True,
    "save_raw_data": True,
    "timestamp_format": "%Y%m%d_%H%M%S"
}

# Default Oboyu configurations for benchmarking
OBOYU_CONFIG = {
    "crawler": {
        "depth": 10,
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "follow_symlinks": False,
        "japanese_encodings": ["utf-8", "shift-jis", "euc-jp"],
        "max_workers": 4
    },
    "indexer": {
        "chunk_size": 300,
        "chunk_overlap": 75,
        "embedding_model": "cl-nagoya/ruri-v3-30m",
        "embedding_device": "cpu",
        "batch_size": 8
    },
    "query": {
        "default_mode": "vector",  # Only vector search for now
        "top_k": 5,
        "snippet_length": 160
    }
}

def get_dataset_config(size: str) -> Dict[str, Any]:
    """Get configuration for a specific dataset size."""
    if size not in DATASET_SIZES:
        raise ValueError(f"Invalid dataset size: {size}. Choose from: {list(DATASET_SIZES.keys())}")
    return DATASET_SIZES[size]

def get_query_languages() -> List[str]:
    """Get list of supported query languages."""
    return list(QUERY_CONFIG.keys())

def get_benchmark_metrics(benchmark_type: str) -> List[str]:
    """Get list of metrics for a specific benchmark type."""
    if benchmark_type not in BENCHMARK_CONFIG:
        raise ValueError(f"Invalid benchmark type: {benchmark_type}. Choose from: {list(BENCHMARK_CONFIG.keys())}")
    return BENCHMARK_CONFIG[benchmark_type]["metrics"]
