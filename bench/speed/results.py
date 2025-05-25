"""Data classes and storage for benchmark results."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench.config import RESULTS_DIR
from bench.utils import get_python_info, get_timestamp, load_json, save_json


@dataclass
class SystemInfo:
    """System information at benchmark time."""
    
    python: Dict[str, str]
    oboyu_version: str
    cpu_count: int
    memory_gb: float
    platform: str
    timestamp: str = field(default_factory=get_timestamp)
    
    @classmethod
    def gather(cls) -> "SystemInfo":
        """Gather current system information."""
        import platform

        import psutil
        
        try:
            import oboyu
            oboyu_version = getattr(oboyu, "__version__", "unknown")
        except Exception:
            oboyu_version = "unknown"
        
        return cls(
            python=get_python_info(),
            oboyu_version=oboyu_version,
            cpu_count=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            platform=platform.platform()
        )


@dataclass
class TimingResult:
    """Timing result for a single operation."""
    
    name: str
    value: float  # seconds
    unit: str = "seconds"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """General metric result."""
    
    name: str
    value: float
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexingResult:
    """Results from indexing benchmark."""
    
    dataset_size: str
    total_files: int
    total_chunks: int
    total_time: float
    file_discovery_time: float
    content_extraction_time: float
    embedding_generation_time: float
    database_storage_time: float
    files_per_second: float
    chunks_per_second: float
    memory_usage: Dict[str, float] = field(default_factory=dict)
    system_metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Results from search benchmark."""
    
    query_id: str
    query_text: str
    response_time: float
    result_count: int
    top_k: int
    first_result_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchBenchmarkResult:
    """Aggregated results from search benchmark."""
    
    dataset_size: str
    query_language: str
    total_queries: int
    top_k_values: List[int]
    response_times: Dict[int, List[float]]  # top_k -> list of times
    queries_per_second: Dict[int, float]  # top_k -> qps
    statistics: Dict[int, Dict[str, float]]  # top_k -> stats
    memory_usage: Dict[str, float] = field(default_factory=dict)
    system_metrics: Dict[str, float] = field(default_factory=dict)
    individual_results: List[SearchResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    """Complete benchmark run with all results."""
    
    run_id: str
    timestamp: str
    system_info: SystemInfo
    config: Dict[str, Any]
    indexing_results: Dict[str, IndexingResult] = field(default_factory=dict)  # size -> result
    search_results: Dict[str, SearchBenchmarkResult] = field(default_factory=dict)  # key -> result
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, output_dir: Path = RESULTS_DIR) -> Path:
        """Save benchmark run to file."""
        filename = f"benchmark_run_{self.run_id}.json"
        filepath = output_dir / filename
        
        # Convert to dict
        data = asdict(self)
        
        # Save
        save_json(data, filepath)
        
        # Also save a summary file for quick access
        summary_file = output_dir / f"summary_{self.run_id}.json"
        summary = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "system": {
                "platform": self.system_info.platform,
                "python": self.system_info.python["version"].split()[0],
                "oboyu": self.system_info.oboyu_version
            },
            "indexing": {
                size: {
                    "total_time": result.total_time,
                    "files_per_second": result.files_per_second,
                    "total_files": result.total_files
                }
                for size, result in self.indexing_results.items()
            },
            "search": {
                key: {
                    "total_queries": result.total_queries,
                    "avg_response_time": sum(sum(times) for times in result.response_times.values()) /
                                        sum(len(times) for times in result.response_times.values())
                }
                for key, result in self.search_results.items()
            }
        }
        save_json(summary, summary_file)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> "BenchmarkRun":
        """Load benchmark run from file."""
        data = load_json(filepath)
        
        # Convert nested dicts back to dataclasses
        system_info = SystemInfo(**data["system_info"])
        
        indexing_results = {
            size: IndexingResult(**result)
            for size, result in data.get("indexing_results", {}).items()
        }
        
        search_results = {}
        for key, result in data.get("search_results", {}).items():
            # Convert individual results
            individual_results = [
                SearchResult(**r) for r in result.get("individual_results", [])
            ]
            result["individual_results"] = individual_results
            search_results[key] = SearchBenchmarkResult(**result)
        
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            system_info=system_info,
            config=data["config"],
            indexing_results=indexing_results,
            search_results=search_results,
            metadata=data.get("metadata", {})
        )


class ResultsManager:
    """Manage benchmark results storage and retrieval."""
    
    def __init__(self, results_dir: Path = RESULTS_DIR) -> None:
        """Initialize results manager."""
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all benchmark runs."""
        runs = []
        
        for summary_file in self.results_dir.glob("summary_*.json"):
            try:
                summary = load_json(summary_file)
                runs.append(summary)
            except Exception:
                continue  # Skip invalid files
        
        # Sort by timestamp
        runs.sort(key=lambda x: x["timestamp"], reverse=True)
        return runs
    
    def get_run(self, run_id: str) -> Optional[BenchmarkRun]:
        """Get a specific benchmark run."""
        filepath = self.results_dir / f"benchmark_run_{run_id}.json"
        if filepath.exists():
            return BenchmarkRun.load(filepath)
        return None
    
    def get_latest_run(self) -> Optional[BenchmarkRun]:
        """Get the most recent benchmark run."""
        runs = self.list_runs()
        if runs:
            return self.get_run(runs[0]["run_id"])
        return None
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple benchmark runs."""
        runs = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                runs.append(run)
        
        if len(runs) < 2:
            return {"error": "Need at least 2 runs to compare"}
        
        comparison = {
            "runs": [
                {
                    "run_id": run.run_id,
                    "timestamp": run.timestamp,
                    "system": run.system_info.platform
                }
                for run in runs
            ],
            "indexing": {},
            "search": {}
        }
        
        # Compare indexing results
        for size in runs[0].indexing_results.keys():
            if all(size in run.indexing_results for run in runs):
                comparison["indexing"][size] = {
                    "total_time": [run.indexing_results[size].total_time for run in runs],
                    "files_per_second": [run.indexing_results[size].files_per_second for run in runs]
                }
        
        # Compare search results
        for key in runs[0].search_results.keys():
            if all(key in run.search_results for run in runs):
                all_response_times = []
                for run in runs:
                    times = []
                    for top_k_times in run.search_results[key].response_times.values():
                        times.extend(top_k_times)
                    avg_time = sum(times) / len(times) if times else 0
                    all_response_times.append(avg_time)
                
                comparison["search"][key] = {
                    "avg_response_time": all_response_times
                }
        
        return comparison
