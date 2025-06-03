#!/usr/bin/env python3
"""Benchmark search performance for Oboyu."""

import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.progress import track

from bench.config import BENCHMARK_CONFIG, QUERIES_DIR
from bench.speed.results import SearchBenchmarkResult, SearchResult
from bench.utils import SystemMonitor, Timer, calculate_statistics, print_metric, print_section
from oboyu.config.indexer import IndexerConfig, ProcessingConfig
from oboyu.retriever import Retriever

console = Console()


class SearchBenchmark:
    """Benchmark for measuring search performance."""
    
    def __init__(
        self,
        db_path: Path,
        query_language: str = "japanese",
        queries_dir: Path = QUERIES_DIR
    ) -> None:
        """Initialize search benchmark."""
        self.db_path = db_path
        self.query_language = query_language
        self.queries_dir = queries_dir
        self.config = BENCHMARK_CONFIG["search"]
        
        # Load queries
        self.queries = self._load_queries()
        
        # Initialize retriever with proper config
        config = IndexerConfig(
            processing=ProcessingConfig(db_path=db_path)
        )
        self.retriever = Retriever(config=config)
        
        # Get dataset info
        self.dataset_info = self._get_dataset_info()
    
    def _load_queries(self) -> List[Dict[str, str]]:
        """Load queries for the specified language."""
        query_file = self.queries_dir / f"{self.query_language}_queries.json"
        if not query_file.exists():
            raise ValueError(f"Query file not found: {query_file}")
        
        with open(query_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _get_dataset_info(self) -> Dict[str, int]:
        """Get information about the indexed dataset."""
        # Use the database service from retriever
        db_service = self.retriever.database_service
        conn = db_service.conn
        
        # Count documents and chunks
        doc_count = conn.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        
        return {
            "document_count": doc_count,
            "chunk_count": chunk_count
        }
    
    def _run_single_search(
        self,
        query_text: str,
        top_k: int,
        warm_cache: bool = False
    ) -> Tuple[float, int, Optional[float]]:
        """Run a single search query and return timing metrics."""
        # Generate embedding for query
        if not warm_cache:
            gc.collect()  # Force garbage collection for consistent timing
        
        with Timer("total_search") as total_timer:
            # Use retriever to perform search
            with Timer("vector_search") as search_timer:
                results = self.retriever.search(
                    query=query_text,
                    limit=top_k,
                    mode="vector"  # Use vector search for benchmarking
                )
            
            first_result_time = search_timer.elapsed if results else None
        
        return total_timer.elapsed, len(results), first_result_time
    
    def run(
        self,
        test_queries: Optional[int] = None,
        monitor_system: bool = True
    ) -> SearchBenchmarkResult:
        """Run the search benchmark."""
        print_section(f"Search Benchmark - {self.query_language} queries")
        console.print(f"Database: {self.db_path}")
        console.print(f"Documents: {self.dataset_info['document_count']:,}")
        console.print(f"Chunks: {self.dataset_info['chunk_count']:,}")
        console.print(f"Queries: {len(self.queries)}")
        
        # Limit queries if specified
        queries_to_test = self.queries[:test_queries] if test_queries else self.queries
        
        # System monitoring
        monitor = SystemMonitor() if monitor_system else None
        
        # Results storage
        response_times: Dict[int, List[float]] = {k: [] for k in self.config["top_k_values"]}
        individual_results: List[SearchResult] = []
        
        # Warmup runs
        if self.config["warmup_runs"] > 0:
            console.print(f"\nRunning {self.config['warmup_runs']} warmup queries...")
            warmup_queries = queries_to_test[:self.config["warmup_runs"]]
            for query in warmup_queries:
                self._run_single_search(query["text"], top_k=5, warm_cache=True)
        
        # Test runs
        console.print(f"\nRunning {len(queries_to_test)} test queries...")
        
        if monitor:
            monitor.start()
        
        # Test each top_k value
        for top_k in self.config["top_k_values"]:
            console.print(f"\n[bold]Testing top_k={top_k}[/bold]")
            
            for query in track(queries_to_test, description=f"Running queries (k={top_k})"):
                # Run search
                response_time, result_count, first_result_time = self._run_single_search(
                    query["text"],
                    top_k=top_k
                )
                
                # Store results
                response_times[top_k].append(response_time)
                
                # Create individual result
                result = SearchResult(
                    query_id=query["id"],
                    query_text=query["text"],
                    response_time=response_time,
                    result_count=result_count,
                    top_k=top_k,
                    first_result_time=first_result_time,
                    metadata=query.get("metadata", {})
                )
                individual_results.append(result)
                
                # Sample system metrics
                if monitor and len(response_times[top_k]) % 10 == 0:
                    monitor.samples.append(monitor.sample())
            
            # Print statistics for this top_k
            stats = calculate_statistics(response_times[top_k])
            print_metric("Average response time", stats["mean"], "time")
            print_metric("Median response time", stats["median"], "time")
            print_metric("95th percentile", stats["p95"], "time")
        
        if monitor:
            monitor.stop()
        
        # Calculate queries per second for each top_k
        queries_per_second = {}
        statistics = {}
        
        for top_k, times in response_times.items():
            if times:
                total_time = sum(times)
                queries_per_second[top_k] = len(times) / total_time
                statistics[top_k] = calculate_statistics(times)
        
        # Get dataset size from db path name
        dataset_size = "unknown"
        if "small" in str(self.db_path):
            dataset_size = "small"
        elif "medium" in str(self.db_path):
            dataset_size = "medium"
        elif "large" in str(self.db_path):
            dataset_size = "large"
        
        # Create result
        result = SearchBenchmarkResult(
            dataset_size=dataset_size,
            query_language=self.query_language,
            total_queries=len(queries_to_test),
            top_k_values=self.config["top_k_values"],
            response_times=response_times,
            queries_per_second=queries_per_second,
            statistics=statistics,
            individual_results=individual_results,
            metadata={
                "dataset_info": self.dataset_info,
                "warmup_runs": self.config["warmup_runs"],
                "test_runs": len(queries_to_test)
            }
        )
        
        # Add system metrics
        if monitor:
            result.system_metrics = monitor.get_summary()
        
        return result
    
    def close(self) -> None:
        """Close database connection."""
        # Retriever handles cleanup automatically
        pass


def benchmark_search(
    db_paths: Dict[str, Path],
    query_languages: List[str],
    queries_dir: Path = QUERIES_DIR,
    test_queries: Optional[int] = None,
    monitor_system: bool = True
) -> Dict[str, SearchBenchmarkResult]:
    """Run search benchmarks for multiple configurations."""
    results = {}
    
    for dataset_name, db_path in db_paths.items():
        if not db_path.exists():
            console.print(f"[yellow]Warning: Database not found for {dataset_name}: {db_path}[/yellow]")
            continue
        
        for language in query_languages:
            key = f"{dataset_name}_{language}"
            console.print(f"\n[bold blue]Benchmarking {key}[/bold blue]")
            
            try:
                benchmark = SearchBenchmark(db_path, language, queries_dir)
                result = benchmark.run(test_queries, monitor_system)
                results[key] = result
                benchmark.close()
                
                # Print summary
                print_section(f"Summary - {key}")
                for top_k, qps in result.queries_per_second.items():
                    print_metric(f"QPS (top_k={top_k})", f"{qps:.2f}")
                
            except Exception as e:
                console.print(f"[red]Error benchmarking {key}: {e}[/red]")
                continue
    
    return results


if __name__ == "__main__":
    # Test with a dummy database
    import tempfile

    from bench.speed.benchmark_indexing import IndexingBenchmark
    
    # Create a small test index
    console.print("Creating test index...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Generate small dataset
        from bench.speed.generate_test_data import generate_dataset
        data_dir = temp_path / "data"
        generate_dataset("small", data_dir / "small", clean=True)
        
        # Index it
        indexing = IndexingBenchmark("small", data_dir)
        db_path = indexing._create_temp_db()
        
        # Run a simple indexing (simplified for testing)
        console.print("Indexing test data...")
        # ... indexing code would go here ...
        
        # Run search benchmark
        results = benchmark_search(
            {"test": db_path},
            ["english"],
            test_queries=5
        )
        
        for key, result in results.items():
            console.print(f"\n[bold]Results for {key}:[/bold]")
            print_metric("Total queries", result.total_queries)
            for top_k, qps in result.queries_per_second.items():
                print_metric(f"QPS (top_k={top_k})", f"{qps:.2f}")
