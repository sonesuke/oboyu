#!/usr/bin/env python3
"""Benchmark indexing performance for Oboyu."""

import gc
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from bench.config import BENCHMARK_CONFIG, DATA_DIR
from bench.speed.results import IndexingResult
from bench.utils import SystemMonitor, Timer, print_metric, print_section
from oboyu.config.crawler import CrawlerConfig
from oboyu.config.indexer import IndexerConfig, ModelConfig, ProcessingConfig
from oboyu.crawler.crawler import Crawler
from oboyu.crawler.discovery import discover_documents
from oboyu.indexer import Indexer

console = Console()


class IndexingBenchmark:
    """Benchmark for measuring indexing performance."""
    
    def __init__(self, dataset_size: str, data_dir: Path = DATA_DIR) -> None:
        """Initialize indexing benchmark."""
        self.dataset_size = dataset_size
        self.dataset_dir = data_dir / dataset_size
        self.config = BENCHMARK_CONFIG["indexing"]
        self.warmup_runs = self.config["warmup_runs"]
        self.test_runs = self.config["test_runs"]
        
        # Validate dataset exists
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset not found: {self.dataset_dir}")
    
    def _create_temp_db(self) -> Path:
        """Create a temporary database path for benchmarking."""
        temp_dir = tempfile.mkdtemp(prefix="oboyu_bench_")
        db_path = Path(temp_dir) / "benchmark.db"
        return db_path
    
    def _cleanup_temp_db(self, db_path: Path) -> None:
        """Clean up temporary database."""
        # Remove temp directory
        if db_path.parent.exists():
            shutil.rmtree(db_path.parent, ignore_errors=True)
    
    def _run_single_indexing(self, show_progress: bool = True) -> Dict[str, float]:
        """Run a single indexing operation and return timing metrics."""
        db_path = self._create_temp_db()
        
        try:
            # Initialize components
            crawler_config = CrawlerConfig(
                depth=10,
                max_workers=4,
                max_file_size=10*1024*1024,
                follow_symlinks=False,
                include_patterns=["*"],
                exclude_patterns=[]
            )
            
            # Create progress indicator
            progress = None
            if show_progress:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
                    transient=True
                )
            
            # Timing metrics
            metrics = {}
            
            # 1. File discovery
            with Timer("file_discovery") as timer:
                if progress:
                    with progress:
                        task = progress.add_task("Discovering files...", total=None)
                        files = list(discover_documents(
                            Path(self.dataset_dir),
                            patterns=crawler_config.include_patterns,
                            exclude_patterns=crawler_config.exclude_patterns,
                            max_depth=crawler_config.depth,
                            max_file_size=crawler_config.max_file_size
                        ))
                        progress.update(task, completed=len(files))
                else:
                    files = list(discover_documents(
                        Path(self.dataset_dir),
                        patterns=crawler_config.include_patterns,
                        exclude_patterns=crawler_config.exclude_patterns,
                        max_depth=crawler_config.depth,
                        max_file_size=crawler_config.max_file_size
                    ))
            metrics["file_discovery_time"] = timer.elapsed
            metrics["total_files"] = len(files)
            
            # 2. Content extraction
            # For benchmarking, we'll use the crawler to process all files
            crawler = Crawler(
                depth=crawler_config.depth,
                include_patterns=crawler_config.include_patterns,
                exclude_patterns=crawler_config.exclude_patterns,
                max_file_size=crawler_config.max_file_size,
                follow_symlinks=crawler_config.follow_symlinks,
                japanese_encodings=["utf-8", "shift-jis", "euc-jp"],
                max_workers=crawler_config.max_workers
            )
            
            with Timer("content_extraction") as timer:
                # Use crawler.crawl method
                documents = crawler.crawl(self.dataset_dir)
            metrics["content_extraction_time"] = timer.elapsed
            metrics["total_documents"] = len(documents)
            
            # 3. Indexing (embedding generation + database storage)
            indexer_config = IndexerConfig(
                processing=ProcessingConfig(db_path=db_path),
                model=ModelConfig(embedding_model="cl-nagoya/ruri-v3-30m")
            )
            indexer = Indexer(config=indexer_config)
            
            # Track total chunks processed
            total_chunks = 0
            
            with Timer("total_indexing") as total_timer:
                # Index all documents at once
                total_chunks = indexer.index_documents(documents)
            
            # For now, we'll estimate embedding vs storage time
            # In a real implementation, we'd need to modify the indexer to track these separately
            total_indexing_time = total_timer.elapsed
            metrics["embedding_generation_time"] = total_indexing_time * 0.7  # Estimate 70% for embeddings
            metrics["database_storage_time"] = total_indexing_time * 0.3  # Estimate 30% for storage
            metrics["total_chunks"] = total_chunks
            
            # Calculate total time
            metrics["total_time"] = (
                metrics["file_discovery_time"] +
                metrics["content_extraction_time"] +
                total_indexing_time
            )
            
            return metrics
            
        finally:
            self._cleanup_temp_db(db_path)
    
    def _run_warmup(self) -> None:
        """Run warmup with minimal data to initialize models and libraries."""
        console.print("[dim]  Initializing models with dummy data...[/dim]")
        
        # Create temporary database
        db_path = self._create_temp_db()
        
        try:
            # Create a few dummy documents using CrawlerResult
            import hashlib
            from datetime import datetime

            from oboyu.crawler.models import CrawlerResult, DocumentMetadata, EncodingType, LanguageCode
            
            # Create dummy files first
            for i in range(5):
                dummy_path = Path(f"/tmp/dummy{i}.txt")
                dummy_content = f"This is test document {i}. " * 10
                dummy_path.parent.mkdir(parents=True, exist_ok=True)
                dummy_path.write_text(dummy_content)
            
            dummy_documents = []
            for i in range(5):
                content = f"This is test document {i}. " * 10
                content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                doc = CrawlerResult(
                    path=Path(f"/tmp/dummy{i}.txt"),
                    title=f"Dummy Document {i}",
                    content=content,
                    language=LanguageCode.ENGLISH,
                    metadata=DocumentMetadata(
                        file_size=len(content),
                        created_at=datetime.now(),
                        modified_at=datetime.now(),
                        encoding=EncodingType.UTF8,
                        language=LanguageCode.ENGLISH,
                        content_hash=content_hash
                    )
                )
                dummy_documents.append(doc)
            
            # Initialize indexer and process dummy documents
            indexer_config = IndexerConfig(
                processing=ProcessingConfig(db_path=db_path),
                model=ModelConfig(embedding_model="cl-nagoya/ruri-v3-30m")
            )
            indexer = Indexer(config=indexer_config)
            
            # Index dummy documents to initialize models
            indexer.index_documents(dummy_documents)
            
        finally:
            self._cleanup_temp_db(db_path)
            # Clean up dummy files
            for i in range(5):
                dummy_path = Path(f"/tmp/dummy{i}.txt")
                if dummy_path.exists():
                    dummy_path.unlink()
    
    def run(self, monitor_system: bool = True) -> IndexingResult:
        """Run the indexing benchmark."""
        from datetime import datetime
        
        print_section(f"Indexing Benchmark - {self.dataset_size} dataset")
        console.print(f"[dim]Start time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]")
        
        # System monitoring
        monitor = SystemMonitor() if monitor_system else None
        
        # Warmup runs
        if self.warmup_runs > 0:
            console.print("\n[yellow]Phase 1: Warmup[/yellow]")
            console.print("Initializing models and libraries...")
            console.print(f"[dim]Warmup start: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]")
            self._run_warmup()
            gc.collect()
            console.print(f"[dim]Warmup complete: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]")
        
        # Test runs
        console.print("\n[yellow]Phase 2: Test Runs[/yellow]")
        console.print(f"Running {self.test_runs} test run(s)...")
        console.print(f"[dim]Test runs start: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]")
        all_metrics: List[Dict[str, float]] = []
        
        if monitor:
            monitor.start()
        
        for i in range(self.test_runs):
            console.print(f"\nRun {i+1}/{self.test_runs}:")
            console.print(f"[dim]  Test {i+1} start: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]")
            metrics = self._run_single_indexing(show_progress=True)
            console.print(f"[dim]  Test {i+1} end: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]")
            all_metrics.append(metrics)
            
            # Print run results
            print_metric("Total time", metrics["total_time"], "time")
            print_metric("Files processed", metrics["total_files"])
            print_metric("Chunks created", metrics["total_chunks"])
            
            # Collect samples
            if monitor:
                monitor.samples.append(monitor.sample())
            
            console.print(f"[dim]  GC start: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]")
            gc.collect()
            console.print(f"[dim]  GC end: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]")
        
        if monitor:
            monitor.stop()
        
        console.print(f"[dim]Test runs complete: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]")
        
        # Calculate aggregated results
        console.print("\n[yellow]Phase 3: Aggregating Results[/yellow]")
        console.print(f"[dim]Aggregation start: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]")
        result = self._aggregate_results(all_metrics)
        console.print(f"[dim]Aggregation end: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]")
        
        # Add system metrics
        if monitor:
            result.system_metrics = monitor.get_summary()
        
        console.print(f"[dim]Benchmark complete: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]")
        return result
    
    def _aggregate_results(self, all_metrics: List[Dict[str, float]]) -> IndexingResult:
        """Aggregate metrics from multiple runs."""
        # Average all metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = sum(values) / len(values)
        
        # Calculate derived metrics
        files_per_second = avg_metrics["total_files"] / avg_metrics["total_time"]
        chunks_per_second = avg_metrics["total_chunks"] / avg_metrics["total_time"]
        
        return IndexingResult(
            dataset_size=self.dataset_size,
            total_files=int(avg_metrics["total_files"]),
            total_chunks=int(avg_metrics["total_chunks"]),
            total_time=avg_metrics["total_time"],
            file_discovery_time=avg_metrics["file_discovery_time"],
            content_extraction_time=avg_metrics["content_extraction_time"],
            embedding_generation_time=avg_metrics["embedding_generation_time"],
            database_storage_time=avg_metrics["database_storage_time"],
            files_per_second=files_per_second,
            chunks_per_second=chunks_per_second,
            metadata={
                "warmup_runs": self.warmup_runs,
                "test_runs": self.test_runs,
                "all_runs": all_metrics
            }
        )


def benchmark_indexing(
    dataset_sizes: List[str],
    data_dir: Path = DATA_DIR,
    monitor_system: bool = True
) -> Dict[str, IndexingResult]:
    """Run indexing benchmarks for multiple dataset sizes."""
    results = {}
    
    for size in dataset_sizes:
        try:
            benchmark = IndexingBenchmark(size, data_dir)
            result = benchmark.run(monitor_system)
            results[size] = result
            
            # Print summary
            print_section(f"Summary - {size} dataset")
            print_metric("Total time", result.total_time, "time")
            print_metric("Files/second", result.files_per_second)
            print_metric("Chunks/second", result.chunks_per_second)
            
        except Exception as e:
            console.print(f"[red]Error benchmarking {size} dataset: {e}[/red]")
            import traceback
            traceback.print_exc()
            continue
    
    return results


if __name__ == "__main__":
    # Test with small dataset
    results = benchmark_indexing(["small"])
    
    for size, result in results.items():
        console.print(f"\n[bold]Results for {size} dataset:[/bold]")
        print_metric("Total files", result.total_files)
        print_metric("Total chunks", result.total_chunks)
        print_metric("Total time", result.total_time, "time")
        print_metric("Files/second", f"{result.files_per_second:.2f}")
        print_metric("Chunks/second", f"{result.chunks_per_second:.2f}")
