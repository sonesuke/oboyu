"""Orchestrate benchmark execution."""

import uuid
from pathlib import Path
from typing import Dict, List

from rich.console import Console

from bench.config import DATA_DIR, DATASET_SIZES, OBOYU_CONFIG, QUERIES_DIR, RESULTS_DIR, get_query_languages
from bench.speed.benchmark_indexing import benchmark_indexing
from bench.speed.benchmark_search import benchmark_search
from bench.speed.generate_queries import generate_all_queries, save_queries
from bench.speed.generate_test_data import generate_dataset
from bench.speed.reporter import BenchmarkReporter
from bench.speed.results import BenchmarkRun, SystemInfo
from bench.utils import check_oboyu_installation, get_timestamp, print_header, print_section
from oboyu.indexer import LegacyIndexer as Indexer

console = Console()


class BenchmarkRunner:
    """Orchestrate the execution of benchmarks."""
    
    def __init__(
        self,
        dataset_sizes: List[str],
        query_languages: List[str],
        data_dir: Path = DATA_DIR,
        queries_dir: Path = QUERIES_DIR,
        results_dir: Path = RESULTS_DIR
    ) -> None:
        """Initialize benchmark runner."""
        self.dataset_sizes = dataset_sizes
        self.query_languages = query_languages
        self.data_dir = data_dir
        self.queries_dir = queries_dir
        self.results_dir = results_dir
        
        # Validate inputs
        for size in dataset_sizes:
            if size not in DATASET_SIZES:
                raise ValueError(f"Invalid dataset size: {size}")
        
        valid_languages = get_query_languages()
        for lang in query_languages:
            if lang not in valid_languages:
                raise ValueError(f"Invalid query language: {lang}")
    
    def prepare_data(self, force_regenerate: bool = False) -> bool:
        """Prepare test data and queries."""
        print_section("Preparing Test Data")
        
        # Check and generate datasets
        for size in self.dataset_sizes:
            dataset_path = self.data_dir / size
            if not dataset_path.exists() or force_regenerate:
                console.print(f"Generating {size} dataset...")
                stats = generate_dataset(size, dataset_path, clean=True)
                console.print(f"✓ Generated {stats['total_files']} files")
            else:
                console.print(f"✓ {size} dataset already exists")
        
        # Check and generate queries
        queries_exist = all(
            (self.queries_dir / f"{lang}_queries.json").exists()
            for lang in self.query_languages
        )
        
        if not queries_exist or force_regenerate:
            console.print("Generating query datasets...")
            queries = generate_all_queries()
            
            # Filter by requested languages
            filtered_queries = {
                lang: q for lang, q in queries.items()
                if lang in self.query_languages
            }
            
            save_queries(filtered_queries, self.queries_dir)
            console.print(f"✓ Generated queries for {len(filtered_queries)} languages")
        else:
            console.print("✓ Query datasets already exist")
        
        return True
    
    def run_indexing_benchmarks(self) -> Dict[str, any]:
        """Run indexing benchmarks for all dataset sizes."""
        print_section("Running Indexing Benchmarks")
        
        results = benchmark_indexing(
            self.dataset_sizes,
            self.data_dir,
            monitor_system=True
        )
        
        return results
    
    def create_test_indexes(self) -> Dict[str, Path]:
        """Create indexed databases for search benchmarks."""
        print_section("Creating Test Indexes")
        
        db_paths = {}
        
        for size in self.dataset_sizes:
            console.print(f"\nIndexing {size} dataset...")
            
            # Create temporary database
            db_path = self.results_dir / f"test_index_{size}.db"
            
            # Remove existing database
            if db_path.exists():
                db_path.unlink()
            
            # Initialize indexer with config
            from oboyu.indexer.config import IndexerConfig, ModelConfig, ProcessingConfig, SearchConfig
            indexer_dict = {
                **OBOYU_CONFIG["indexer"],
                "db_path": str(db_path)
            }
            
            model_config = ModelConfig(
                embedding_model=indexer_dict.get("embedding_model", "cl-nagoya/ruri-v3-30m"),
                batch_size=indexer_dict.get("batch_size", 8),
            )
            
            processing_config = ProcessingConfig(
                chunk_size=indexer_dict.get("chunk_size", 300),
                chunk_overlap=indexer_dict.get("chunk_overlap", 75),
                db_path=Path(indexer_dict["db_path"]),
            )
            
            search_config = SearchConfig()
            
            indexer_config = IndexerConfig(
                model=model_config,
                processing=processing_config,
                search=search_config,
            )
            indexer = Indexer(config=indexer_config)
            
            # Index dataset
            dataset_path = self.data_dir / size
            chunks_indexed, files_processed = indexer.index_directory(
                str(dataset_path),
                incremental=False  # Always fresh index for benchmarks
            )
            
            # Display stats
            console.print(f"✓ Indexed {files_processed} documents, {chunks_indexed} chunks")
            
            db_paths[size] = db_path
        
        return db_paths
    
    def run_search_benchmarks(self, db_paths: Dict[str, Path]) -> Dict[str, any]:
        """Run search benchmarks for all configurations."""
        print_section("Running Search Benchmarks")
        
        results = benchmark_search(
            db_paths,
            self.query_languages,
            self.queries_dir,
            test_queries=None,  # Use all queries
            monitor_system=True
        )
        
        return results
    
    def run(
        self,
        force_regenerate: bool = False,
        skip_indexing: bool = False,
        skip_search: bool = False,
        use_existing_indexes: bool = False
    ) -> BenchmarkRun:
        """Run the complete benchmark suite."""
        print_header("Oboyu Performance Benchmark Suite")
        
        # Check Oboyu installation
        if not check_oboyu_installation():
            raise RuntimeError("Oboyu is not installed")
        
        # Generate run ID and gather system info
        run_id = f"{get_timestamp()}_{uuid.uuid4().hex[:8]}"
        system_info = SystemInfo.gather()
        
        console.print(f"Run ID: {run_id}")
        console.print(f"System: {system_info.platform}")
        console.print("")
        
        # Prepare data
        if not self.prepare_data(force_regenerate):
            raise RuntimeError("Failed to prepare test data")
        
        # Initialize results
        indexing_results = {}
        search_results = {}
        
        # Run indexing benchmarks
        if not skip_indexing:
            indexing_results = self.run_indexing_benchmarks()
        
        # Create or locate test indexes
        db_paths = {}
        if not skip_search:
            if use_existing_indexes:
                # Look for existing indexes
                for size in self.dataset_sizes:
                    db_path = self.results_dir / f"test_index_{size}.db"
                    if db_path.exists():
                        db_paths[size] = db_path
                    else:
                        console.print(f"[yellow]Warning: No existing index for {size} dataset[/yellow]")
            else:
                # Create new indexes
                db_paths = self.create_test_indexes()
            
            # Run search benchmarks
            if db_paths:
                search_results = self.run_search_benchmarks(db_paths)
        
        # Create benchmark run
        run = BenchmarkRun(
            run_id=run_id,
            timestamp=get_timestamp(),
            system_info=system_info,
            config={
                "dataset_sizes": self.dataset_sizes,
                "query_languages": self.query_languages,
                "oboyu_config": OBOYU_CONFIG
            },
            indexing_results=indexing_results,
            search_results=search_results,
            metadata={
                "force_regenerate": force_regenerate,
                "skip_indexing": skip_indexing,
                "skip_search": skip_search,
                "use_existing_indexes": use_existing_indexes
            }
        )
        
        # Save results
        print_section("Saving Results")
        result_file = run.save(self.results_dir)
        console.print(f"✓ Results saved to: {result_file}")
        
        # Generate and save reports
        reporter = BenchmarkReporter(run)
        report_files = reporter.save_reports(self.results_dir)
        console.print(f"✓ Reports saved: {', '.join(f.name for f in report_files.values())}")
        
        # Print summary
        print_section("Results Summary")
        reporter.print_summary()
        
        # Clean up temporary indexes if not using existing
        if not use_existing_indexes and not skip_search:
            for db_path in db_paths.values():
                if db_path.exists():
                    db_path.unlink()
        
        return run


def run_quick_benchmark() -> BenchmarkRun:
    """Run a quick benchmark with small dataset for testing."""
    runner = BenchmarkRunner(
        dataset_sizes=["small"],
        query_languages=["english", "japanese"],
    )
    
    return runner.run(
        force_regenerate=False,
        skip_indexing=False,
        skip_search=False,
        use_existing_indexes=False
    )


def run_full_benchmark() -> BenchmarkRun:
    """Run full benchmark suite with all datasets."""
    runner = BenchmarkRunner(
        dataset_sizes=["small", "medium", "large"],
        query_languages=["english", "japanese", "mixed"],
    )
    
    return runner.run(
        force_regenerate=False,
        skip_indexing=False,
        skip_search=False,
        use_existing_indexes=False
    )
