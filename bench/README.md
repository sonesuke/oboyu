# Oboyu Benchmark Suite

This directory contains a comprehensive benchmark suite for Oboyu, designed to measure and track performance metrics and accuracy over time.

## Overview

The benchmark suite provides three main evaluation areas:

1. **Speed Benchmarks** (`speed/`): Measure indexing and search performance
2. **Accuracy Benchmarks** (`accuracy/`): Evaluate retrieval accuracy for RAG performance
3. **Reranker Benchmarks** (`reranker/`): Test reranker model effectiveness

## Quick Start

### Unified Benchmark Runner (Recommended)

The easiest way to run benchmarks is using the unified runner:

```bash
# Run all benchmarks quickly (reduced scope)
PYTHONPATH=. uv run python bench/run_benchmarks.py all --quick

# Run only speed benchmarks
PYTHONPATH=. uv run python bench/run_benchmarks.py speed --datasets small medium

# Run only accuracy evaluation
PYTHONPATH=. uv run python bench/run_benchmarks.py accuracy --datasets synthetic

# Run comprehensive evaluation (full scope)
PYTHONPATH=. uv run python bench/run_benchmarks.py all --comprehensive

# Run reranker-specific benchmarks
PYTHONPATH=. uv run python bench/run_benchmarks.py reranker --models small large
```

**Note**: All benchmark commands require `PYTHONPATH=.` to be set from the project root directory to properly resolve imports.

### Individual Benchmark Modules

You can also run benchmarks individually for more control:

```bash
# Speed benchmarks
PYTHONPATH=. uv run python bench/speed/run_speed_benchmark.py --datasets small medium

# Accuracy evaluation
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py --datasets synthetic miracl-ja

# Reranker evaluation
PYTHONPATH=. uv run python bench/reranker/benchmark_reranking.py --models cl-nagoya/ruri-reranker-small

# Analyze results
PYTHONPATH=. uv run python bench/speed/analyze.py --latest 5
```

### Unified Runner Options

The `run_benchmarks.py` script provides several useful options:

```bash
# Show help for all options
PYTHONPATH=. uv run python bench/run_benchmarks.py --help

# Quick benchmarks (reduced scope for faster execution)
PYTHONPATH=. uv run python bench/run_benchmarks.py all --quick

# Comprehensive benchmarks (full scope for thorough evaluation)  
PYTHONPATH=. uv run python bench/run_benchmarks.py all --comprehensive

# Verbose output for debugging
PYTHONPATH=. uv run python bench/run_benchmarks.py speed --verbose

# Custom output directory
PYTHONPATH=. uv run python bench/run_benchmarks.py accuracy --output-dir /custom/path
```

## Directory Structure

```
bench/
├── __init__.py                    # Package initialization with exports
├── README.md                      # This file
├── run_benchmarks.py              # Unified benchmark runner (main entry point)
├── config.py                      # Shared configuration
├── utils.py                       # Common utilities  
├── logger.py                      # Shared logging utilities
├── core/                          # Shared core functionality
│   ├── __init__.py               # Core module exports
│   ├── benchmark_base.py         # Base benchmark class
│   └── metrics.py                # Unified metrics calculation
├── speed/                         # Speed benchmarks
│   ├── __init__.py               # Speed module initialization
│   ├── run_speed_benchmark.py    # Main speed benchmark runner
│   ├── benchmark_indexing.py     # Indexing performance tests
│   ├── benchmark_search.py       # Search performance tests
│   ├── runner.py                 # Speed benchmark orchestration
│   ├── analyze.py                # Results analysis
│   ├── reporter.py               # Report generation
│   ├── results.py                # Results management
│   ├── generate_queries.py       # Query generation
│   └── generate_test_data.py     # Test data generation
├── accuracy/                      # Accuracy evaluation (renamed from rag_accuracy)
│   ├── __init__.py               # Accuracy module exports
│   ├── benchmark_rag_accuracy.py # Main accuracy benchmark runner
│   └── rag_accuracy/             # RAG evaluation implementation
│       ├── __init__.py           # RAG accuracy module initialization
│       ├── rag_evaluator.py      # Core RAG evaluation logic
│       ├── dataset_manager.py    # JMTEB dataset management
│       ├── metrics_calculator.py # IR metrics calculation
│       ├── reranker_evaluator.py # Reranking evaluation
│       └── results_analyzer.py   # Results analysis and reporting
├── reranker/                      # Reranker-specific benchmarks (new)
│   ├── __init__.py               # Reranker module exports
│   └── benchmark_reranking.py    # Reranker evaluation benchmarks
├── config/                        # Configuration files
│   └── rag_eval_config.yaml      # RAG evaluation configuration
├── data/                          # Test datasets (generated)
│   ├── small/                    # Small dataset (~50 files)
│   ├── medium/                   # Medium dataset (~1,000 files)
│   └── large/                    # Large dataset (~10,000 files)
├── queries/                       # Query datasets (generated)
│   ├── japanese_queries.json
│   ├── english_queries.json
│   └── mixed_queries.json
└── results/                       # Benchmark results and reports
    ├── speed_*.json              # Speed benchmark results
    ├── accuracy_*.json           # Accuracy benchmark results  
    ├── reranker_*.json           # Reranker benchmark results
    ├── *_report_*.txt            # Human-readable reports
    └── *_summary_*.json          # Summary files for CI/CD
```

## Speed Benchmarks

### Generate Test Datasets

```bash
# Generate all dataset sizes
PYTHONPATH=. uv run python bench/speed/generate_test_data.py all

# Generate specific dataset size
PYTHONPATH=. uv run python bench/speed/generate_test_data.py small medium

# Custom output directory
PYTHONPATH=. uv run python bench/speed/generate_test_data.py small --output-dir /path/to/data
```

Dataset characteristics:
- **Small**: 50 files, 1-5KB each
- **Medium**: 1,000 files, 2-10KB each  
- **Large**: 10,000 files, 3-15KB each

### Generate Query Datasets

```bash
# Generate all query languages
PYTHONPATH=. uv run python bench/speed/generate_queries.py

# Generate specific languages
PYTHONPATH=. uv run python bench/speed/generate_queries.py --languages japanese english
```

Query characteristics:
- **Japanese**: 50 queries (technical, business, general, code)
- **English**: 50 queries (technical, business, general, code)
- **Mixed**: 20 queries (Japanese-English mixed)

## RAG Accuracy Benchmarks

The RAG accuracy evaluation framework measures retrieval quality using standard Information Retrieval (IR) metrics on Japanese datasets based on JMTEB (Japanese Massive Text Embedding Benchmark).

### Supported Datasets

- **MIRACL-ja**: Academic research papers and queries
- **MLDR-ja**: Long document retrieval scenarios  
- **JaGovFaqs-22k**: Government FAQ-style content
- **JaCWIR**: Casual web information retrieval

### Available Metrics

- **Precision@K**: Proportion of relevant documents in top-K results
- **Recall@K**: Proportion of relevant documents retrieved in top-K  
- **NDCG@K**: Normalized Discounted Cumulative Gain at K
- **MRR**: Mean Reciprocal Rank
- **Hit Rate**: Percentage of queries with at least one relevant result
- **F1@K**: Harmonic mean of precision and recall at K

### Running Accuracy Evaluations

```bash
# Run via unified runner (recommended)
PYTHONPATH=. uv run python bench/run_benchmarks.py accuracy --datasets synthetic miracl-ja

# Or run directly
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py

# Evaluate specific datasets
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py --datasets miracl-ja mldr-ja

# Evaluate with specific search modes and top-k values
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py --search-modes vector hybrid --top-k-values 1 5 10

# Generate report from existing results
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py --report-only

# Compare with previous results for regression detection
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py --compare-with bench/results/accuracy_20250101_120000.json
```

## Speed Benchmarks

### Running Speed Benchmarks

#### Command Line Options

```bash
PYTHONPATH=. uv run python bench/speed/run_speed_benchmark.py [options]
```

Options:
- `--datasets SIZES`: Specify dataset sizes (small, medium, large)
- `--languages LANGS`: Specify query languages (japanese, english, mixed)
- `--skip-indexing`: Skip indexing benchmarks
- `--skip-search`: Skip search benchmarks
- `--use-existing-indexes`: Use existing test indexes
- `--force-regenerate`: Force regeneration of test data
- `--analyze-only`: Only analyze existing results

#### Examples

```bash
# Run benchmark on medium dataset with Japanese queries only
PYTHONPATH=. uv run python bench/speed/run_speed_benchmark.py --datasets medium --languages japanese

# Skip indexing and use existing indexes for search benchmark
PYTHONPATH=. uv run python bench/speed/run_speed_benchmark.py --skip-indexing --use-existing-indexes

# Force regenerate all test data and run benchmark
PYTHONPATH=. uv run python bench/speed/run_speed_benchmark.py --datasets small medium large --force-regenerate
```

## Understanding Results

### Indexing Metrics

- **Total Time**: Total time to index the dataset
- **File Discovery Time**: Time to find all files
- **Content Extraction Time**: Time to read and process files
- **Embedding Generation Time**: Time to generate embeddings
- **Database Storage Time**: Time to store in database
- **Files/Second**: Throughput metric for files
- **Chunks/Second**: Throughput metric for chunks

### Search Metrics

- **Response Time**: Time to execute a single query
- **Queries/Second (QPS)**: Throughput metric
- **First Result Time**: Time to get first result
- **Statistics**: Mean, median, P95, P99 response times

### System Metrics

- **CPU Usage**: Average and peak CPU usage
- **Memory Usage**: Average and peak memory usage
- **Disk I/O**: Read and write metrics

## Analyzing Performance

### View Latest Results

```bash
# Analyze last 5 runs
PYTHONPATH=. uv run python bench/analyze.py --latest 5

# Analyze with custom results directory
PYTHONPATH=. uv run python bench/analyze.py --results-dir /path/to/results
```

### Compare Runs

```bash
# Compare two specific runs
PYTHONPATH=. uv run python bench/analyze.py --compare <run_id_1> <run_id_2>

# Set regression threshold (default: 10%)
PYTHONPATH=. uv run python bench/analyze.py --compare <run_id_1> <run_id_2> --regression-threshold 0.05
```

### Performance Trends

The analyzer automatically detects:
- Performance improvements (📈)
- Performance regressions (📉)
- Stable performance (➡️)

## Output Formats

### JSON Results

Complete benchmark data in `results/benchmark_run_*.json`:
- System information
- Configuration used
- Raw timing data
- Individual query results

### Text Reports

Human-readable reports in `results/report_*.txt`:
- Summary statistics
- Performance breakdown
- System usage metrics

### Summary Files

Quick access summaries in `results/summary_*.json`:
- Key metrics only
- Easy to parse for CI/CD

## Configuration

### Benchmark Configuration

Edit `config.py` to customize benchmark settings:

```python
# Dataset sizes
DATASET_SIZES = {
    "small": {
        "file_count": 50,
        "content_size_range": (1000, 5000)
    },
    # ...
}

# Benchmark settings
BENCHMARK_CONFIG = {
    "indexing": {
        "warmup_runs": 1,
        "test_runs": 3
    },
    "search": {
        "warmup_runs": 5,
        "test_runs": 100
    }
}
```

### Current Architecture Configuration

The benchmarks now use the new type-safe configuration system with Pydantic models:

```python
from oboyu.config.indexer import IndexerConfig, ModelConfig, ProcessingConfig
from oboyu.config.crawler import CrawlerConfig

# Example configuration for benchmarks
indexer_config = IndexerConfig(
    processing=ProcessingConfig(
        db_path=Path("benchmark.db"),
        chunk_size=1000,
        chunk_overlap=200
    ),
    model=ModelConfig(
        embedding_model="cl-nagoya/ruri-v3-30m",
        use_onnx=True,
        reranker_model="cl-nagoya/ruri-reranker-small",
        use_reranker=False
    )
)

crawler_config = CrawlerConfig(
    depth=10,
    max_workers=4,
    max_file_size=10*1024*1024,
    include_patterns=["*.txt", "*.md", "*.py"]
)
```

This replaces the old dict-based configuration system and provides:
- Type safety with automatic validation
- Clear documentation of available options
- Better error messages for invalid configurations
- IDE support with autocompletion

### Key Changes from Legacy Configuration

- **No more `config_dict` parameter**: Configurations are now created directly with named parameters
- **Typed sub-configurations**: `ModelConfig`, `ProcessingConfig`, `SearchConfig` for organized settings
- **Validation**: Invalid configurations raise clear errors immediately
- **Backward compatibility**: Convenience properties maintain API compatibility where possible

## Integration with CI/CD

### Basic Integration

```bash
# Run quick benchmark suite and check for regressions
PYTHONPATH=. uv run python bench/run_benchmarks.py all --quick
PYTHONPATH=. uv run python bench/speed/analyze.py --latest 2 --regression-threshold 0.1
```

### GitHub Actions Example

```yaml
- name: Run Quick Performance Benchmark
  run: PYTHONPATH=. uv run python bench/run_benchmarks.py all --quick
  
- name: Check for Speed Regressions
  run: |
    PYTHONPATH=. uv run python bench/speed/analyze.py --latest 2 --regression-threshold 0.1
    if [ $? -ne 0 ]; then
      echo "Performance regression detected!"
      exit 1
    fi

- name: Run Accuracy Benchmark (Weekly)
  if: github.event_name == 'schedule'
  run: PYTHONPATH=. uv run python bench/run_benchmarks.py accuracy --comprehensive
```

## Troubleshooting

### Common Issues

1. **"Oboyu is not installed"**
   ```bash
   uv pip install -e .
   ```

2. **"Dataset not found"**
   ```bash
   PYTHONPATH=. uv run python bench/speed/generate_test_data.py all
   ```

3. **"Query file not found"**
   ```bash
   PYTHONPATH=. uv run python bench/speed/generate_queries.py
   ```

4. **Memory issues with large dataset**
   - Reduce batch size in `config.py`
   - Use smaller dataset for testing

### Debug Mode

Enable verbose output:
```bash
# For unified runner
PYTHONPATH=. uv run python bench/run_benchmarks.py speed --datasets small --verbose

# For individual benchmarks
PYTHONPATH=. uv run python bench/speed/run_speed_benchmark.py --datasets small --verbose
```

## RAG Accuracy Evaluation

### Overview

The RAG accuracy evaluation suite measures Oboyu's effectiveness as a complete RAG (Retrieval-Augmented Generation) system, with special focus on Japanese document search performance.

### Running RAG Accuracy Benchmarks

**Important**: The RAG accuracy benchmark requires the Ruri v3 embedding model (~90MB) to be downloaded on first run. Ensure you have a stable internet connection.

```bash
# Run via unified runner (recommended)
PYTHONPATH=. uv run python bench/run_benchmarks.py accuracy --datasets synthetic

# Run with synthetic dataset (quick test)
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py --datasets synthetic

# Run with Japanese evaluation datasets  
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py --datasets miracl-ja mldr-ja jagovfaqs-22k

# Evaluate specific search modes
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py --search-modes vector hybrid

# Include reranking evaluation
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py --evaluate-reranking
```

### RAG Evaluation Datasets

#### Available Datasets
- **synthetic**: Basic synthetic Japanese dataset for testing
- **miracl-ja**: Multilingual information retrieval (Japanese) - Academic papers and research
- **mldr-ja**: Long document retrieval (Japanese) - Government reports and policy documents
- **jagovfaqs-22k**: Japanese government FAQ dataset - Administrative procedures
- **jacwir**: Japanese casual web information retrieval - Blog posts and web content

Note: These are synthetic implementations that mimic the structure and characteristics of real JMTEB datasets. They provide realistic Japanese content for testing the RAG evaluation framework.

#### Custom Datasets
```bash
# Evaluate with custom dataset
PYTHONPATH=. uv run python bench/benchmark_rag_accuracy.py --datasets custom --custom-dataset-path /path/to/dataset.json
```

### RAG Metrics

#### Retrieval Metrics
- **Precision@K**: Fraction of retrieved documents that are relevant
- **Recall@K**: Fraction of relevant documents that are retrieved
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Hit Rate**: Percentage of queries with at least one relevant result

#### Reranking Metrics (Planned)
- **Ranking Improvement**: Improvement over initial retrieval
- **MAP**: Mean Average Precision
- **Position Improvement**: Average position change for relevant documents

### Analysis and Reporting

```bash
# Generate report from existing results
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py --report-only

# Compare with previous run
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py --compare-with bench/results/accuracy_20250101_120000.json

# Set custom regression threshold (default: 10%)
PYTHONPATH=. uv run python bench/accuracy/benchmark_rag_accuracy.py --compare-with previous.json --regression-threshold 0.05
```

### Configuration

Edit `bench/config/rag_eval_config.yaml` to customize:
- Dataset selection
- Evaluation metrics
- Performance baselines
- Resource limits
- Output formats

## Future Enhancements

- [x] Accuracy measurement benchmarks (RAG evaluation implemented)
- [ ] BM25 and hybrid search benchmarks (partially implemented in RAG evaluation)
- [ ] Multi-threaded indexing benchmarks
- [ ] Network-based MCP server benchmarks
- [ ] Automated performance regression alerts
- [ ] Grafana dashboard integration
- [ ] Real-time RAG evaluation during development
- [ ] Cross-language RAG evaluation (Japanese-English)

## Contributing

When adding new benchmarks:

1. Follow the existing pattern in `benchmark_*.py`
2. Add configuration to `config.py`
3. Update result classes in `results.py`
4. Add reporting logic to `reporter.py`
5. Document in this README

## License

Same as Oboyu project (MIT License)