# Oboyu Performance Benchmark Suite

This directory contains a comprehensive performance benchmark suite for Oboyu, designed to measure and track performance metrics over time.

## Overview

The benchmark suite measures two key performance areas:

1. **Indexing Performance**: How fast Oboyu can discover, process, and index documents
2. **Search Performance**: How fast Oboyu can execute vector searches and return results

## Quick Start

### Running Your First Benchmark

```bash
# Run benchmark with small dataset (default)
uv run python bench/run_speed_benchmark.py

# Run benchmark with specific datasets
uv run python bench/run_speed_benchmark.py --datasets small medium
```

### Analyzing Results

```bash
# Analyze the latest benchmark results
uv run python bench/run_speed_benchmark.py --analyze-only

# Compare specific runs
uv run python bench/analyze.py --compare <run_id_1> <run_id_2>
```

## Directory Structure

```
bench/
├── __init__.py              # Package initialization
├── README.md                # This file
├── config.py                # Centralized configuration
├── utils.py                 # Common utilities
├── data/                    # Test datasets (generated)
│   ├── small/              # Small dataset (~50 files)
│   ├── medium/             # Medium dataset (~1,000 files)
│   └── large/              # Large dataset (~10,000 files)
├── queries/                 # Query datasets (generated)
│   ├── japanese_queries.json
│   ├── english_queries.json
│   └── mixed_queries.json
├── results/                 # Benchmark results and reports
│   ├── benchmark_run_*.json
│   ├── summary_*.json
│   └── report_*.txt
├── generate_test_data.py    # Generate test datasets
├── generate_queries.py      # Generate query datasets
├── benchmark_indexing.py    # Indexing performance benchmarks
├── benchmark_search.py      # Search performance benchmarks
├── results.py              # Result data structures
├── reporter.py             # Report generation
├── runner.py               # Benchmark orchestration
├── analyze.py              # Result analysis tools
└── run_speed_benchmark.py  # Main entry point
```

## Generating Test Data

### Generate Datasets

```bash
# Generate all dataset sizes
uv run python -m bench.generate_test_data all

# Generate specific dataset size
uv run python -m bench.generate_test_data small medium

# Custom output directory
uv run python -m bench.generate_test_data small --output-dir /path/to/data
```

Dataset characteristics:
- **Small**: 50 files, 1-5KB each
- **Medium**: 1,000 files, 2-10KB each  
- **Large**: 10,000 files, 3-15KB each

### Generate Queries

```bash
# Generate all query languages
uv run python -m bench.generate_queries

# Generate specific languages
uv run python -m bench.generate_queries --languages japanese english
```

Query characteristics:
- **Japanese**: 50 queries (technical, business, general, code)
- **English**: 50 queries (technical, business, general, code)
- **Mixed**: 20 queries (Japanese-English mixed)

## Running Benchmarks

### Command Line Options

```bash
uv run python bench/run_speed_benchmark.py [options]
```

Options:
- `--datasets SIZES`: Specify dataset sizes (small, medium, large)
- `--languages LANGS`: Specify query languages (japanese, english, mixed)
- `--skip-indexing`: Skip indexing benchmarks
- `--skip-search`: Skip search benchmarks
- `--use-existing-indexes`: Use existing test indexes
- `--force-regenerate`: Force regeneration of test data
- `--analyze-only`: Only analyze existing results

### Examples

```bash
# Run benchmark on medium dataset with Japanese queries only
uv run python bench/run_speed_benchmark.py --datasets medium --languages japanese

# Skip indexing and use existing indexes for search benchmark
uv run python bench/run_speed_benchmark.py --skip-indexing --use-existing-indexes

# Force regenerate all test data and run benchmark
uv run python bench/run_speed_benchmark.py --datasets small medium large --force-regenerate
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
uv run python bench/analyze.py --latest 5

# Analyze with custom results directory
uv run python bench/analyze.py --results-dir /path/to/results
```

### Compare Runs

```bash
# Compare two specific runs
uv run python bench/analyze.py --compare <run_id_1> <run_id_2>

# Set regression threshold (default: 10%)
uv run python bench/analyze.py --compare <run_id_1> <run_id_2> --regression-threshold 0.05
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

Edit `config.py` to customize:

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

## Integration with CI/CD

### Basic Integration

```bash
# Run benchmark and check for regressions
uv run python bench/run_speed_benchmark.py --datasets small
uv run python bench/analyze.py --latest 2 --regression-threshold 0.1
```

### GitHub Actions Example

```yaml
- name: Run Performance Benchmark
  run: uv run python bench/run_speed_benchmark.py --datasets small
  
- name: Check for Regressions
  run: |
    uv run python bench/analyze.py --latest 2 --regression-threshold 0.1
    if [ $? -ne 0 ]; then
      echo "Performance regression detected!"
      exit 1
    fi
```

## Troubleshooting

### Common Issues

1. **"Oboyu is not installed"**
   ```bash
   uv pip install -e .
   ```

2. **"Dataset not found"**
   ```bash
   uv run python -m bench.generate_test_data all
   ```

3. **"Query file not found"**
   ```bash
   uv run python -m bench.generate_queries
   ```

4. **Memory issues with large dataset**
   - Reduce batch size in `config.py`
   - Use smaller dataset for testing

### Debug Mode

Enable verbose output:
```bash
uv run python bench/run_speed_benchmark.py --datasets small --verbose
```

## RAG Accuracy Evaluation

### Overview

The RAG accuracy evaluation suite measures Oboyu's effectiveness as a complete RAG (Retrieval-Augmented Generation) system, with special focus on Japanese document search performance.

### Running RAG Accuracy Benchmarks

**Important**: The RAG accuracy benchmark requires the Ruri v3 embedding model (~90MB) to be downloaded on first run. Ensure you have a stable internet connection.

```bash
# Run with synthetic dataset (quick test)
uv run python bench/benchmark_rag_accuracy.py --datasets synthetic

# Run with Japanese evaluation datasets
uv run python bench/benchmark_rag_accuracy.py --datasets miracl-ja mldr-ja jagovfaqs-22k

# Evaluate specific search modes (currently only vector is implemented)
uv run python bench/benchmark_rag_accuracy.py --search-modes vector

# Include reranking evaluation (for planned feature)
uv run python bench/benchmark_rag_accuracy.py --evaluate-reranking

# Test components without full indexing
uv run python bench/test_rag_minimal.py
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
uv run python bench/benchmark_rag_accuracy.py --datasets custom --custom-dataset-path /path/to/dataset.json
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
uv run python bench/benchmark_rag_accuracy.py --report-only

# Compare with previous run
uv run python bench/benchmark_rag_accuracy.py --compare-with bench/results/rag_accuracy_20250101_120000.json

# Set custom regression threshold (default: 10%)
uv run python bench/benchmark_rag_accuracy.py --compare-with previous.json --regression-threshold 0.05
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