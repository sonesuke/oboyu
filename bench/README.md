# Oboyu Performance Benchmark Suite

This directory contains a comprehensive performance benchmark suite for Oboyu, designed to measure and track performance metrics over time.

## Overview

The benchmark suite measures two key performance areas:

1. **Indexing Performance**: How fast Oboyu can discover, process, and index documents
2. **Search Performance**: How fast Oboyu can execute vector searches and return results

## Quick Start

### Running Your First Benchmark

```bash
# Install dependencies (if not already installed)
pip install psutil

# Run a quick benchmark with small dataset
python bench/run_speed_benchmark.py --quick

# Run full benchmark suite
python bench/run_speed_benchmark.py --full
```

### Analyzing Results

```bash
# Analyze the latest benchmark results
python bench/run_speed_benchmark.py --analyze-only

# Compare specific runs
python bench/analyze.py --compare <run_id_1> <run_id_2>
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
python -m bench.generate_test_data all

# Generate specific dataset size
python -m bench.generate_test_data small medium

# Custom output directory
python -m bench.generate_test_data small --output-dir /path/to/data
```

Dataset characteristics:
- **Small**: 50 files, 1-5KB each
- **Medium**: 1,000 files, 2-10KB each  
- **Large**: 10,000 files, 3-15KB each

### Generate Queries

```bash
# Generate all query languages
python -m bench.generate_queries

# Generate specific languages
python -m bench.generate_queries --languages japanese english
```

Query characteristics:
- **Japanese**: 50 queries (technical, business, general, code)
- **English**: 50 queries (technical, business, general, code)
- **Mixed**: 20 queries (Japanese-English mixed)

## Running Benchmarks

### Command Line Options

```bash
python bench/run_speed_benchmark.py [options]
```

Options:
- `--quick`: Run quick benchmark with small dataset only
- `--full`: Run full benchmark with all datasets
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
python bench/run_speed_benchmark.py --datasets medium --languages japanese

# Skip indexing and use existing indexes for search benchmark
python bench/run_speed_benchmark.py --skip-indexing --use-existing-indexes

# Force regenerate all test data and run full benchmark
python bench/run_speed_benchmark.py --full --force-regenerate
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
python bench/analyze.py --latest 5

# Analyze with custom results directory
python bench/analyze.py --results-dir /path/to/results
```

### Compare Runs

```bash
# Compare two specific runs
python bench/analyze.py --compare <run_id_1> <run_id_2>

# Set regression threshold (default: 10%)
python bench/analyze.py --compare <run_id_1> <run_id_2> --regression-threshold 0.05
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
python bench/run_speed_benchmark.py --quick
python bench/analyze.py --latest 2 --regression-threshold 0.1
```

### GitHub Actions Example

```yaml
- name: Run Performance Benchmark
  run: python bench/run_speed_benchmark.py --quick
  
- name: Check for Regressions
  run: |
    python bench/analyze.py --latest 2 --regression-threshold 0.1
    if [ $? -ne 0 ]; then
      echo "Performance regression detected!"
      exit 1
    fi
```

## Troubleshooting

### Common Issues

1. **"Oboyu is not installed"**
   ```bash
   pip install -e .
   ```

2. **"Dataset not found"**
   ```bash
   python -m bench.generate_test_data all
   ```

3. **"Query file not found"**
   ```bash
   python -m bench.generate_queries
   ```

4. **Memory issues with large dataset**
   - Reduce batch size in `config.py`
   - Use smaller dataset for testing

### Debug Mode

Enable verbose output:
```bash
python bench/run_speed_benchmark.py --quick --verbose
```

## Future Enhancements

- [ ] Accuracy measurement benchmarks
- [ ] BM25 and hybrid search benchmarks
- [ ] Multi-threaded indexing benchmarks
- [ ] Network-based MCP server benchmarks
- [ ] Automated performance regression alerts
- [ ] Grafana dashboard integration

## Contributing

When adding new benchmarks:

1. Follow the existing pattern in `benchmark_*.py`
2. Add configuration to `config.py`
3. Update result classes in `results.py`
4. Add reporting logic to `reporter.py`
5. Document in this README

## License

Same as Oboyu project (MIT License)