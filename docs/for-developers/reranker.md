# Oboyu Reranker Guide

## Overview

The Oboyu reranker feature significantly improves search result quality by applying Cross-Encoder models to re-score and reorder initial retrieval results. This is particularly beneficial for Retrieval-Augmented Generation (RAG) applications where the quality of retrieved context directly impacts the generated output.

## What is Reranking?

Reranking is a two-stage retrieval process:

1. **Initial Retrieval**: Fast vector/BM25/hybrid search retrieves a larger set of candidates (e.g., 30-60 documents)
2. **Reranking**: A more sophisticated Cross-Encoder model re-scores these candidates for better relevance

This approach combines the efficiency of vector search with the accuracy of Cross-Encoder models.

## Supported Models

Oboyu supports the following Ruri reranker models:

- **`cl-nagoya/ruri-reranker-small`** (default): Lightweight model offering excellent balance of performance and resource usage
- **`cl-nagoya/ruri-v3-reranker-310m`**: Heavy model with superior accuracy for quality-focused applications

Both models are specifically optimized for Japanese text while maintaining good multilingual capabilities.

### Model Comparison

| Model | Size | Memory Usage | Speed | Accuracy | Best For |
|-------|------|--------------|-------|----------|----------|
| `cl-nagoya/ruri-reranker-small` | ~100M params | ~400MB | Fast (20-40ms) | Excellent | General use, real-time applications |
| `cl-nagoya/ruri-v3-reranker-310m` | 310M params | ~1.2GB | Moderate (50-100ms) | Superior | Quality-focused, batch processing |

**Recommendation**: The default `ruri-reranker-small` model provides the best balance for most use cases. It offers excellent accuracy with significantly lower resource requirements (~67% memory reduction) compared to the 310m model.

## Key Features

### 1. ONNX Optimization
- Automatic conversion to ONNX format for 2-4x faster CPU inference
- Lazy model loading to minimize startup time
- Persistent model caching in XDG-compliant directories

### 2. Flexible Integration
- Works with all search modes (vector, BM25, hybrid)
- Optional feature that can be enabled/disabled per query
- Configurable top-k multiplier for initial retrieval

### 3. Performance Benefits
- **Hit Rate**: 4-10% improvement in retrieval accuracy
- **MRR (Mean Reciprocal Rank)**: 20%+ improvement in result ranking
- **Japanese Queries**: Significant enhancement over embedding-only search

## Configuration

### Global Configuration

Add reranker settings to your `~/.config/oboyu/config.yaml`:

```yaml
indexer:
  # Reranker settings
  reranker_model: "cl-nagoya/ruri-reranker-small"  # Model to use
  use_reranker: true                                 # Enable by default
  reranker_use_onnx: true                           # Use ONNX optimization
  reranker_device: "cpu"                            # Device (cpu/cuda)
  reranker_top_k_multiplier: 3                      # Retrieve 3x candidates
  reranker_batch_size: 8                            # Batch size
  reranker_max_length: 512                          # Max sequence length
  reranker_threshold: null                          # Score threshold (optional)
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `reranker_model` | Model name or path | `cl-nagoya/ruri-reranker-small` |
| `use_reranker` | Enable reranking by default | `true` |
| `reranker_use_onnx` | Use ONNX optimization | `true` |
| `reranker_device` | Device for inference | `cpu` |
| `reranker_top_k_multiplier` | Multiplier for initial retrieval | `3` |
| `reranker_batch_size` | Batch size for reranking | `8` |
| `reranker_max_length` | Maximum input length | `512` |
| `reranker_threshold` | Minimum score threshold | `null` |

## Usage

### Command Line Interface

```bash
# Search with reranking (uses config default)
oboyu query "システムの設計原則について"

# Explicitly enable reranking
oboyu query "design principles" --rerank

# Disable reranking for this query
oboyu query "quick lookup" --no-rerank

# Rerank with custom top-k
oboyu query "重要な概念" --rerank --top-k 5
```

### Python API

```python
from oboyu.indexer.indexer import Indexer
from oboyu.indexer.config import IndexerConfig

# Initialize with reranker enabled
config = IndexerConfig(config_dict={
    "indexer": {
        "db_path": "oboyu.db",
        "use_reranker": True,
        "reranker_model": "cl-nagoya/ruri-reranker-small",
    }
})
indexer = Indexer(config=config)

# Search with reranking
results = indexer.search(
    query="日本語の文書検索",
    limit=5,
    use_reranker=True  # Or None to use config default
)

# Search without reranking
results = indexer.search(
    query="quick search",
    limit=10,
    use_reranker=False
)
```

### Custom Reranker

```python
from oboyu.indexer.reranker import BaseReranker, create_reranker

# Create a custom reranker
reranker = create_reranker(
    model_name="cl-nagoya/ruri-reranker-small",
    use_onnx=True,
    device="cpu",
    batch_size=16,
)

# Use with indexer
indexer = Indexer(config=config, reranker=reranker)
```

## How Reranking Works

### 1. Initial Retrieval Phase
```
Query → Embedding → Vector Search → Top 15 candidates (if top_k=5, multiplier=3)
```

### 2. Reranking Phase
```
Query + Each Candidate → Cross-Encoder → Relevance Score → Re-order → Top 5 results
```

### 3. Architecture Comparison

**Bi-Encoder (Embedding Model)**:
- Encodes query and documents separately
- Fast but less accurate for nuanced matching
- Used in initial retrieval

**Cross-Encoder (Reranker)**:
- Processes query-document pairs together
- Slower but more accurate
- Captures fine-grained semantic relationships

## Performance Considerations

### Speed vs. Accuracy Trade-offs

| Configuration | Speed | Accuracy | Use Case |
|---------------|-------|----------|----------|
| No reranking | Fastest | Good | High-volume, latency-sensitive |
| Small model + ONNX | Fast | Better | Balanced performance |
| 310m model + ONNX | Moderate | Best | Quality-focused RAG |
| 310m model + PyTorch | Slower | Best | GPU environments |

### Optimization Tips

1. **Batch Size**: Larger batches improve throughput but increase latency
2. **Top-k Multiplier**: Higher values improve recall but increase processing time
3. **ONNX**: Always use for CPU deployments (2-4x speedup)
4. **Model Selection**: Use small model for real-time applications

### Resource Requirements

**Memory Usage (Approximate)**:
- 310m model: ~1.2GB (ONNX) / ~1.5GB (PyTorch)
- Small model: ~400MB (ONNX) / ~500MB (PyTorch)

**Processing Time (per query, 15 candidates)**:
- 310m + ONNX: ~50-100ms
- 310m + PyTorch: ~150-300ms
- Small + ONNX: ~20-40ms

## Benchmarking

Run the reranking benchmark to evaluate performance:

```bash
# Run benchmark with test queries
python -m bench.benchmark_reranking \
    --db-path oboyu.db \
    --test-queries test_queries.json \
    --output results.json

# Custom configuration
python -m bench.benchmark_reranking \
    --config reranker_config.yaml \
    --db-path oboyu.db \
    --test-queries queries.json \
    --top-k 5 10 20 \
    --initial-k 60
```

## Model Cache Management

ONNX models are cached for faster subsequent loads:

```
~/.cache/oboyu/embedding/cache/models/onnx/
├── cl-nagoya_ruri-v3-reranker-310m/
│   ├── model.onnx
│   ├── model_optimized.onnx
│   ├── tokenizer_config.json
│   └── onnx_config.json
└── cl-nagoya_ruri-reranker-small/
    └── ...
```

To clear the cache:
```bash
rm -rf ~/.cache/oboyu/embedding/cache/models/onnx/
```

## Troubleshooting

### Common Issues

1. **Slow First Query**: Models are loaded lazily on first use. Subsequent queries will be faster.

2. **Out of Memory**: Reduce batch size or disable reranking temporarily:
   ```yaml
   reranker_batch_size: 4
   # Or disable for this query: oboyu query "text" --no-rerank
   ```

3. **ONNX Conversion Fails**: Disable ONNX optimization:
   ```yaml
   reranker_use_onnx: false
   ```

### Debug Mode

Enable debug logging to see reranking details:

```python
import logging
logging.getLogger("oboyu.indexer.reranker").setLevel(logging.DEBUG)
```

## Best Practices

1. **For RAG Applications**: Always enable reranking for context retrieval
2. **For Japanese Content**: Use the default 310m model for best results
3. **For Mixed Language**: The models work well for multilingual content
4. **Initial Retrieval**: Set multiplier based on result diversity needs (3-5x is typical)
5. **Threshold Setting**: Use threshold to filter low-confidence results in critical applications


## Future Enhancements

- Support for custom Cross-Encoder models
- GPU acceleration for batch processing
- Async reranking for better concurrency
- Fine-tuning support for domain-specific ranking