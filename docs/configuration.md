# Oboyu Configuration

Oboyu supports extensive configuration through YAML configuration files. Configuration can be provided via command-line options, configuration files, or environment variables.

## Configuration File Location

Oboyu looks for configuration files in the following order:

1. File specified with `--config` or `-c` command-line option
2. `~/.config/oboyu/config.yaml` (XDG-compliant default location)
3. Built-in defaults

## Configuration Structure

The configuration file uses YAML format with three main sections:

```yaml
# Crawler settings for document discovery and processing
crawler:
  # ... crawler options

# Indexer settings for embedding generation and storage
indexer:
  # ... indexer options

# Query settings for search behavior
query:
  # ... query options
```

## Crawler Configuration

Controls how Oboyu discovers and processes documents during indexing.

```yaml
crawler:
  # Maximum directory traversal depth
  depth: 10
  
  # File patterns to include (glob patterns)
  include_patterns:
    - "*.txt"
    - "*.md"
    - "*.html"
    - "*.py"
    - "*.java"
    - "*.js"
    - "*.ts"
    - "*.yaml"
    - "*.yml"
    - "*.json"
    - "*.toml"
    - "*.rst"
    - "*.ipynb"
  
  # Patterns to exclude (glob patterns)
  exclude_patterns:
    - "*/node_modules/*"
    - "*/.venv/*"
    - "*/__pycache__/*"
    - "*/.git/*"
    - "*/build/*"
    - "*/dist/*"
  
  # Maximum file size in bytes (10MB default)
  max_file_size: 10485760
  
  # Whether to follow symbolic links
  follow_symlinks: false
  
  # Japanese encodings to detect and handle
  japanese_encodings:
    - "utf-8"
    - "shift-jis"
    - "euc-jp"
  
  # Number of worker threads for parallel processing
  max_workers: 4
  
  # Whether to respect .gitignore files
  respect_gitignore: true
```

### Crawler Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `depth` | int | 10 | Maximum directory traversal depth |
| `include_patterns` | list[str] | [*.txt, *.md, ...] | File patterns to include (glob syntax) |
| `exclude_patterns` | list[str] | [*/node_modules/*, ...] | Patterns to exclude (glob syntax) |
| `max_file_size` | int | 10485760 | Maximum file size in bytes |
| `follow_symlinks` | bool | false | Whether to follow symbolic links |
| `japanese_encodings` | list[str] | [utf-8, shift-jis, euc-jp] | Japanese encodings to detect |
| `max_workers` | int | 4 | Number of worker threads |
| `respect_gitignore` | bool | true | Whether to respect .gitignore files |

## Indexer Configuration

Controls embedding generation, model loading, and database storage.

```yaml
indexer:
  # Text chunking settings
  chunk_size: 1024
  chunk_overlap: 256
  
  # Embedding model configuration
  embedding_model: "cl-nagoya/ruri-v3-30m"
  embedding_device: "cpu"
  batch_size: 8
  
  # Database configuration
  db_path: "~/.oboyu/index.db"
  
  # Reranker settings
  use_reranker: true
  reranker_model: "cl-nagoya/ruri-reranker-small"
  reranker_use_onnx: true
  reranker_top_k_multiplier: 3
  reranker_score_threshold: 0.5
  
  # Change detection for incremental indexing
  change_detection_strategy: "smart"  # "timestamp", "hash", or "smart"
  cleanup_deleted_files: true
  
  # Performance settings
  enable_onnx_optimization: true
  onnx_cache_dir: "~/.cache/oboyu/onnx"
```

### Indexer Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | int | 1024 | Size of text chunks in characters |
| `chunk_overlap` | int | 256 | Overlap between chunks in characters |
| `embedding_model` | str | "cl-nagoya/ruri-v3-30m" | Hugging Face model ID for embeddings |
| `embedding_device` | str | "cpu" | Device for embedding generation (cpu/cuda) |
| `batch_size` | int | 8 | Batch size for embedding generation |
| `db_path` | str | "~/.oboyu/index.db" | Path to the database file |
| `use_reranker` | bool | true | Enable reranking for improved accuracy |
| `reranker_model` | str | "cl-nagoya/ruri-reranker-small" | Hugging Face model ID for reranker |
| `reranker_use_onnx` | bool | true | Use ONNX optimization for reranker |
| `reranker_top_k_multiplier` | int | 3 | Multiplier for initial retrieval before reranking |
| `reranker_score_threshold` | float | 0.5 | Minimum score threshold for reranking |
| `change_detection_strategy` | str | "smart" | Strategy for detecting file changes |
| `cleanup_deleted_files` | bool | true | Remove deleted files from index |
| `enable_onnx_optimization` | bool | true | Enable ONNX optimization for models |
| `onnx_cache_dir` | str | "~/.cache/oboyu/onnx" | Directory for ONNX model cache |

### Change Detection Strategies

- **`timestamp`**: Use file modification time to detect changes (fastest)
- **`hash`**: Use content hash to detect changes (most accurate, slower)
- **`smart`**: Use timestamp for most files, hash for critical files (balanced)

## Query Configuration

Controls search behavior and result formatting.

```yaml
query:
  # Search settings
  default_mode: "hybrid"
  vector_weight: 0.7
  bm25_weight: 0.3
  top_k: 5
  
  # Reranking settings
  use_reranker: true
  reranker_top_k: 15  # Number of results to rerank
  
  # Output formatting
  snippet_length: 160
  highlight_matches: true
  show_scores: false
  
  # Language filtering
  language_filter: null  # null for all languages, or "ja", "en", etc.
```

### Query Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_mode` | str | "hybrid" | Default search mode (vector, bm25, hybrid) |
| `vector_weight` | float | 0.7 | Weight for vector search in hybrid mode |
| `bm25_weight` | float | 0.3 | Weight for BM25 search in hybrid mode |
| `top_k` | int | 5 | Number of results to return |
| `use_reranker` | bool | true | Enable reranking by default |
| `reranker_top_k` | int | 15 | Number of results to rerank |
| `snippet_length` | int | 160 | Length of result snippets in characters |
| `highlight_matches` | bool | true | Highlight query terms in snippets |
| `show_scores` | bool | false | Show relevance scores in results |
| `language_filter` | str\|null | null | Filter results by language |

## Example Configurations

### Minimal Configuration

```yaml
# Basic configuration for general use
indexer:
  db_path: "./my_index.db"
  
query:
  top_k: 10
  use_reranker: false  # Disable for faster searches
```

### Japanese-Optimized Configuration

```yaml
# Optimized for Japanese content
crawler:
  include_patterns:
    - "*.txt"
    - "*.md"
    - "*.html"
  japanese_encodings:
    - "utf-8"
    - "shift-jis"
    - "euc-jp"
    - "iso-2022-jp"

indexer:
  chunk_size: 512  # Smaller chunks for Japanese
  chunk_overlap: 128
  use_reranker: true  # Essential for Japanese accuracy

query:
  default_mode: "hybrid"
  vector_weight: 0.8  # Favor vector search for Japanese
  bm25_weight: 0.2
  language_filter: "ja"
```

### Performance-Optimized Configuration

```yaml
# Optimized for speed
crawler:
  max_workers: 8  # More parallel processing

indexer:
  batch_size: 16  # Larger batches
  use_reranker: false  # Skip reranking for speed
  enable_onnx_optimization: true
  change_detection_strategy: "timestamp"  # Fastest change detection

query:
  default_mode: "vector"  # Vector-only for speed
  use_reranker: false
  top_k: 3  # Fewer results
```

### Accuracy-Optimized Configuration

```yaml
# Optimized for accuracy
indexer:
  chunk_size: 512  # Smaller chunks for precision
  chunk_overlap: 128
  use_reranker: true
  reranker_model: "cl-nagoya/ruri-reranker-large"  # Larger reranker
  reranker_top_k_multiplier: 5  # More candidates for reranking
  change_detection_strategy: "hash"  # Most accurate change detection

query:
  default_mode: "hybrid"
  use_reranker: true
  reranker_top_k: 30  # Rerank more candidates
  top_k: 10  # More results
```

## Environment Variables

Some configuration options can be overridden with environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `OBOYU_DB_PATH` | Database file path | `export OBOYU_DB_PATH="/path/to/index.db"` |
| `OBOYU_CONFIG` | Configuration file path | `export OBOYU_CONFIG="/path/to/config.yaml"` |
| `OBOYU_EMBEDDING_MODEL` | Embedding model | `export OBOYU_EMBEDDING_MODEL="custom-model"` |
| `TOKENIZERS_PARALLELISM` | Tokenizer parallelism | `export TOKENIZERS_PARALLELISM="false"` |

## Command-Line Overrides

Most configuration options can be overridden via command-line arguments. Command-line options take precedence over configuration files.

Example:
```bash
# Override database path and chunk size
oboyu index ./docs --db-path custom.db --chunk-size 2048

# Override search mode and weights
oboyu query "search term" --mode hybrid --vector-weight 0.8 --bm25-weight 0.2
```

## Configuration Validation

Oboyu validates configuration on startup and provides helpful error messages for invalid settings:

- Numeric values must be positive
- File paths must be accessible
- Model names must be valid Hugging Face model IDs
- Weights must sum to reasonable values in hybrid mode

Invalid configurations will fall back to safe defaults with warnings.

## Advanced Features

### Model Caching

Oboyu caches downloaded models in `~/.cache/huggingface/` by default. You can configure this with the `HF_HOME` environment variable:

```bash
export HF_HOME="/custom/cache/path"
```

### ONNX Optimization

ONNX optimization provides 2-4x speed improvements. Configure the cache directory:

```yaml
indexer:
  enable_onnx_optimization: true
  onnx_cache_dir: "~/.cache/oboyu/onnx"
```

### Memory Management

For large document collections, consider these memory-conscious settings:

```yaml
indexer:
  batch_size: 4  # Smaller batches
  embedding_device: "cpu"  # Use CPU to avoid GPU memory issues

crawler:
  max_workers: 2  # Fewer parallel workers
  max_file_size: 5242880  # 5MB limit
```