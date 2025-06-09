---
id: configuration-reference
title: Configuration Options Reference
sidebar_position: 3
---

# Configuration Options Reference

Complete reference for all Oboyu configuration options, their types, valid values, and performance implications.

## Configuration File Structure

Oboyu uses YAML configuration with three main sections:

```yaml
# Essential indexer settings
indexer:
  db_path: "~/.oboyu/index.db"
  chunk_size: 1024
  # ... other indexer options

# Essential crawler settings  
crawler:
  include_patterns: ["*.txt", "*.md"]
  exclude_patterns: ["*/node_modules/*"]
  # ... other crawler options

# Essential query settings
query:
  default_mode: "hybrid"
  top_k: 5
  # ... other query options
```

## Indexer Section

Controls how documents are processed and indexed.

### `indexer.db_path`

**Type**: String (file path)  
**Default**: `"~/.oboyu/index.db"`  
**Description**: Location where the search index is stored

```yaml
indexer:
  db_path: "~/.oboyu/index.db"          # Default location
  # db_path: "/fast/ssd/oboyu/index.db" # SSD for performance
  # db_path: "~/work/.oboyu/work.db"    # Project-specific
```

**Constraints**:
- Must be a valid file path
- Directory must be writable
- Supports `~` expansion for home directory

**Performance Impact**:
- SSD vs HDD: 10-50x performance difference
- Network storage: Not recommended for performance

### `indexer.chunk_size`

**Type**: Integer  
**Default**: `1024`  
**Description**: Size of document chunks in tokens

```yaml
indexer:
  chunk_size: 512    # Small chunks - more precise
  # chunk_size: 1024  # Balanced (default)
  # chunk_size: 2048  # Large chunks - more context
```

**Valid Range**: 128 - 8192  
**Recommended**: 512-2048

**Guidelines by Content Type**:
- **Code files**: 512-1024
- **Technical docs**: 1024-1536  
- **Academic papers**: 1536-2048
- **Personal notes**: 1024-1536

**Performance Impact**:
- Smaller chunks: Higher memory usage, more precise retrieval
- Larger chunks: Faster indexing, less precise retrieval

### `indexer.chunk_overlap`

**Type**: Integer  
**Default**: `256`  
**Description**: Number of tokens overlapping between chunks

```yaml
indexer:
  chunk_overlap: 128   # Minimal overlap
  # chunk_overlap: 256  # Standard overlap (default)
  # chunk_overlap: 512  # High overlap for context
```

**Valid Range**: 0 - (chunk_size / 2)  
**Recommended**: chunk_size / 4

**Guidelines**:
- **No overlap (0)**: Smallest index size, possible context loss
- **25% overlap**: Good balance (recommended)
- **50% overlap**: Maximum context preservation

**Performance Impact**:
- More overlap: Larger index size, better context preservation
- Less overlap: Smaller index, faster searches, possible boundary issues

### `indexer.embedding_model`

**Type**: String (Hugging Face model ID)  
**Default**: `"cl-nagoya/ruri-v3-30m"`  
**Description**: Embedding model for semantic search

```yaml
indexer:
  embedding_model: "cl-nagoya/ruri-v3-30m"           # Japanese-optimized (default)
  # embedding_model: "sentence-transformers/all-MiniLM-L6-v2"  # English-focused
  # embedding_model: "intfloat/multilingual-e5-small"          # Multilingual
```

**Popular Models**:

| Model | Language | Size | Quality | Speed |
|-------|----------|------|---------|-------|
| `cl-nagoya/ruri-v3-30m` | Japanese + English | Medium | High | Medium |
| `sentence-transformers/all-MiniLM-L6-v2` | English | Small | Good | Fast |
| `intfloat/multilingual-e5-small` | Multilingual | Small | Good | Fast |
| `sentence-transformers/all-mpnet-base-v2` | English | Large | Excellent | Slow |

**Performance Impact**:
- Larger models: Better quality, slower indexing, more memory
- Smaller models: Faster processing, may reduce quality

### `indexer.use_reranker`

**Type**: Boolean  
**Default**: `true`  
**Description**: Enable cross-encoder reranking for better result quality

```yaml
indexer:
  use_reranker: true   # Better quality (default)
  # use_reranker: false # Faster processing
```

**Performance Impact**:
- `true`: +20-30% indexing time, significantly better search quality
- `false`: Faster indexing and searching, may miss relevant results

**Recommendations**:
- Enable for quality-critical applications
- Disable for speed-critical or very large collections

## Crawler Section

Controls which files are discovered and processed.

### `crawler.include_patterns`

**Type**: List of strings (glob patterns)  
**Default**: `["*.txt", "*.md", "*.py", "*.rst", "*.java", "*.html"]`  
**Description**: File patterns to include in indexing

```yaml
crawler:
  include_patterns:
    - "*.txt"                 # Plain text
    - "*.md"                  # Markdown
    - "*.py"                  # Python
    - "**/*.rst"              # reStructuredText (recursive)
    - "docs/**/*"             # All files in docs
```

**Common Patterns**:

| Pattern | Description |
|---------|-------------|
| `"*.ext"` | Files with specific extension |
| `"**/*.ext"` | Recursive file search |
| `"dir/**/*"` | All files in directory tree |
| `"**/README*"` | All README files |
| `"*.{md,txt}"` | Multiple extensions |

**Content Type Examples**:
```yaml
# Documentation
include_patterns: ["*.md", "*.rst", "*.txt", "**/README*"]

# Code
include_patterns: ["*.py", "*.js", "*.java", "*.go", "*.rs"]

# Mixed content
include_patterns: ["*.md", "*.txt", "*.py", "*.json", "*.yaml"]
```

### `crawler.exclude_patterns`

**Type**: List of strings (glob patterns)  
**Default**: `["*/node_modules/*", "*/.git/*", "*/venv/*", "*/__pycache__/*"]`  
**Description**: File patterns to exclude from indexing

```yaml
crawler:
  exclude_patterns:
    - "*/node_modules/*"      # Node.js dependencies
    - "*/.git/*"              # Git repository files
    - "*/venv/*"              # Python virtual environments
    - "*/__pycache__/*"       # Python cache
    - "*.tmp"                 # Temporary files
    - "*/backup/*"            # Backup directories
```

**Common Exclusions**:

| Pattern | Purpose |
|---------|---------|
| `"*/node_modules/*"` | Node.js dependencies |
| `"*/venv/*"` | Python virtual environments |
| `"*/.git/*"` | Git repository data |
| `"*/target/*"` | Build directories |
| `"*.log"` | Log files |
| `"*.tmp"` | Temporary files |

**Performance Tip**: Specific exclusions significantly speed up crawling.

## Query Section

Controls default search behavior.

### `query.default_mode`

**Type**: String (enum)  
**Default**: `"hybrid"`  
**Description**: Default search mode for queries

```yaml
query:
  default_mode: "hybrid"    # Best balance (default)
  # default_mode: "vector"  # Semantic search
  # default_mode: "bm25"    # Keyword search
```

**Valid Values**:

| Mode | Description | Best For |
|------|-------------|----------|
| `"bm25"` | Keyword search | Exact terms, technical identifiers |
| `"vector"` | Semantic search | Concepts, natural language queries |
| `"hybrid"` | Combined approach | General use, best quality |

**Performance Comparison**:
- `bm25`: Fastest
- `vector`: Medium speed, best for concepts
- `hybrid`: Slowest, best overall quality

### `query.top_k`

**Type**: Integer  
**Default**: `5`  
**Description**: Default number of results to return

```yaml
query:
  top_k: 5      # Quick results (default)
  # top_k: 10   # More results
  # top_k: 20   # Comprehensive results
```

**Valid Range**: 1 - 100  
**Recommended**: 5-20

**Guidelines**:
- **Interactive use**: 5-10
- **API/programmatic**: 10-20
- **Research/analysis**: 20-50

**Performance Impact**: Linear scaling (2x top_k ≈ 2x search time)

## Advanced Configuration

### Auto-Optimized Parameters

These parameters are automatically optimized based on your system but can be overridden:

```yaml
# Advanced parameters (usually auto-optimized)
indexer:
  batch_size: 32              # Processing batch size
  max_workers: 4              # Thread count
  ef_construction: 200        # HNSW build parameter
  ef_search: 50               # HNSW search parameter
  
crawler:
  max_workers: 4              # Crawling threads
  timeout: 30                 # File read timeout (seconds)
  max_file_size: 10485760     # 10MB file size limit
  
query:
  reranker_batch_size: 32     # Reranker batch size
  cache_size: 1000            # Query cache size
```

### Environment Variable Overrides

All configuration can be overridden with environment variables:

```bash
# Pattern: OBOYU_<SECTION>_<OPTION>
export OBOYU_INDEXER_CHUNK_SIZE=2048
export OBOYU_INDEXER_USE_RERANKER=true
export OBOYU_QUERY_DEFAULT_MODE=hybrid
export OBOYU_QUERY_TOP_K=20

# Special variables
export OBOYU_CONFIG_PATH="/custom/config.yaml"
export OBOYU_LOG_LEVEL=DEBUG
export OBOYU_CACHE_DIR="~/.cache/oboyu"
```

## Configuration Templates

### Personal Knowledge Base
```yaml
# Personal notes and documents
indexer:
  db_path: "~/Documents/.oboyu/personal.db"
  chunk_size: 1536
  chunk_overlap: 384
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true

crawler:
  include_patterns:
    - "*.md"
    - "*.txt"
    - "*.org"
    - "notes/**/*"
    - "journal/**/*"
  exclude_patterns:
    - "*/Archive/*"
    - "*/.obsidian/*"
    - "*.tmp"

query:
  default_mode: "hybrid"
  top_k: 10
```

### Software Development
```yaml
# Code and technical documentation
indexer:
  db_path: "~/dev/.oboyu/code.db"
  chunk_size: 1024
  chunk_overlap: 256
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true

crawler:
  include_patterns:
    - "*.py"
    - "*.js"
    - "*.ts"
    - "*.java"
    - "*.go"
    - "*.rs"
    - "*.md"
    - "*.rst"
    - "**/README*"
    - "**/docs/**/*"
  exclude_patterns:
    - "*/node_modules/*"
    - "*/.git/*"
    - "*/venv/*"
    - "*/target/*"
    - "*/build/*"
    - "*/dist/*"
    - "*/__pycache__/*"

query:
  default_mode: "hybrid"
  top_k: 8
```

### Academic Research
```yaml
# Research papers and academic content
indexer:
  db_path: "~/Research/.oboyu/papers.db"
  chunk_size: 2048
  chunk_overlap: 512
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true

crawler:
  include_patterns:
    - "*.tex"
    - "*.bib"
    - "*.md"
    - "*.txt"
    - "papers/**/*"
    - "notes/**/*"
  exclude_patterns:
    - "*/backup/*"
    - "*/old/*"
    - "*.aux"
    - "*.log"
    - "*.bbl"

query:
  default_mode: "hybrid"
  top_k: 15
```

### High Performance
```yaml
# Optimized for speed
indexer:
  db_path: "/ssd/oboyu/index.db"
  chunk_size: 512
  chunk_overlap: 64
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  use_reranker: false

crawler:
  include_patterns: ["*.txt", "*.md"]
  exclude_patterns:
    - "*"  # Very selective

query:
  default_mode: "bm25"
  top_k: 5
```

## Validation Rules

### Required Fields
- At least one `crawler.include_patterns` entry
- Valid `indexer.db_path` (writable directory)
- Valid `indexer.embedding_model` (Hugging Face model ID)

### Constraints
- `indexer.chunk_size`: 128 ≤ value ≤ 8192
- `indexer.chunk_overlap`: 0 ≤ value ≤ chunk_size/2
- `query.default_mode`: Must be "vector", "bm25", or "hybrid"
- `query.top_k`: 1 ≤ value ≤ 100

### Recommended Relationships
- `chunk_overlap` = `chunk_size` / 4 (25% overlap)
- For Japanese content: `chunk_size` ≤ 1024
- For code content: `chunk_size` ≤ 1024
- For academic papers: `chunk_size` ≥ 1536

## Configuration Best Practices

1. **Start Simple**: Begin with default configuration
2. **Measure Impact**: Test changes with realistic workloads
3. **Document Changes**: Keep notes on customizations
4. **Version Control**: Store configuration in version control
5. **Environment Specific**: Use different configs for different environments

## Migration and Compatibility

### Migrating Old Configurations
```bash
# Automatic migration
oboyu config migrate

# Manual migration
oboyu config convert old-config.yaml new-config.yaml
```

### Deprecated Options
The following options are deprecated and automatically migrated:

| Old Option | New Option | Migration |
|------------|------------|-----------|
| `indexer.batch_size` | Auto-optimized | Removed |
| `indexer.max_workers` | Auto-optimized | Removed |
| `query.vector_weight` | N/A | Uses RRF algorithm |
| `query.bm25_weight` | N/A | Uses RRF algorithm |

## Troubleshooting Configuration

### Common Issues

**"Configuration validation failed"**
```bash
# Check syntax
oboyu config validate

# Show detailed errors  
oboyu config validate --verbose
```

**"Model not found"**
- Check model name on [Hugging Face](https://huggingface.co)
- Verify internet connection
- Try default model: `cl-nagoya/ruri-v3-30m`

**"Permission denied for db_path"**
```bash
# Check directory permissions
ls -la $(dirname ~/.oboyu)

# Create directory
mkdir -p ~/.oboyu && chmod 755 ~/.oboyu
```

### Debug Configuration
```bash
# Show effective configuration
oboyu config show --effective

# Show configuration sources
oboyu config show --sources

# Validate specific file
oboyu config validate ~/.oboyu/config.yaml
```