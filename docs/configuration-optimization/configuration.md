---
id: configuration
title: Detailed Configuration Guide
sidebar_position: 1
---

# Detailed Configuration Guide

This comprehensive guide covers all aspects of configuring Oboyu for your specific needs. Whether you're setting up a simple personal search or managing enterprise-scale document collections, this guide will help you optimize your configuration.

## Configuration Basics

### Zero Configuration Start

Oboyu works out of the box with sensible defaults:

```bash
# Just works - no configuration needed
oboyu index /path/to/documents
oboyu query --query "your search"
```

### Configuration File Locations

Oboyu searches for configuration in this order:

1. **Command-line specified**: `--config` or `-c` flag
2. **User config**: `~/.config/oboyu/config.yaml`
3. **Built-in defaults**: Optimized for most use cases

### Creating Your First Config

```bash
# Generate a starter config
oboyu config init

# Or create manually
mkdir -p ~/.config/oboyu
cat > ~/.config/oboyu/config.yaml << EOF
indexer:
  db_path: "~/.oboyu/index.db"
  chunk_size: 1024
EOF
```

## Essential Configuration Options

### Indexer Settings

The most important settings that affect indexing performance and quality:

```yaml
indexer:
  # Where to store the search index
  db_path: "~/.oboyu/index.db"
  
  # How to split documents (128-8192)
  chunk_size: 1024
  
  # Overlap between chunks (0 to chunk_size/2)
  chunk_overlap: 256
  
  # Which embedding model to use
  embedding_model: "cl-nagoya/ruri-v3-30m"
  
  # Enable reranking for better results
  use_reranker: true
```

### Crawler Settings

Control which files to index:

```yaml
crawler:
  # File patterns to include (glob patterns)
  include_patterns:
    - "*.txt"
    - "*.md"
    - "*.py"
    - "*.java"
    - "*.html"
    - "**/*.rst"  # Include subdirectories
  
  # Patterns to exclude
  exclude_patterns:
    - "*/node_modules/*"
    - "*/.git/*"
    - "*/venv/*"
    - "*/__pycache__/*"
    - "*.log"
    - "*.tmp"
```

### Query Settings

Default search behavior:

```yaml
query:
  # Search mode: vector, bm25, or hybrid
  default_mode: "hybrid"
  
  # Number of results to return
  top_k: 5
```

## Configuration by Use Case

### Personal Knowledge Base

For personal notes and documents:

```yaml
# ~/.config/oboyu/personal.yaml
indexer:
  db_path: "~/Documents/.oboyu/personal.db"
  chunk_size: 1536          # Larger chunks for mixed content
  chunk_overlap: 384        # 25% overlap
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true

crawler:
  include_patterns:
    - "*.md"
    - "*.txt"
    - "*.org"               # Org-mode files
    - "journal/**/*"        # All journal entries
    - "notes/**/*"          # All notes
  exclude_patterns:
    - "*/Archive/*"
    - "*/.obsidian/*"       # Obsidian metadata
    - "*.tmp"

query:
  default_mode: "hybrid"
  top_k: 10                 # More results for exploration
```

### Software Development

For code and technical documentation:

```yaml
# ~/.config/oboyu/dev.yaml
indexer:
  db_path: "~/dev/.oboyu/code.db"
  chunk_size: 1024          # Balanced for code
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

query:
  default_mode: "hybrid"
  top_k: 8
```

### Academic Research

For papers and research notes:

```yaml
# ~/.config/oboyu/research.yaml
indexer:
  db_path: "~/Research/.oboyu/papers.db"
  chunk_size: 2048          # Larger for academic text
  chunk_overlap: 512        # More context preservation
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true

crawler:
  include_patterns:
    - "*.tex"               # LaTeX source
    - "*.bib"               # Bibliography
    - "*.md"                # Notes
    - "*.txt"               # Plain text
    - "papers/**/*"         # Papers directory
  exclude_patterns:
    - "*/backup/*"
    - "*/old/*"
    - "*.aux"               # LaTeX temp files
    - "*.log"

query:
  default_mode: "hybrid"
  top_k: 15                 # More results for research
```

### Large Enterprise Collection

For massive document repositories:

```yaml
# /etc/oboyu/enterprise.yaml
indexer:
  db_path: "/data/oboyu/main.db"
  chunk_size: 1024          # Balanced for scale
  chunk_overlap: 128        # Minimal for space
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: false       # Disable for speed at scale

crawler:
  include_patterns:
    - "*.txt"
    - "*.md"
    - "*.html"
    - "*.xml"
    - "*.json"
  exclude_patterns:
    - "*/archive/*"
    - "*/backup/*"
    - "*/temp/*"
    - "*.log"
    - "*.bak"

query:
  default_mode: "bm25"      # Fastest for scale
  top_k: 20                 # Pre-filter more
```

## Advanced Configuration

### Environment Variables

Override any setting with environment variables:

```bash
# Pattern: OBOYU_<SECTION>_<OPTION>
export OBOYU_INDEXER_DB_PATH="/fast/ssd/index.db"
export OBOYU_INDEXER_CHUNK_SIZE="2048"
export OBOYU_QUERY_DEFAULT_MODE="hybrid"
export OBOYU_QUERY_TOP_K="10"

# Special variables
export OBOYU_CONFIG_PATH="/custom/config.yaml"
export OBOYU_LOG_LEVEL="DEBUG"
export OBOYU_CACHE_DIR="~/.cache/oboyu"
```

### Multiple Configurations

Manage different search contexts:

```bash
# Create named configs
oboyu config save personal ~/.config/oboyu/personal.yaml
oboyu config save work ~/.config/oboyu/work.yaml
oboyu config save research ~/.config/oboyu/research.yaml

# Use specific config
oboyu index ~/Documents --config personal
oboyu query --query "meeting notes" --config work
```

### Performance Tuning

#### For Speed
```yaml
indexer:
  chunk_size: 512          # Smaller chunks
  chunk_overlap: 64        # Minimal overlap
  use_reranker: false      # Skip reranking

query:
  default_mode: "bm25"     # Fastest mode
  top_k: 5                 # Fewer results
```

#### For Quality
```yaml
indexer:
  chunk_size: 2048         # Larger context
  chunk_overlap: 512       # More overlap
  use_reranker: true       # Enable reranking

query:
  default_mode: "hybrid"   # Best quality
  top_k: 15                # More candidates
```

## Configuration Validation

### Checking Your Config

```bash
# Validate syntax and values
oboyu config validate

# Show effective configuration
oboyu config show

# Test with dry run
oboyu index /test/path --dry-run
```

### Common Validation Errors

#### Invalid YAML
```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

#### Invalid Values
- `chunk_size`: Must be 128-8192
- `chunk_overlap`: Must be < chunk_size/2
- `default_mode`: Must be vector/bm25/hybrid
- `top_k`: Must be 1-100

## Migration from Older Versions

### Automatic Migration

```bash
# Backup old config
cp ~/.config/oboyu/config.yaml ~/.config/oboyu/config.yaml.bak

# Run migration
oboyu config migrate

# Verify new config
oboyu config show
```

### Manual Migration

Old complex configs are simplified:

```yaml
# Old (v0.x)
indexer:
  batch_size: 32
  max_workers: 4
  ef_construction: 200
  ef_search: 50
  
# New (v1.0+) - Auto-optimized
indexer:
  # batch_size: (auto-optimized)
  # max_workers: (auto-optimized)
  # ef_construction: (auto-optimized)
```

## Best Practices

### 1. Start Simple
Begin with minimal config and add as needed:

```yaml
indexer:
  db_path: "~/.oboyu/index.db"
  chunk_size: 1024
```

### 2. Use Named Configurations
Create purpose-specific configs:

```bash
# Work documents
oboyu index ~/work --config ~/.config/oboyu/work.yaml

# Personal notes  
oboyu index ~/personal --config ~/.config/oboyu/personal.yaml
```

### 3. Regular Updates
Keep indices fresh:

```bash
# Add to crontab
0 2 * * * oboyu index ~/Documents --update --config personal
```

### 4. Monitor Performance
Track indexing and search metrics:

```bash
# Enable metrics
export OBOYU_METRICS=true
oboyu index ~/Documents --verbose
```

## Troubleshooting Configuration

### Debug Mode
```bash
# Enable detailed logging
OBOYU_LOG_LEVEL=DEBUG oboyu index /path
```

### Common Issues

**"Model not found"**
- Check internet connection
- Verify model name on HuggingFace
- Try default: `cl-nagoya/ruri-v3-30m`

**"Permission denied"**
```bash
# Check permissions
ls -la ~/.oboyu
# Fix if needed
chmod 755 ~/.oboyu
```

**"Out of memory"**
- Reduce `chunk_size`
- Index in smaller batches
- Disable reranker temporarily

## Next Steps

- Explore [Indexing Strategies](indexing-strategies.md) for optimal document processing
- Learn about [Search Optimization](search-optimization.md) for better results
- Configure [Performance Tuning](performance-tuning.md) for your hardware