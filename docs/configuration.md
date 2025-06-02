# Oboyu Configuration

Oboyu has been significantly simplified to provide a better user experience. The configuration system now focuses on essential options while automatically optimizing advanced parameters behind the scenes.

## Quick Start

For most users, Oboyu works great with zero configuration. Simply run:

```bash
oboyu index /path/to/documents
oboyu query "your search"
```

If you need to customize behavior, create a simple configuration file.

## Configuration File Location

Oboyu looks for configuration files in this order:

1. File specified with `--config` or `-c` command-line option
2. `~/.config/oboyu/config.yaml` (XDG-compliant default location)
3. Built-in defaults (works out of the box)

## Simplified Configuration Format

The new simplified configuration focuses on what users actually need to change:

```yaml
# Essential indexer settings (most important)
indexer:
  db_path: "~/.oboyu/index.db"          # Where to store the search index
  chunk_size: 1024                      # Document chunk size for processing
  chunk_overlap: 256                    # Overlap between chunks
  embedding_model: "cl-nagoya/ruri-v3-30m"  # Which embedding model to use
  use_reranker: true                    # Enable reranking for better quality

# Essential crawler settings
crawler:
  include_patterns:                     # File types to index
    - "*.txt"
    - "*.md" 
    - "*.py"
    - "*.rst"
    - "*.ipynb"
  exclude_patterns:                     # Directories/files to skip
    - "*/node_modules/*"
    - "*/.git/*" 
    - "*/venv/*"
    - "*/__pycache__/*"

# Essential query settings  
query:
  default_mode: "hybrid"                # Search mode: vector, bm25, or hybrid
  top_k: 5                             # Number of results to return
```

## What's Been Simplified

The new configuration removes ~60% of options that were rarely changed:

### Auto-Optimized Parameters
These are now automatically optimized based on your system:
- `batch_size` - Optimized based on available memory
- `max_workers` - Optimized based on CPU cores  
- `ef_construction`, `ef_search`, `m` - Vector index parameters
- `bm25_k1`, `bm25_b` - Search algorithm parameters
- `reranker_batch_size`, `reranker_max_length` - Model parameters

### Hard-Coded Sensible Defaults
These use proven defaults that work well for 99% of cases:
- `max_file_size: 10MB` - Prevents memory issues
- `follow_symlinks: false` - Security best practice
- `encoding: utf-8` - Auto-detected at runtime
- `timeout: 30s` - Reasonable for file operations

### Runtime-Only Options
These are now CLI flags only (not in config files):
- `show_scores` - Use `--show-scores` flag
- `interactive` - Use `--interactive` flag  
- `snippet_length` - Use `--snippet-length` flag
- `language_filter` - Use `--language` flag

### Replaced by Better Algorithms
- `vector_weight`, `bm25_weight` - Replaced by RRF (Reciprocal Rank Fusion)
- `change_detection_strategy` - Smart default handles all cases

## Backward Compatibility

Existing configuration files continue to work with automatic migration:

```bash
# Your old config will be automatically migrated
oboyu index /path/to/docs

# To permanently migrate your config file:
oboyu config migrate
```

Deprecated options will show warnings with explanations:
```
Warning: indexer.batch_size is deprecated - now auto-optimized based on system memory
Warning: query.vector_weight is deprecated - replaced by RRF algorithm
```

## Environment Variables

Essential settings can be overridden with environment variables:

```bash
export OBOYU_DB_PATH="/custom/path/index.db"
export OBOYU_EMBEDDING_MODEL="custom-model"
export OBOYU_CHUNK_SIZE="2048"
```

## Advanced Configuration

If you need advanced control, you can still access all parameters programmatically:

```python
from oboyu.config import ConfigManager

# Get auto-optimized config with all advanced parameters
config_mgr = ConfigManager()
full_config = config_mgr.get_auto_optimized_config('indexer')

print(f"Auto-optimized batch size: {full_config['batch_size']}")
print(f"Auto-optimized HNSW params: {full_config['ef_construction']}")
```

## Configuration Examples

### Minimal Configuration
```yaml
# Just override the essentials
indexer:
  db_path: "/tmp/my_index.db"
  chunk_size: 512
```

### Documentation Project
```yaml
indexer:
  db_path: "~/docs_index.db"
  chunk_size: 2048              # Larger chunks for documentation
  embedding_model: "cl-nagoya/ruri-v3-30m"

crawler:
  include_patterns:
    - "*.md"
    - "*.rst" 
    - "*.txt"
  exclude_patterns:
    - "*/build/*"
    - "*/dist/*"

query:
  default_mode: "hybrid"
  top_k: 10
```

### Code Project  
```yaml
indexer:
  db_path: "~/code_index.db"
  chunk_size: 1024              # Good for code files
  use_reranker: true            # Better for technical content

crawler:
  include_patterns:
    - "*.py"
    - "*.js" 
    - "*.ts"
    - "*.java"
    - "*.md"
  exclude_patterns:
    - "*/node_modules/*"
    - "*/.git/*"
    - "*/venv/*"
    - "*/__pycache__/*"
    - "*/target/*"
    - "*/build/*"

query:
  default_mode: "hybrid"
  top_k: 8
```

## Migration Guide

### From Version 0.x
If you're upgrading from an older version with complex configuration:

1. **Back up your old config**:
   ```bash
   cp ~/.config/oboyu/config.yaml ~/.config/oboyu/config.yaml.backup
   ```

2. **Run migration**:
   ```bash
   oboyu config migrate
   ```

3. **Review the simplified config**:
   ```bash
   cat ~/.config/oboyu/config.yaml
   ```

4. **Test everything still works**:
   ```bash
   oboyu query "test search"
   ```

### Removed Options Reference

| Old Option | Status | Replacement |
|------------|--------|-------------|
| `indexer.batch_size` | Auto-optimized | Based on system memory |
| `indexer.max_workers` | Auto-optimized | Based on CPU cores |
| `indexer.ef_construction` | Auto-optimized | Performance-tuned default |
| `indexer.bm25_k1` | Hard-coded | Proven default (1.2) |
| `crawler.max_workers` | Auto-optimized | Based on CPU cores |
| `crawler.timeout` | Hard-coded | 30 seconds |
| `crawler.max_file_size` | Hard-coded | 10MB |
| `query.vector_weight` | Removed | RRF algorithm |
| `query.bm25_weight` | Removed | RRF algorithm |
| `query.show_scores` | Runtime flag | `--show-scores` |
| `query.interactive` | Runtime flag | `--interactive` |

## Troubleshooting

### "Configuration option not found"
If you see warnings about deprecated options:
1. Check the migration guide above
2. Remove the deprecated options from your config file
3. Use the new simplified options or CLI flags

### "Performance seems slower"
The auto-optimized parameters should perform better, but if needed:
1. Check system resources (memory, CPU)
2. Try increasing `chunk_size` for larger documents
3. Ensure `use_reranker: true` for better quality

### "Old behavior not working"
Some advanced options are now hard-coded for stability:
1. File size limit is now always 10MB (prevents memory issues)
2. Symlinks are never followed (security best practice)
3. RRF algorithm is always used (better than weighted combining)

For any other issues, please check the [troubleshooting guide](troubleshooting.md) or file an issue.