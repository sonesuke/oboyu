# Oboyu Configuration

Oboyu has been significantly simplified to provide a better user experience. The configuration system now focuses on essential options while automatically optimizing advanced parameters behind the scenes.

## Quick Start

For most users, Oboyu works great with zero configuration. Simply run:

```bash
oboyu index /path/to/documents
oboyu query --query "your search"
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
  include_patterns:                     # File types to index (text-based only)
    - "*.txt"                          # Plain text files
    - "*.md"                           # Markdown documents
    - "*.py"                           # Python source code
    - "*.rst"                          # reStructuredText
    - "*.java"                         # Java source code
    - "*.html"                         # HTML files
    # Note: Binary formats like PDF, Word docs, Excel files are NOT supported
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
- `snippet_length` - Use `--snippet-length` flag
- `language_filter` - Use `--language` flag

### Replaced by Better Algorithms
- `change_detection_strategy` - Smart default handles all cases

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

## Validation Rules and Constraints

### Configuration Option Constraints

#### indexer.db_path
- **Type**: String (file path)
- **Constraints**: 
  - Must be a valid file path
  - Directory must be writable
  - Supports `~` expansion for home directory
- **Performance Impact**: Use SSD storage for best performance

#### indexer.chunk_size
- **Type**: Integer
- **Constraints**: 
  - Minimum: 128
  - Maximum: 8192
  - Recommended: 512-2048
- **Validation**: Must be power of 2 for optimal performance
- **Performance Impact**: 
  - Smaller chunks (512): More precise retrieval, higher memory usage
  - Larger chunks (2048): Faster indexing, less precise retrieval

#### indexer.chunk_overlap
- **Type**: Integer
- **Constraints**: 
  - Minimum: 0
  - Maximum: chunk_size / 2
  - Recommended: chunk_size / 4
- **Validation**: Must be less than chunk_size
- **Performance Impact**: 
  - More overlap: Better context preservation, larger index size
  - Less overlap: Smaller index, possible context loss at boundaries

#### indexer.embedding_model
- **Type**: String
- **Constraints**: 
  - Must be a valid Hugging Face model ID
  - Model must support sentence embeddings
  - Recommended: "cl-nagoya/ruri-v3-30m"
- **Validation**: Checked against Hugging Face model registry
- **Performance Impact**: 
  - Larger models: Better quality, slower indexing
  - Smaller models: Faster processing, may reduce quality

#### indexer.use_reranker
- **Type**: Boolean
- **Constraints**: true or false
- **Performance Impact**: 
  - true: Better search quality, 20-30% slower
  - false: Faster searches, may miss relevant results

#### crawler.include_patterns
- **Type**: List of strings (glob patterns)
- **Constraints**: 
  - Must be valid glob patterns
  - At least one pattern required
- **Validation**: Patterns tested against file system
- **Performance Impact**: More specific patterns = faster crawling

#### crawler.exclude_patterns
- **Type**: List of strings (glob patterns)
- **Constraints**: 
  - Must be valid glob patterns
  - Optional (empty list allowed)
- **Validation**: Patterns tested against file system
- **Performance Impact**: More exclusions = faster crawling

#### query.default_mode
- **Type**: String (enum)
- **Constraints**: 
  - Must be one of: "vector", "bm25", "hybrid"
- **Validation**: Checked against allowed values
- **Performance Impact**: 
  - vector: Semantic search, moderate speed
  - bm25: Keyword search, fastest
  - hybrid: Best quality, slowest

#### query.top_k
- **Type**: Integer
- **Constraints**: 
  - Minimum: 1
  - Maximum: 100
  - Recommended: 5-20
- **Validation**: Must be positive integer
- **Performance Impact**: Linear with value (2x top_k = ~2x time)

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
```

## Performance Impact Notes

### Indexing Performance

| Setting | Impact | Recommendation |
|---------|--------|----------------|
| `chunk_size` | ↓ size = ↑ memory, ↑ precision | 1024 for balance |
| `chunk_overlap` | ↑ overlap = ↑ index size | 25% of chunk_size |
| `embedding_model` | Larger = slower but better | ruri-v3-30m for balance |
| `use_reranker` | +20-30% index time | Enable for quality |
| File patterns | More specific = faster | Be selective |

### Query Performance

| Setting | Impact | Recommendation |
|---------|--------|----------------|
| `default_mode` | hybrid > vector > bm25 | hybrid for quality |
| `top_k` | Linear scaling | 5-10 for most cases |
| Reranker enabled | +50-100ms latency | Worth it for quality |
| Database location | SSD > HDD (10x faster) | Always use SSD |

### Memory Usage

| Component | Typical Usage | Scaling Factor |
|-----------|--------------|----------------|
| Indexing | 2-4 GB | ~1GB per 100k chunks |
| Embedding model | 500MB-2GB | Depends on model size |
| Query runtime | 500MB-1GB | Mostly model loading |
| Database | 10-50 MB/1000 docs | Depends on content |

### Optimization Tips

1. **For Speed**: 
   - Use BM25 mode
   - Disable reranker
   - Smaller chunk_size
   - SSD storage

2. **For Quality**:
   - Use hybrid mode
   - Enable reranker
   - Larger chunk_size
   - More chunk_overlap

3. **For Large Scale**:
   - Minimal chunk_overlap
   - Specific file patterns
   - Fast SSD storage
   - Consider sharding

## Configuration Error Troubleshooting

### Common Configuration Errors

#### "Invalid configuration file"
**Cause**: YAML syntax error
**Solution**: 
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

#### "Model not found"
**Cause**: Invalid embedding model name
**Solution**: 
- Check model exists on Hugging Face
- Ensure internet connection for download
- Try default: `cl-nagoya/ruri-v3-30m`

#### "Permission denied for db_path"
**Cause**: No write permissions
**Solution**:
```bash
# Check permissions
ls -la $(dirname ~/oboyu/index.db)
# Create directory with proper permissions
mkdir -p ~/oboyu && chmod 755 ~/oboyu
```

#### "Chunk size validation failed"
**Cause**: Invalid chunk_size value
**Solution**:
- Ensure value is between 128 and 8192
- Use power of 2 for best performance
- Start with 1024 if unsure

#### "Pattern matching no files"
**Cause**: Include patterns too restrictive
**Solution**:
```bash
# Test patterns
find /path -name "*.md" -o -name "*.txt"
# Use broader patterns
include_patterns: ["*"]  # Then add exclusions
```

### Debugging Configuration

```bash
# Show effective configuration
oboyu config show

# Validate configuration
oboyu config validate

# Test with minimal config
oboyu index /path --config minimal.yaml

# Enable debug logging
OBOYU_LOG_LEVEL=DEBUG oboyu index /path
```

### Getting Help

If configuration issues persist:
1. Check the [troubleshooting guide](../troubleshooting/troubleshooting.md)
2. Review [examples](#configuration-examples) above
3. Run diagnostic: `oboyu diagnose`
4. File an issue with config and error output

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
   oboyu query --query "test search"
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
| `query.show_scores` | Runtime flag | `--show-scores` |

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

For any other issues, please check the [troubleshooting guide](../troubleshooting/troubleshooting.md) or file an issue.