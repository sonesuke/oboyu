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

## Configuration Templates

### Academic Research Setup
```yaml
# Optimized for research papers, theses, and academic documents
indexer:
  db_path: "~/research/index.db"
  chunk_size: 2048              # Larger chunks for academic papers
  chunk_overlap: 512            # More overlap for context preservation
  embedding_model: "cl-nagoya/ruri-v3-30m"  # Good for mixed content
  use_reranker: true            # Important for precise retrieval

crawler:
  include_patterns:
    # - "*.pdf"                   # Not supported - requires proper extraction
    - "*.tex"                   # LaTeX documents (as text)
    - "*.bib"                   # Bibliography files
    - "*.md"                    # Notes and documentation
    - "*.txt"                   # Plain text documents
    # - "*.docx"                  # Not supported - requires proper extraction
  exclude_patterns:
    - "*/backup/*"
    - "*/drafts/*"
    - "*/temp/*"
    - "*/.git/*"

query:
  default_mode: "hybrid"        # Best for technical content
  top_k: 10                     # More results for research
```

### Software Documentation Indexing
```yaml
# Optimized for API docs, README files, and technical documentation
indexer:
  db_path: "~/dev/docs_index.db"
  chunk_size: 1536              # Balanced for code examples and text
  chunk_overlap: 384            # Good overlap for context
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true            # Better code understanding

crawler:
  include_patterns:
    - "*.md"                    # Markdown documentation
    - "*.rst"                   # reStructuredText
    - "*.adoc"                  # AsciiDoc
    - "README*"                 # All README files
    - "CHANGELOG*"              # Changelog files
    # - "*.ipynb"                 # Note: Jupyter notebooks are treated as plain text only
    - "docs/**/*"               # Documentation folders
    - "*.html"                  # Generated docs
  exclude_patterns:
    - "*/node_modules/*"
    - "*/venv/*"
    - "*/.git/*"
    - "*/site-packages/*"
    - "*/dist/*"
    - "*/build/*"
    - "*/_build/*"              # Sphinx build directory

query:
  default_mode: "hybrid"
  top_k: 8
```

### Multilingual Document Collection
```yaml
# Optimized for documents in multiple languages
indexer:
  db_path: "~/multilingual/index.db"
  chunk_size: 1024              # Smaller chunks for varied languages
  chunk_overlap: 256            # Standard overlap
  embedding_model: "cl-nagoya/ruri-v3-30m"  # Supports multiple languages
  use_reranker: true            # Important for cross-lingual search

crawler:
  include_patterns:
    - "*.txt"
    - "*.md"
    # - "*.pdf"                   # Not supported - requires proper extraction
    # - "*.docx"                  # Not supported - requires proper extraction
    # - "*.odt"                   # Not supported - requires proper extraction
    - "*.rtf"
  exclude_patterns:
    - "*/temp/*"
    - "*/.git/*"
    - "*/cache/*"

query:
  default_mode: "hybrid"        # Works well across languages
  top_k: 10
```

### Large-Scale Indexing
```yaml
# Optimized for indexing large document collections (100k+ files)
indexer:
  db_path: "/data/large_index.db"  # Use fast SSD storage
  chunk_size: 1024              # Smaller chunks for memory efficiency
  chunk_overlap: 128            # Minimal overlap to save space
  embedding_model: "cl-nagoya/ruri-v3-30m"  # Efficient model
  use_reranker: false           # Disable for initial indexing speed

crawler:
  include_patterns:
    - "*.txt"
    - "*.md"
    # - "*.pdf"                   # Not supported - requires proper extraction
    - "*.html"
    - "*.xml"
    - "*.json"
  exclude_patterns:
    - "*/logs/*"
    - "*/cache/*"
    - "*/tmp/*"
    - "*/.git/*"
    - "*.log"
    - "*.bak"
    - "*.tmp"

query:
  default_mode: "bm25"          # Faster for large collections
  top_k: 20                     # Pre-filter more before reranking
```

### Performance-Optimized Setup
```yaml
# Optimized for maximum search speed
indexer:
  db_path: "/fast/ssd/index.db"  # Use SSD for best performance
  chunk_size: 512               # Smaller chunks for faster processing
  chunk_overlap: 64             # Minimal overlap
  embedding_model: "cl-nagoya/ruri-v3-30m"  # Fast model
  use_reranker: false           # Disable for speed

crawler:
  include_patterns:
    - "*.txt"
    - "*.md"
  exclude_patterns:
    - "*"                       # Very selective indexing

query:
  default_mode: "bm25"          # Fastest search mode
  top_k: 5                      # Fewer results for speed
```

## Practical Examples

### Example 1: Indexing a Python Project
```yaml
# Index Python project with documentation
indexer:
  db_path: "~/projects/myproject/.oboyu/index.db"
  chunk_size: 1024
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true

crawler:
  include_patterns:
    - "*.py"                    # Python source files
    - "*.pyi"                   # Type stub files
    - "*.md"                    # Documentation
    - "*.rst"                   # reStructuredText docs
    # - "*.ipynb"                 # Note: Jupyter notebooks are treated as plain text only
    - "requirements*.txt"       # Dependencies
    - "pyproject.toml"          # Project config
    - "setup.py"                # Setup files
    - "setup.cfg"
  exclude_patterns:
    - "*/__pycache__/*"
    - "*.pyc"
    - "*/.pytest_cache/*"
    - "*/.mypy_cache/*"
    - "*/venv/*"
    - "*/env/*"
    - "*/.venv/*"
    - "*/site-packages/*"
    - "*/.tox/*"
    - "*/dist/*"
    - "*/build/*"
    - "*.egg-info/*"

query:
  default_mode: "hybrid"
  top_k: 8
```

### Example 2: Personal Knowledge Base
```yaml
# Index personal notes, documents, and references
indexer:
  db_path: "~/Documents/.oboyu/knowledge.db"
  chunk_size: 1536              # Good for mixed content
  chunk_overlap: 384
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true

crawler:
  include_patterns:
    - "*.md"                    # Markdown notes
    - "*.txt"                   # Text files
    # - "*.pdf"                   # Not supported - requires proper extraction
    # - "*.docx"                  # Not supported - requires proper extraction
    - "*.org"                   # Org-mode files
    - "*.rtf"                   # Rich text
    - "notes/**/*"              # Notes directory
    - "journal/**/*"            # Journal entries
  exclude_patterns:
    - "*/Archive/*"             # Archived content
    - "*/.obsidian/*"           # Obsidian config
    - "*/.trash/*"              # Trash folders
    - "*.tmp"
    - "~$*"                     # Temporary Word files

query:
  default_mode: "hybrid"
  top_k: 10
```

### Example 3: Legal Document Repository
```yaml
# Index legal documents with high precision requirements
indexer:
  db_path: "/secure/legal_index.db"
  chunk_size: 2048              # Larger chunks for context
  chunk_overlap: 512            # More overlap for precision
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true            # Critical for accuracy

crawler:
  include_patterns:
    # - "*.pdf"                   # Not supported - requires proper extraction
    # - "*.docx"                  # Not supported - requires proper extraction
    - "*.txt"                   # Plain text
    - "contracts/**/*"          # Contracts folder
    - "cases/**/*"              # Case files
    - "statutes/**/*"           # Legal statutes
  exclude_patterns:
    - "*/drafts/*"              # Draft documents
    - "*/temp/*"                # Temporary files
    - "*.bak"                   # Backup files

query:
  default_mode: "hybrid"        # Maximum accuracy
  top_k: 15                     # More results for thorough review
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

## Environment Variables

Oboyu supports environment variables for all essential configuration options. Environment variables take precedence over configuration file values.

### Syntax Pattern
All environment variables follow the pattern: `OBOYU_<SECTION>_<OPTION>`

### Available Environment Variables

```bash
# Indexer settings
export OBOYU_INDEXER_DB_PATH="/custom/path/index.db"
export OBOYU_INDEXER_CHUNK_SIZE="1024"
export OBOYU_INDEXER_CHUNK_OVERLAP="256"
export OBOYU_INDEXER_EMBEDDING_MODEL="cl-nagoya/ruri-v3-30m"
export OBOYU_INDEXER_USE_RERANKER="true"

# Crawler settings
export OBOYU_CRAWLER_INCLUDE_PATTERNS="*.md,*.txt,*.py"  # Comma-separated
export OBOYU_CRAWLER_EXCLUDE_PATTERNS="*/node_modules/*,*/.git/*"

# Query settings
export OBOYU_QUERY_DEFAULT_MODE="hybrid"
export OBOYU_QUERY_TOP_K="10"

# Global settings
export OBOYU_CONFIG_PATH="/path/to/config.yaml"  # Override config file location
export OBOYU_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
export OBOYU_CACHE_DIR="~/.cache/oboyu"  # Model cache directory
```

### Usage Examples

```bash
# Temporary override for a single command
OBOYU_INDEXER_CHUNK_SIZE=2048 oboyu index /path/to/docs

# Set for current session
export OBOYU_QUERY_TOP_K=20
oboyu query "search term"

# Permanent setup (add to ~/.bashrc or ~/.zshrc)
echo 'export OBOYU_INDEXER_DB_PATH="~/my_indices/main.db"' >> ~/.bashrc
```

### Docker Usage
```dockerfile
# In Dockerfile
ENV OBOYU_INDEXER_DB_PATH=/data/index.db
ENV OBOYU_INDEXER_CHUNK_SIZE=1024
ENV OBOYU_QUERY_DEFAULT_MODE=hybrid
```

### CI/CD Usage
```yaml
# GitHub Actions example
env:
  OBOYU_INDEXER_USE_RERANKER: "false"  # Faster for CI
  OBOYU_QUERY_TOP_K: "5"
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
1. Check the [troubleshooting guide](troubleshooting.md)
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