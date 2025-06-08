# Oboyu CLI Commands

This document describes the available command-line interface (CLI) commands for Oboyu.

## Quick Command Reference

| Command | Description | Common Options |
|---------|-------------|----------------|
| `oboyu version` | Display version information | - |
| `oboyu mcp` | Start MCP server for AI integration | `--transport`, `--port` |
| `oboyu clear` | Clear index database | `--force` |
| `oboyu index <path>` | Index documents | `--force`, `--chunk-size`, `--include-patterns` |
| `oboyu index manage status` | Show indexing status | `--detailed` |
| `oboyu index manage diff` | Preview indexing changes | `--change-detection` |
| `oboyu index manage clear` | Clear index data | `--force` |
| `oboyu query <text>` | Search indexed documents | `--mode`, `--top-k`, `--rerank` |
| `oboyu query --interactive` | Interactive search session | `--mode`, `--rerank` |

### Common Workflows

```bash
# Quick start: Index a codebase and search
oboyu index ~/projects/myapp --include-patterns "*.py,*.js"
oboyu query "database connection" --top-k 10

# Incremental updates with cleanup
oboyu index ~/documents --cleanup-deleted
oboyu index manage status ~/documents --detailed

# Interactive exploration with reranking
oboyu query --interactive --rerank --mode hybrid
```

## Global Options

The following options are available for all commands:

- `--config`, `-c`: Path to configuration file (see [Configuration Guide](configuration.md))
- `--db-path`: Path to database file
- `--verbose`, `-v`: Enable verbose output

**Note:** If you're upgrading from an older version, see the [Configuration Migration Guide](immutable-configuration-migration.md) for information about the new immutable configuration system.

## Main Commands

### `oboyu version`

Display the current version of Oboyu.

```bash
oboyu version
```

### `oboyu mcp`

Run an MCP (Model Context Protocol) server for AI assistant integration.

```bash
# Start MCP server with stdio transport (default)
oboyu mcp

# Start with specific transport
oboyu mcp --transport stdio

# Start with HTTP transport on specific port
oboyu mcp --transport streamable-http --port 8000

# Start with SSE transport
oboyu mcp --transport sse --port 8001

# Start with custom database path and verbose output
oboyu mcp --db-path /path/to/custom.db --verbose

# Start with debug mode for development
oboyu mcp --verbose --debug
```

Options:
- `--transport`, `-t`: Transport mechanism (stdio, sse, streamable-http) - default: stdio
- `--port`, `-p`: Port number for HTTP or SSE transport (required for non-stdio transports)
- `--db-path`: Path to database file - default: ~/.oboyu/index.db
- `--verbose`, `-v`: Enable verbose output
- `--debug`, `-d`: Enable debug mode with additional logging

**Transport Types:**
- `stdio`: Standard input/output communication (for direct AI assistant integration)
- `sse`: Server-Sent Events over HTTP (for web-based integrations)
- `streamable-http`: HTTP with streaming support (for network-based integrations)

### `oboyu clear`

Clear all data from the index database while preserving the database schema and structure.

```bash
# Clear with confirmation prompt
oboyu clear

# Force clear without confirmation
oboyu clear --force

# Clear a specific database
oboyu clear --db-path custom.db
```

Options:
- `--force`, `-f`: Force operation without confirmation
- `--db-path`: Path to database file to clear

### `oboyu index`

Index documents for search.

```bash
# Index a single directory
oboyu index /path/to/documents

# Index multiple directories
oboyu index /path/to/docs1 /path/to/docs2

# Index with specific file patterns
oboyu index /path/to/documents --include-patterns "*.txt,*.md"

# Force re-index of all documents (non-incremental)
oboyu index /path/to/documents --force

# Index with custom chunking settings
oboyu index /path/to/documents --chunk-size 2048 --chunk-overlap 512


# Index with change detection strategy
oboyu index /path/to/documents --change-detection hash

# Index with cleanup of deleted files
oboyu index /path/to/documents --cleanup-deleted

# Quiet progress output (minimal screen updates)
oboyu index /path/to/documents --quiet-progress
```

Options:
- `--recursive/--no-recursive`: Process directories recursively (default: recursive)
- `--include-patterns`: File patterns to include (e.g., `*.txt,*.md`)
- `--exclude-patterns`: File patterns to exclude (e.g., `*/node_modules/*`)
- `--max-depth`: Maximum recursion depth
- `--force`, `-f`: Force re-index of all documents (disable incremental indexing)
- `--encoding-detection/--no-encoding-detection`: Enable/disable automatic encoding detection (default: enabled)
- `--chunk-size`: Chunk size in characters (default: 1024)
- `--chunk-overlap`: Chunk overlap in characters (default: 256)
- `--embedding-model`: Embedding model to use (default: cl-nagoya/ruri-v3-30m)
- `--db-path`: Path to database file
- `--change-detection`: Strategy for detecting changes (timestamp, hash, smart) - default: smart
- `--cleanup-deleted/--no-cleanup-deleted`: Remove deleted files from index during incremental update
- `--verify-integrity`: Verify file integrity using content hashes (slower but more accurate)
- `--quiet-progress`, `-q`: Minimal progress output to avoid screen flickering

#### Practical Examples

**Index a Python codebase excluding tests and cache:**
```bash
oboyu index ~/projects/myapp \
  --include-patterns "*.py" \
  --exclude-patterns "*test*,*__pycache__*,*.pyc" \
  --chunk-size 2048
```

**Index documentation with language detection:**
```bash
oboyu index ~/docs \
  --include-patterns "*.md,*.rst,*.txt" \
  --encoding-detection \
  --chunk-overlap 512
```

**Fast incremental update for large repositories:**
```bash
# First time: full index with hash-based change detection
oboyu index ~/large-repo --change-detection hash

# Daily updates: incremental with cleanup
oboyu index ~/large-repo --cleanup-deleted --quiet-progress
```

**Index multiple project directories:**
```bash
# Index several projects at once
oboyu index ~/work/project1 ~/work/project2 ~/personal/blog \
  --recursive \
  --max-depth 5
```

## Index Management Commands

### `oboyu index manage clear`

Clear all data from the index database (alternative to `oboyu clear`).

```bash
# Clear with confirmation prompt
oboyu index manage clear

# Force clear without confirmation
oboyu index manage clear --force

# Clear specific database
oboyu index manage clear --db-path custom.db
```

Options:
- `--force`, `-f`: Force clearing without confirmation
- `--db-path`: Path to database file to clear

### `oboyu index manage status`

Show indexing status for specified directories.

```bash
# Show basic status for directories
oboyu index manage status /path/to/docs

# Show detailed file-by-file status
oboyu index manage status /path/to/docs --detailed

# Check status with custom database
oboyu index manage status /path/to/docs --db-path custom.db
```

Options:
- `--detailed`, `-d`: Show detailed file-by-file status
- `--db-path`: Path to database file

### `oboyu index manage diff`

Show what would be updated if indexing were run now (dry-run).

```bash
# Show what would change
oboyu index manage diff /path/to/docs

# Use specific change detection strategy
oboyu index manage diff /path/to/docs --change-detection hash

# Check diff with custom database
oboyu index manage diff /path/to/docs --db-path custom.db
```

Options:
- `--change-detection`: Strategy for detecting changes (timestamp, hash, smart) - default: smart
- `--db-path`: Path to database file

### `oboyu query`

Search indexed documents.

```bash
# Basic query
oboyu query "search term"

# Specify search mode
oboyu query "search term" --mode vector

# Get more results
oboyu query "search term" --top-k 10

# Use hybrid search with custom weights
oboyu query "search term" --mode hybrid --vector-weight 0.8 --bm25-weight 0.2

# Enable reranking for better accuracy
oboyu query "search term" --rerank

# Disable reranking for faster results
oboyu query "search term" --no-rerank

# Get detailed explanation of results
oboyu query "search term" --explain

# Start interactive search session
oboyu query --interactive

# Interactive mode with specific settings
oboyu query --interactive --mode hybrid --rerank
```

Options:
- `--mode`: Search mode (vector, bm25, hybrid) - default: hybrid
- `--top-k`: Number of results to return - default: 5
- `--explain`: Show detailed explanation of results
- `--format`: Output format (text, json) - default: text
- `--vector-weight`: Weight for vector scores in hybrid search (0.0-1.0) - default: 0.7
- `--bm25-weight`: Weight for BM25 scores in hybrid search (0.0-1.0) - default: 0.3
- `--db-path`: Path to database file
- `--rerank/--no-rerank`: Enable or disable reranking of search results - default: enabled
- `--interactive`: Start interactive search session for continuous queries

#### Query Examples

**Search for function definitions in code:**
```bash
# Find function definitions with specific names
oboyu query "def process_data" --mode bm25 --top-k 20

# Search for class implementations
oboyu query "class DatabaseConnection" --explain
```

**Multi-language search with reranking:**
```bash
# Search across English and Japanese content
oboyu query "machine learning 機械学習" --rerank --top-k 10
```

**Export results for processing:**
```bash
# Get JSON output for scripting
oboyu query "error handling" --format json | jq '.results[].file_path'

# Save results to file
oboyu query "TODO" --top-k 50 --format json > todos.json
```

**Fine-tuned hybrid search:**
```bash
# Emphasize semantic similarity
oboyu query "authentication flow" \
  --mode hybrid \
  --vector-weight 0.9 \
  --bm25-weight 0.1

# Balance keyword and semantic search
oboyu query "database optimization techniques" \
  --mode hybrid \
  --vector-weight 0.5 \
  --bm25-weight 0.5
```

#### Interactive Mode

The interactive mode provides a powerful REPL (Read-Eval-Print Loop) interface for performing multiple searches without reloading models and database. This is particularly beneficial when using rerankers, which have significant initialization overhead.

**Key Features:**
- **Model Persistence**: Models stay loaded between queries for faster subsequent searches
- **Command History**: Previous queries are saved and can be recalled with arrow keys
- **Auto-Suggestions**: Tab completion and history-based suggestions
- **Real-time Configuration**: Change search settings without restarting
- **Session State**: Settings persist throughout the session
- **Rich Output**: Colorized results and status information

**Starting Interactive Mode:**

```bash
# Basic interactive mode
oboyu query --interactive

# Interactive mode with reranker pre-loaded
oboyu query --interactive --rerank

# Interactive mode with custom settings
oboyu query --interactive --mode hybrid --top-k 10 --vector-weight 0.8
```

**Interactive Commands:**

All interactive commands start with `/` to distinguish them from search queries.

| Command | Description | Example | Notes |
|---------|-------------|---------|-------|
| `<query>` | Search for documents | `machine learning algorithms` | Any text without `/` prefix |
| `/help` | Show available commands | `/help` | Displays command reference |
| `/exit`, `/quit`, `/q` | Exit interactive mode | `/exit` | Graceful shutdown |
| `/mode <mode>` | Change search mode | `/mode vector` | Options: vector, bm25, hybrid |
| `/topk <number>` | Change number of results | `/topk 10` | Must be positive integer |
| `/top-k <number>` | Alias for topk | `/top-k 15` | Same as `/topk` |
| `/weights <v> <b>` | Change hybrid weights | `/weights 0.8 0.2` | Vector weight, BM25 weight (0.0-1.0) |
| `/rerank on/off` | Toggle reranker | `/rerank on` | Enable or disable reranking |
| `/settings` | Show current settings | `/settings` | Display all current configuration |
| `/clear` | Clear screen | `/clear` | Unix-style screen clearing |
| `/stats` | Show index statistics | `/stats` | Database and index information |

**Example Interactive Session:**

```bash
$ oboyu query --interactive --rerank

🔍 Oboyu Interactive Search
📊 Mode: hybrid | Top-K: 5 | RRF-K: 60 | Reranker: enabled

✅ Ready for search!
Type your search query (or '/help' for commands, '/exit' to quit):

> machine learning algorithms
🔍 Searching...
📊 Found 3 results in 0.12 seconds

• Deep Learning Fundamentals (Score: 0.89)
  This chapter covers the basic principles of deep learning, including neural networks,
  backpropagation, and optimization algorithms used in machine learning...
  Source: /docs/ml-guide.md (en)

• Algorithm Comparison Study (Score: 0.84)
  A comprehensive comparison of supervised learning algorithms including decision trees,
  random forests, support vector machines, and neural networks...
  Source: /research/algorithms.txt (en)

• 機械学習アルゴリズムの概要 (Score: 0.78)
  機械学習の主要なアルゴリズムについて説明します。線形回帰、ロジスティック回帰、
  ランダムフォレスト、ニューラルネットワークなど...
  Source: /docs/ml-overview-ja.md (ja)

> /mode vector
✅ Search mode changed to: vector

> /topk 10
✅ Top-K changed to: 10

> /weights 0.9 0.1
✅ Weights changed to: Vector=0.9, BM25=0.1

> neural networks
🔍 Searching...
📊 Found 7 results in 0.08 seconds

[More results with vector-focused search...]

> /rerank off
✅ Reranker disabled

> /settings

Current Settings:
- Mode: vector
- Top-K: 10
- Vector weight: 0.9
- BM25 weight: 0.1
- Reranker: disabled
- Database: /home/user/.oboyu/index.db

> /stats

Index Statistics:
- Total documents: 1,247
- Total chunks: 8,934
- Unique files: 1,247
- Database size: 89.32 MB

> /exit
👋 Goodbye!
```

**Performance Benefits:**

Interactive mode provides significant performance advantages:

- **Model Loading**: Embedding models are loaded once at startup (typically 2-5 seconds)
- **Reranker Warmup**: Reranker models are pre-loaded and warmed up (typically 3-8 seconds)
- **Database Connection**: Database stays connected and indexed for faster queries
- **Query Speed**: Subsequent queries run 5-10x faster than CLI mode
- **Memory Efficiency**: Models stay in memory, avoiding repeated loading

**Tips for Interactive Mode:**

1. **Use Tab Completion**: Press Tab to see available commands and complete partially typed commands
2. **Command History**: Use ↑/↓ arrow keys to recall previous queries
3. **Quick Exits**: Use Ctrl+D or Ctrl+C as alternative exit methods
4. **Screen Management**: Use `/clear` to clean up the display during long sessions
5. **Settings Monitoring**: Use `/settings` regularly to verify your current configuration
6. **Performance Monitoring**: Use `/stats` to monitor index growth and database size

**When to Use Interactive Mode:**

- **Exploratory Search**: When you need to try multiple related queries
- **Research Sessions**: Extended periods of document searching
- **Reranker Usage**: When reranker accuracy is needed for multiple queries
- **Parameter Tuning**: Testing different search modes and RRF parameters
- **Large Datasets**: When model loading time becomes significant
- **Iterative Refinement**: Progressively refining search queries

## Health Monitoring Commands

Oboyu provides health monitoring commands to check system status, track operations, and debug issues.

### `oboyu health check`

Quick health check of the system status.

```bash
# Basic health check
oboyu health check

# Health check with JSON output
oboyu health check --format json

# Health check with custom database
oboyu health check --db-path custom.db
```

Options:
- `--format`: Output format (table, json) - default: table
- `--db-path`: Path to database file

### `oboyu health events`

Show recent events for debugging and monitoring.

```bash
# Show events from last 24 hours
oboyu health events

# Show events from last 48 hours
oboyu health events --hours 48

# Filter by event type
oboyu health events --event-type indexing

# Use custom event database
oboyu health events --event-db /path/to/events.db

# JSON output for processing
oboyu health events --format json
```

Options:
- `--hours`: Hours of events to show - default: 24
- `--event-type`: Filter by event type (indexing, search, etc.)
- `--event-db`: Path to event database
- `--format`: Output format (table, json) - default: table

### `oboyu health timeline`

Show timeline of events for a specific operation.

```bash
# Show timeline for an operation ID
oboyu health timeline op_12345

# Timeline with JSON output
oboyu health timeline op_12345 --format json

# Use custom event database
oboyu health timeline op_12345 --event-db /path/to/events.db
```

Arguments:
- `operation-id`: The operation ID to show timeline for (required)

Options:
- `--event-db`: Path to event database
- `--format`: Output format (table, json) - default: table

### `oboyu health operations`

Show recent operations for debugging.

```bash
# Show last 20 operations
oboyu health operations

# Show last 50 operations
oboyu health operations --limit 50

# Operations with JSON output
oboyu health operations --format json
```

Options:
- `--limit`: Number of recent operations to show - default: 20
- `--format`: Output format (table, json) - default: table

### Health Monitoring Examples

**Monitor indexing health:**
```bash
# Check system health before indexing
oboyu health check

# Index documents
oboyu index ~/documents

# Check recent operations
oboyu health operations --limit 5

# View indexing events
oboyu health events --event-type indexing --hours 1
```

**Debug failed operations:**
```bash
# Find failed operations
oboyu health operations --format json | jq '.[] | select(.status == "failed")'

# Get timeline for failed operation
oboyu health timeline op_failed_123

# View all events around failure time
oboyu health events --hours 2
```

**Automated monitoring script:**
```bash
#!/bin/bash
# Health check monitoring script

# Run health check
if ! oboyu health check --format json > /tmp/health.json; then
    echo "Health check failed!"
    exit 1
fi

# Check for recent failures
failed_ops=$(oboyu health operations --format json | \
    jq '[.[] | select(.status == "failed")] | length')

if [ "$failed_ops" -gt 0 ]; then
    echo "Warning: $failed_ops failed operations in recent history"
    oboyu health operations --limit 10
fi
```

## Exit Codes

Oboyu uses standard exit codes to indicate the result of operations:

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | Success | Command completed successfully |
| 1 | General Error | Unspecified error occurred |
| 2 | Invalid Arguments | Command line arguments were invalid |
| 3 | File Not Found | Specified file or directory doesn't exist |
| 4 | Permission Denied | Insufficient permissions for operation |
| 5 | Database Error | Database operation failed |
| 6 | Index Error | Indexing operation failed |
| 7 | Query Error | Search query failed |
| 8 | Configuration Error | Invalid configuration |
| 9 | Model Loading Error | Failed to load ML models |
| 10 | Network Error | Network operation failed (MCP server) |

### Using Exit Codes in Scripts

```bash
#!/bin/bash
# Script that indexes and searches with error handling

# Index documents
if oboyu index ~/documents --quiet-progress; then
    echo "Indexing successful"
else
    exit_code=$?
    case $exit_code in
        3) echo "Error: Directory not found" ;;
        4) echo "Error: Permission denied" ;;
        6) echo "Error: Indexing failed" ;;
        *) echo "Error: Unknown error (code: $exit_code)" ;;
    esac
    exit $exit_code
fi

# Search with error handling
oboyu query "important document" --format json > results.json
if [ $? -eq 0 ]; then
    echo "Found $(jq '.total' results.json) results"
else
    echo "Search failed"
    exit 1
fi
```

## Shell Completion

Oboyu supports shell completion for bash, zsh, and fish shells to help you type commands faster.

### Bash

Add to your `~/.bashrc`:

```bash
# Oboyu bash completion
eval "$(_OBOYU_COMPLETE=bash_source oboyu)"
```

Or create a completion file:

```bash
_OBOYU_COMPLETE=bash_source oboyu > ~/.local/share/bash-completion/completions/oboyu
```

### Zsh

Add to your `~/.zshrc`:

```zsh
# Oboyu zsh completion
eval "$(_OBOYU_COMPLETE=zsh_source oboyu)"
```

Or create a completion file:

```zsh
_OBOYU_COMPLETE=zsh_source oboyu > ~/.zfunc/_oboyu
```

Make sure `~/.zfunc` is in your `fpath`:

```zsh
fpath=(~/.zfunc $fpath)
autoload -Uz compinit && compinit
```

### Fish

Create a completion file:

```fish
_OBOYU_COMPLETE=fish_source oboyu > ~/.config/fish/completions/oboyu.fish
```

### Testing Completion

After setting up completion, test it:

```bash
# Type and press TAB
oboyu <TAB>
oboyu index --<TAB>
oboyu query --mode <TAB>
```

## Troubleshooting

### Common CLI Errors and Solutions

#### "Database locked" error

**Problem:** Multiple Oboyu processes trying to access the same database.

**Solution:**
```bash
# Find and kill other Oboyu processes
ps aux | grep oboyu
kill <PID>

# Or use a different database
oboyu index ~/docs --db-path ~/alternative.db
```

#### "Model loading failed" error

**Problem:** ML models couldn't be downloaded or loaded.

**Solution:**
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/hub/*oboyu*

# Retry with verbose output
oboyu query "test" --verbose

# Use a different model
oboyu index ~/docs --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

#### "Out of memory" during indexing

**Problem:** Large files causing memory issues.

**Solution:**
```bash
# Reduce chunk size
oboyu index ~/large-docs --chunk-size 512 --chunk-overlap 128

# Process fewer files at once
oboyu index ~/large-docs --max-depth 2

# Use quiet progress to reduce memory overhead
oboyu index ~/large-docs --quiet-progress
```

#### "Permission denied" errors

**Problem:** Insufficient permissions for files or database.

**Solution:**
```bash
# Check file permissions
ls -la ~/.oboyu/

# Fix database permissions
chmod 644 ~/.oboyu/index.db

# Use a different database location
export OBOYU_DB_PATH=~/my-oboyu.db
oboyu index ~/docs
```

#### Slow incremental indexing

**Problem:** Incremental indexing taking too long.

**Solution:**
```bash
# Check what would be updated
oboyu index manage diff ~/docs

# Use hash-based detection for accuracy
oboyu index ~/docs --change-detection hash

# Or use timestamp for speed
oboyu index ~/docs --change-detection timestamp
```

## Tips and Tricks

### Advanced Usage Patterns

#### 1. Parallel Indexing of Multiple Projects

```bash
# Index multiple projects in parallel
for dir in ~/projects/*; do
    oboyu index "$dir" --db-path "${dir##*/}.db" &
done
wait

# Merge databases (future feature)
```

#### 2. Scheduled Incremental Updates

Create a cron job for automatic updates:

```bash
# Add to crontab -e
# Update index every night at 2 AM
0 2 * * * /usr/local/bin/oboyu index ~/documents --cleanup-deleted --quiet-progress
```

#### 3. Search Aliases for Common Queries

Add to your shell configuration:

```bash
# Quick search functions
alias todos='oboyu query "TODO|FIXME" --mode bm25 --top-k 50'
alias errors='oboyu query "error|exception|traceback" --rerank'
alias recent='oboyu query --interactive --mode hybrid'

# Project-specific search
function search_project() {
    oboyu query "$1" --db-path ~/projects/${2:-current}.db
}
```

#### 4. Integration with Development Tools

```bash
# Find and open files in editor
oboyu query "$1" --format json | \
    jq -r '.results[0].file_path' | \
    xargs code

# Search and preview with bat
oboyu query "$1" --format json | \
    jq -r '.results[].file_path' | \
    xargs bat --paging=never
```

#### 5. Performance Optimization

```bash
# Benchmark different search modes
time oboyu query "database connection" --mode vector --no-rerank
time oboyu query "database connection" --mode bm25 --no-rerank
time oboyu query "database connection" --mode hybrid --rerank

# Profile indexing performance
time oboyu index ~/large-codebase --force
time oboyu index ~/large-codebase  # incremental
```

#### 6. Custom Database Management

```bash
# Maintain separate databases for different contexts
export WORK_DB=~/.oboyu/work.db
export PERSONAL_DB=~/.oboyu/personal.db

# Work searches
alias work-search='oboyu query --db-path $WORK_DB'
alias work-index='oboyu index --db-path $WORK_DB'

# Personal searches
alias personal-search='oboyu query --db-path $PERSONAL_DB'
alias personal-index='oboyu index --db-path $PERSONAL_DB'
```

## Using Oboyu in Scripts and Automation

### Basic Script Template

```bash
#!/bin/bash
set -euo pipefail

# Oboyu automation script template
OBOYU_DB="${OBOYU_DB:-$HOME/.oboyu/index.db}"
LOG_FILE="${LOG_FILE:-/tmp/oboyu-automation.log}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Function to safely index with retries
index_with_retry() {
    local path=$1
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        if oboyu index "$path" --quiet-progress --db-path "$OBOYU_DB"; then
            log "Successfully indexed: $path"
            return 0
        else
            retry=$((retry + 1))
            log "Indexing failed (attempt $retry/$max_retries): $path"
            sleep 5
        fi
    done
    
    log "ERROR: Failed to index after $max_retries attempts: $path"
    return 1
}

# Function to search and process results
search_and_process() {
    local query=$1
    local results_file="/tmp/oboyu-results-$$.json"
    
    if oboyu query "$query" --format json --top-k 20 > "$results_file"; then
        # Process results with jq
        jq -r '.results[] | "\(.score)\t\(.file_path)"' "$results_file" | \
        while IFS=$'\t' read -r score path; do
            log "Found: $path (score: $score)"
            # Add your processing logic here
        done
        rm -f "$results_file"
    else
        log "ERROR: Search failed for query: $query"
        return 1
    fi
}

# Main execution
main() {
    log "Starting Oboyu automation"
    
    # Example: Index multiple directories
    for dir in "$@"; do
        index_with_retry "$dir"
    done
    
    # Example: Run searches
    search_and_process "TODO"
    search_and_process "FIXME"
    
    log "Automation complete"
}

# Run main function with all arguments
main "$@"
```

### CI/CD Integration

```yaml
# GitHub Actions example
name: Code Search Index
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

jobs:
  index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install Oboyu
        run: |
          pip install oboyu
          oboyu version
      
      - name: Index codebase
        run: |
          oboyu index . \
            --include-patterns "*.py,*.js,*.md" \
            --exclude-patterns "node_modules/*,venv/*" \
            --db-path ./search-index.db
      
      - name: Run quality checks
        run: |
          # Search for TODOs
          oboyu query "TODO" --format json > todos.json
          
          # Check for security issues
          oboyu query "password|secret|api_key" --format json > security-check.json
          
          # Generate report
          python scripts/analyze_search_results.py
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: search-results
          path: |
            search-index.db
            todos.json
            security-check.json
```

### Docker Integration

```dockerfile
# Dockerfile for Oboyu service
FROM python:3.11-slim

WORKDIR /app

# Install Oboyu
RUN pip install oboyu

# Create volume for persistent database
VOLUME ["/data"]

# Copy indexing script
COPY index.sh /app/

# Default command
CMD ["oboyu", "mcp", "--db-path", "/data/index.db"]
```

```bash
# Docker compose example
# docker-compose.yml
version: '3.8'

services:
  oboyu:
    build: .
    volumes:
      - oboyu-data:/data
      - ./documents:/documents:ro
    environment:
      - OBOYU_DB_PATH=/data/index.db
    command: >
      sh -c "oboyu index /documents --quiet-progress &&
             oboyu mcp --db-path /data/index.db"

volumes:
  oboyu-data:
```

### Monitoring Script

```python
#!/usr/bin/env python3
"""Monitor Oboyu index health and performance"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_oboyu_command(args):
    """Run an oboyu command and return output"""
    result = subprocess.run(
        ["oboyu"] + args,
        capture_output=True,
        text=True
    )
    return result.stdout, result.returncode

def check_index_health(db_path):
    """Check index health and statistics"""
    # Get index status
    output, code = run_oboyu_command([
        "index", "manage", "status", ".", 
        "--db-path", db_path,
        "--detailed"
    ])
    
    if code != 0:
        print(f"Error checking index status: {code}")
        return False
    
    # Run test queries
    test_queries = ["test", "function", "class"]
    for query in test_queries:
        output, code = run_oboyu_command([
            "query", query,
            "--db-path", db_path,
            "--format", "json",
            "--top-k", "1"
        ])
        
        if code != 0:
            print(f"Query '{query}' failed with code: {code}")
            return False
            
        try:
            results = json.loads(output)
            print(f"Query '{query}': {results.get('total', 0)} results")
        except json.JSONDecodeError:
            print(f"Invalid JSON response for query '{query}'")
            return False
    
    return True

def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "~/.oboyu/index.db"
    db_path = Path(db_path).expanduser()
    
    print(f"Monitoring Oboyu index: {db_path}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("-" * 50)
    
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)
    
    # Check database size
    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"Database size: {size_mb:.2f} MB")
    
    # Run health checks
    if check_index_health(str(db_path)):
        print("\nHealth check: PASSED")
    else:
        print("\nHealth check: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
```