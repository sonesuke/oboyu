# Oboyu CLI Commands

This document describes the available command-line interface (CLI) commands for Oboyu.

**Main CLI Description:** A Japanese-enhanced semantic search system with integrated GraphRAG functionality.

## Quick Command Reference

| Command | Description | Common Options |
|---------|-------------|----------------|
| `oboyu version` | Display version information | - |
| `oboyu mcp` | Run MCP server for AI assistant integration | `--transport`, `--port`, `--verbose`, `--debug` |
| `oboyu clear` | Clear all data from index database while preserving schema | `--force` |
| `oboyu status [path]` | Show unified index and knowledge graph status | `--detailed`, `--kg` |
| `oboyu index <path>` | Index documents for search | `--force`, `--chunk-size`, `--include-patterns` |
| `oboyu search <query>` | Search documents with GraphRAG enhancement | `--mode`, `--top-k`, `--no-graph`, `--no-rerank` |
| `oboyu enrich <csv> <schema>` | Enrich CSV data using semantic search and GraphRAG | `--batch-size`, `--confidence`, `--output` |
| `oboyu build-kg` | Build knowledge graph from indexed documents | `--full`, `--batch-size`, `--limit` |
| `oboyu deduplicate` | Deduplicate entities in knowledge graph | `--type`, `--similarity`, `--verification` |

### Common Workflows

```bash
# Quick start: Index a codebase and search
oboyu index ~/projects/myapp --include-patterns "*.py,*.js"
oboyu search "database connection" --top-k 10

# Incremental updates with cleanup
oboyu index ~/documents --cleanup-deleted
oboyu status ~/documents --detailed

# Enrich CSV data with knowledge base information
oboyu enrich companies.csv enrichment-schema.json --output enriched.csv

```

## Global Options

The following options are available for most commands:

- `--config`, `-c`: Path to configuration file (see [Configuration Guide](../reference/configuration.md))
- `--db-path`: Path to database file (env var: OBOYU_DB_PATH)
- `--verbose`, `-v`: Enable verbose output

**Note:** If you're upgrading from an older version, see the [Configuration Migration Guide](immutable-configuration-migration.md) for information about the new immutable configuration system.

## Main Commands

### `oboyu version`

Display the current version of Oboyu.

```bash
oboyu version
```

### `oboyu mcp`

Run MCP server for AI assistant integration.

**Usage:** `oboyu mcp [OPTIONS]`

```bash
# Start MCP server with stdio transport (default)
oboyu mcp

# Start with custom database path and verbose output
oboyu mcp --db-path /path/to/custom.db --verbose

# Start with debug mode for development
oboyu mcp --verbose --debug

# Start with specific transport and port
oboyu mcp --transport stdio --port 8000
```

**Options:**
- `--db-path PATH`: Path to database file
- `--verbose/--no-verbose`: Enable verbose output (default: no-verbose)
- `--debug/--no-debug`: Enable debug mode (default: no-debug)
- `--transport TEXT`: Transport mechanism (default: stdio)
- `--port INTEGER`: Port number

### `oboyu clear`

Clear all data from the index database while preserving the database schema and structure.

**Usage:** `oboyu clear [OPTIONS]`

This command clears the index database, removing all indexed file metadata and content. Unlike the clear-db command which deletes the database files entirely, this command preserves the database structure and only removes the data within it.

```bash
# Clear with confirmation prompt
oboyu clear

# Force clear without confirmation
oboyu clear --force

# Clear a specific database
oboyu clear --db-path custom.db
```

**Options:**
- `--config`, `-c`: Path to configuration file
- `--db-path`, `-d FILE`: Path to the database file (default: from config)
- `--force`, `-f`: Skip confirmation prompt

### `oboyu index`

Index documents for search.

**Usage:** `oboyu index [OPTIONS] DIRECTORIES...`

**Arguments:**
- `DIRECTORIES...`: Directories to index (required)

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

**Options:**
- `--config`, `-c`: Path to configuration file
- `--recursive/--no-recursive`: Process directories recursively (default: recursive)
- `--include-patterns TEXT`: File patterns to include
- `--exclude-patterns TEXT`: File patterns to exclude
- `--max-depth INTEGER`: Maximum recursion depth
- `--force/--no-force`: Force re-index of all documents (default: no-force)
- `--encoding-detection/--no-encoding-detection`: Enable/disable automatic encoding detection (default: encoding-detection)
- `--chunk-size INTEGER`: Chunk size in characters
- `--chunk-overlap INTEGER`: Chunk overlap in characters
- `--embedding-model TEXT`: Embedding model to use
- `--db-path PATH`: Path to database file
- `--change-detection TEXT`: Strategy for detecting changes: timestamp, hash, or smart (default: smart)
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


### `oboyu status`

Show unified index and knowledge graph status.

**Usage:** `oboyu status [OPTIONS] [DIRECTORIES]...`

This command shows indexing status for directories (if provided) and overall statistics including knowledge graph information.

**Arguments:**
- `[DIRECTORIES]...`: Directories to check status for (optional - shows overall stats if not provided)

```bash
# Show overall stats
oboyu status

# Show status for specific directory
oboyu status ~/documents

# Show detailed file-by-file status
oboyu status ~/docs --detailed

# Show only index stats, skip KG
oboyu status --no-kg

# Check status with custom database
oboyu status /path/to/docs --db-path custom.db
```

**Options:**
- `--config`, `-c`: Path to configuration file
- `--db-path`, `-p FILE`: Path to the database file (default: from config)
- `--detailed`, `-d`: Show detailed file-by-file status
- `--kg`: Show knowledge graph statistics (default: True)

## Data Enrichment Commands

### `oboyu enrich`

Enrich CSV data using semantic search and GraphRAG capabilities.

**Usage:** `oboyu enrich [OPTIONS] CSV_FILE SCHEMA_FILE`

This command takes a CSV file and a JSON schema configuration to automatically populate additional columns with relevant information from the indexed knowledge base.

The enrichment process leverages Oboyu's hybrid search (vector + BM25) and GraphRAG functionality to find relevant information and extract specific data points.

**Arguments:**
- `CSV_FILE`: Input CSV file to enrich (required)
- `SCHEMA_FILE`: JSON schema file defining enrichment configuration (required)

```bash
# Basic enrichment
oboyu enrich companies.csv enrichment-schema.json

# Specify output file
oboyu enrich data.csv config.json --output enriched-data.csv

# Adjust batch processing settings
oboyu enrich large-dataset.csv schema.json --batch-size 5

# Fine-tune search parameters
oboyu enrich data.csv config.json \
  --confidence 0.7 \
  --max-results 3 \
  --batch-size 10

# Disable GraphRAG for faster processing
oboyu enrich data.csv config.json --no-graph

# Use custom database
oboyu enrich data.csv config.json --db-path custom.db
```

**Options:**
- `--output`, `-o PATH`: Output CSV file path
- `--batch-size INTEGER`: Processing batch size (default: 10)
- `--max-results INTEGER`: Maximum search results per query (default: 5)
- `--confidence FLOAT`: Minimum confidence threshold (default: 0.5)
- `--no-graph`: Disable GraphRAG enhancement
- `--db-path`, `-p PATH`: Path to database file

#### Schema Configuration

The enrichment schema defines how to enrich your CSV data. Create a JSON file with the following structure:

```json
{
  "input_schema": {
    "columns": {
      "company_name": {
        "type": "string",
        "description": "会社名",
        "required": true
      }
    },
    "primary_keys": ["company_name"]
  },
  "enrichment_schema": {
    "columns": {
      "description": {
        "type": "string",
        "description": "会社概要",
        "source_strategy": "search_content",
        "query_template": "{company_name} 概要 事業内容",
        "extraction_method": "summarize"
      },
      "employees": {
        "type": "integer",
        "description": "従業員数",
        "source_strategy": "search_content",
        "query_template": "{company_name} 従業員数",
        "extraction_method": "pattern_match",
        "extraction_pattern": "\\d+(?:人|名)"
      },
      "founded_year": {
        "type": "integer",
        "description": "設立年",
        "source_strategy": "graph_relations",
        "query_template": "{company_name} 設立",
        "relation_types": ["FOUNDED_IN"],
        "target_entity_types": ["DATE", "YEAR"]
      }
    }
  },
  "search_config": {
    "search_mode": "hybrid",
    "use_graphrag": true,
    "rerank": true,
    "top_k": 5,
    "similarity_threshold": 0.5
  }
}
```

#### Extraction Strategies

**`search_content`**: Semantic search with content extraction
- `extraction_method`: `first_result`, `first_sentence`, `summarize`, `pattern_match`
- `extraction_pattern`: Regex pattern (for pattern_match method)

**`entity_extraction`**: Knowledge graph entity extraction
- `entity_types`: Filter by entity types (e.g., `["PERSON", "ORGANIZATION"]`)
- `similarity_threshold`: Minimum similarity score

**`graph_relations`**: Graph relationship traversal
- `relation_types`: Types of relationships to follow
- `target_entity_types`: Types of target entities

#### Enrichment Examples

**Company Data Enrichment:**
```bash
# Create sample CSV
echo "company_name,industry
株式会社ソフトバンク,通信
トヨタ自動車株式会社,自動車" > companies.csv

# Run enrichment
oboyu enrich companies.csv schema.json --batch-size 2
```

**Performance Tuning:**
```bash
# Fast processing for large datasets
oboyu enrich large-data.csv schema.json \
  --batch-size 20 \
  --confidence 0.3 \
  --no-graph

# High accuracy with smaller batches
oboyu enrich data.csv schema.json \
  --batch-size 5 \
  --confidence 0.8 \
  --max-results 10
```

**Error Recovery:**
```bash
# Resume failed enrichment with lower confidence
oboyu enrich failed-data.csv schema.json \
  --confidence 0.3 \
  --batch-size 2
```

For detailed usage examples and advanced configuration, see the [CSV Enrichment Use Case](../use-cases/csv-enrichment.md) documentation.

## Knowledge Graph Commands

Oboyu provides powerful knowledge graph operations for enhanced search capabilities through GraphRAG (Graph Retrieval-Augmented Generation).

### `oboyu build-kg`

Build knowledge graph from indexed documents.

**Usage:** `oboyu build-kg [OPTIONS]`

```bash
# Build knowledge graph incrementally
oboyu build-kg

# Force rebuild entire knowledge graph
oboyu build-kg --full

# Build with custom batch size
oboyu build-kg --batch-size 100

# Build limited number of chunks
oboyu build-kg --limit 1000
```

**Options:**
- `--full`: Rebuild entire knowledge graph
- `--batch-size INTEGER`: Processing batch size
- `--limit INTEGER`: Limit number of chunks to process

### `oboyu deduplicate`

Deduplicate entities in knowledge graph.

**Usage:** `oboyu deduplicate [OPTIONS]`

```bash
# Deduplicate all entities
oboyu deduplicate

# Deduplicate specific entity type
oboyu deduplicate --type "PERSON"

# Custom similarity thresholds
oboyu deduplicate --similarity 0.9 --verification 0.85

# Custom batch size for large datasets
oboyu deduplicate --batch-size 200

# Show potential duplicates for analysis
oboyu deduplicate --show-duplicates --entity-name "specific entity" --limit 5
```

**Options:**
- `--type TEXT`: Entity type to deduplicate (all if not specified)
- `--similarity FLOAT`: Vector similarity threshold (default: 0.85)
- `--verification FLOAT`: LLM verification threshold (default: 0.8)
- `--batch-size INTEGER`: Processing batch size (default: 100)
- `--show-duplicates`: Show potential duplicate entities for analysis
- `--entity-name TEXT`: Specific entity name to find duplicates for (when --show-duplicates is used)
- `--limit INTEGER`: Maximum number of duplicate results to show (default: 10)

## Legacy Commands

**Note:** Some legacy `manage` commands may still be available, but the top-level equivalents are preferred:
- Use `oboyu clear` instead of `oboyu manage clear`
- Use `oboyu status` instead of `oboyu manage status`

## Search Examples

**GraphRAG-enhanced search (default behavior):**
```bash
# Search with knowledge graph enhancement
oboyu search "machine learning algorithms"

# Search with query expansion details
oboyu search "neural networks" --expand

# Search with detailed processing explanation
oboyu search "database optimization" --explain
```

**Traditional search (GraphRAG disabled):**
```bash
# Find function definitions with specific names
oboyu search "def process_data" --no-graph --mode bm25 --top-k 20

# Search for class implementations
oboyu search "class DatabaseConnection" --no-graph --explain
```

**Multi-language search with reranking:**
```bash
# Search across English and Japanese content
oboyu search "machine learning 機械学習" --top-k 10
```

**Export results for processing:**
```bash
# Get JSON output for scripting
oboyu search "error handling" --format json | jq '.results[].file_path'

# Save results to file
oboyu search "TODO" --top-k 50 --format json > todos.json
```

**Fine-tuned hybrid search:**
```bash
# Use semantic search for concept-based queries
oboyu search "authentication flow" --mode vector

# Use hybrid search for balanced semantic and keyword matching
oboyu search "database optimization techniques" --mode hybrid
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
oboyu search "important document" --format json > results.json
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
oboyu search --mode <TAB>
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
oboyu search "test" --verbose

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
oboyu status ~/docs --detailed

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
alias todos='oboyu search "TODO|FIXME" --no-graph --mode bm25 --top-k 50'
alias errors='oboyu search "error|exception|traceback" --rerank'
alias recent='oboyu search "recent changes" --mode hybrid --top-k 10'

# Project-specific search
function search_project() {
    oboyu search "$1" --db-path ~/projects/${2:-current}.db
}
```

#### 4. Integration with Development Tools

```bash
# Find and open files in editor
oboyu search "$1" --format json | \
    jq -r '.results[0].file_path' | \
    xargs code

# Search and preview with bat
oboyu search "$1" --format json | \
    jq -r '.results[].file_path' | \
    xargs bat --paging=never
```

#### 5. Performance Optimization

```bash
# Benchmark different search modes
time oboyu search "database connection" --no-graph --mode vector --no-rerank
time oboyu search "database connection" --no-graph --mode bm25 --no-rerank
time oboyu search "database connection" --mode hybrid --rerank

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
alias work-search='oboyu search --db-path $WORK_DB'
alias work-index='oboyu index --db-path $WORK_DB'

# Personal searches
alias personal-search='oboyu search --db-path $PERSONAL_DB'
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
    
    if oboyu search "$query" --format json --top-k 20 > "$results_file"; then
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
          oboyu search "TODO" --format json > todos.json
          
          # Check for security issues
          oboyu search "password|secret|api_key" --format json > security-check.json
          
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
        "status", ".", 
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
            "search", query,
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