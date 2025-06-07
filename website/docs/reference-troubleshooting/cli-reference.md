---
id: cli-reference
title: Complete CLI Reference
sidebar_position: 2
---

# Complete CLI Reference

Comprehensive reference for all Oboyu command-line interface commands, options, and usage patterns.

## Global Options

These options are available for all commands:

```bash
oboyu [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]
```

| Option | Description | Example |
|--------|-------------|---------|
| `--config PATH` | Specify config file | `--config ~/.oboyu/work.yaml` |
| `--log-level LEVEL` | Set logging level | `--log-level DEBUG` |
| `--output FORMAT` | Output format | `--output json` |
| `--quiet` | Suppress output | `--quiet` |
| `--verbose` | Verbose output | `--verbose` |
| `--help` | Show help | `--help` |
| `--version` | Show version | `--version` |

### Output Formats

- `text` (default): Human-readable text
- `json`: JSON format
- `yaml`: YAML format  
- `csv`: Comma-separated values
- `table`: Formatted table

## Index Commands

### `oboyu index`

Create and manage document indices.

#### Basic Usage
```bash
oboyu index PATH [OPTIONS]
oboyu index SUBCOMMAND [OPTIONS]
```

#### Index Creation
```bash
# Index a directory
oboyu index ~/Documents

# Index multiple paths
oboyu index ~/docs ~/notes ~/projects

# Index with custom name
oboyu index ~/work --name work-docs

# Index specific file types
oboyu index ~/code --include "*.py" --include "*.md"

# Index with exclusions
oboyu index ~/all --exclude "*.tmp" --exclude "*/cache/*"
```

#### Options for Index Creation

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--name NAME` | Index name | `default` | `--name personal` |
| `--include PATTERN` | Include file pattern | `*` | `--include "*.md"` |
| `--exclude PATTERN` | Exclude file pattern | None | `--exclude "*.tmp"` |
| `--chunk-size SIZE` | Chunk size in tokens | `1024` | `--chunk-size 512` |
| `--overlap SIZE` | Chunk overlap | `256` | `--overlap 128` |
| `--force` | Force reindex | False | `--force` |
| `--update` | Update existing index | False | `--update` |
| `--threads N` | Number of threads | Auto | `--threads 4` |

#### Index Management Subcommands

##### `oboyu index list`
List all indices:
```bash
# List all indices
oboyu index list

# List with details
oboyu index list --detailed

# List specific format
oboyu index list --format json
```

##### `oboyu index info`
Show index information:
```bash
# Info about specific index
oboyu index info --name personal

# Detailed information
oboyu index info --name personal --detailed

# Show configuration
oboyu index info --name personal --show-config
```

##### `oboyu index update`
Update existing indices:
```bash
# Update specific index
oboyu index update --name personal

# Update all indices
oboyu index update --all

# Update with time filter
oboyu index update --name personal --modified-after "1 week ago"
```

##### `oboyu index delete`
Delete indices:
```bash
# Delete specific index
oboyu index delete --name old-index

# Delete with confirmation
oboyu index delete --name old-index --force

# Delete all indices
oboyu index delete --all --force
```

##### `oboyu index optimize`
Optimize index performance:
```bash
# Optimize specific index
oboyu index optimize --name personal

# Optimize all indices
oboyu index optimize --all

# Optimize with vacuum
oboyu index optimize --name personal --vacuum
```

##### `oboyu index verify`
Verify index integrity:
```bash
# Verify specific index
oboyu index verify --name personal

# Verify and repair
oboyu index verify --name personal --repair

# Verify all indices
oboyu index verify --all
```

## Query Commands

### `oboyu query`

Search through indexed documents.

#### Basic Usage
```bash
oboyu query "search terms" [OPTIONS]
```

#### Search Examples
```bash
# Basic search
oboyu query "machine learning"

# Search specific index
oboyu query "python tutorial" --index programming

# Search with filters
oboyu query "meeting" --file-type md --days 7

# Different search modes
oboyu query "semantic search" --mode vector
oboyu query "exact phrase" --mode bm25
oboyu query "hybrid search" --mode hybrid
```

#### Query Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--mode MODE` | Search mode | `hybrid` | `--mode vector` |
| `--index NAME` | Target index | `default` | `--index work` |
| `--limit N` | Max results | `10` | `--limit 20` |
| `--file-type EXT` | Filter by file type | None | `--file-type md,txt` |
| `--days N` | Filter by days | None | `--days 30` |
| `--from DATE` | Start date filter | None | `--from 2024-01-01` |
| `--to DATE` | End date filter | None | `--to 2024-12-31` |
| `--path PATTERN` | Path filter | None | `--path "**/docs/**"` |
| `--language LANG` | Language filter | None | `--language ja` |
| `--sort FIELD` | Sort results | `relevance` | `--sort date` |

#### Search Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `bm25` | Keyword search | Exact terms, technical content |
| `vector` | Semantic search | Concepts, natural language |
| `hybrid` | Combined approach | General use, best quality |

#### Output Options

| Option | Description | Example |
|--------|-------------|---------|
| `--format FORMAT` | Output format | `--format json` |
| `--show-scores` | Show relevance scores | `--show-scores` |
| `--show-paths` | Show file paths only | `--show-paths` |
| `--context N` | Context characters | `--context 500` |
| `--no-highlight` | Disable highlighting | `--no-highlight` |

#### Interactive Mode
```bash
# Start interactive session
oboyu query --interactive

# Interactive with specific index
oboyu query --interactive --index work
```

#### Query History
```bash
# Show query history
oboyu query history

# History for specific period
oboyu query history --days 7

# History statistics
oboyu query history --stats
```

## Configuration Commands

### `oboyu config`

Manage Oboyu configuration.

#### Basic Usage
```bash
oboyu config SUBCOMMAND [OPTIONS]
```

#### Configuration Subcommands

##### `oboyu config show`
Display configuration:
```bash
# Show all configuration
oboyu config show

# Show specific section
oboyu config show indexer

# Show with sources
oboyu config show --sources

# Show effective config
oboyu config show --effective
```

##### `oboyu config set`
Set configuration values:
```bash
# Set simple value
oboyu config set indexer.chunk_size 2048

# Set list value
oboyu config set crawler.include_patterns "*.md,*.txt"

# Set boolean value
oboyu config set indexer.use_reranker true
```

##### `oboyu config get`
Get configuration values:
```bash
# Get specific value
oboyu config get indexer.chunk_size

# Get section
oboyu config get indexer

# Get with type info
oboyu config get indexer.chunk_size --type
```

##### `oboyu config unset`
Remove configuration values:
```bash
# Unset specific value
oboyu config unset indexer.chunk_overlap

# Unset section
oboyu config unset crawler.exclude_patterns
```

##### `oboyu config reset`
Reset configuration:
```bash
# Reset all to defaults
oboyu config reset

# Reset specific section
oboyu config reset indexer

# Reset with confirmation
oboyu config reset --force
```

##### `oboyu config validate`
Validate configuration:
```bash
# Validate current config
oboyu config validate

# Validate specific file
oboyu config validate ~/.oboyu/config.yaml

# Validate and show errors
oboyu config validate --verbose
```

##### `oboyu config init`
Initialize configuration:
```bash
# Create default config
oboyu config init

# Create with template
oboyu config init --template research

# Create in specific location
oboyu config init --path ~/.oboyu/custom.yaml
```

## MCP Commands

### `oboyu mcp`

Model Context Protocol server for Claude integration.

#### Basic Usage
```bash
oboyu mcp SUBCOMMAND [OPTIONS]
```

#### MCP Subcommands

##### `oboyu mcp serve`
Start MCP server:
```bash
# Basic server
oboyu mcp serve

# Server with specific index
oboyu mcp serve --index ~/.oboyu/personal.db

# Server with configuration
oboyu mcp serve --config ~/.oboyu/mcp.yaml

# Debug mode
oboyu mcp serve --debug
```

**MCP Serve Options:**

| Option | Description | Example |
|--------|-------------|---------|
| `--index PATH` | Index file path | `--index ~/.oboyu/main.db` |
| `--config PATH` | MCP config file | `--config mcp.yaml` |
| `--max-results N` | Max search results | `--max-results 20` |
| `--timeout N` | Request timeout | `--timeout 30` |
| `--debug` | Debug logging | `--debug` |

##### `oboyu mcp test`
Test MCP functionality:
```bash
# Test basic functionality
oboyu mcp test

# Test with specific query
oboyu mcp test --query "test search"

# Test connection
oboyu mcp test --connection-only
```

##### `oboyu mcp status`
Check MCP server status:
```bash
# Check if server is running
oboyu mcp status

# Detailed status
oboyu mcp status --detailed
```

## Utility Commands

### `oboyu models`

Manage embedding models.

```bash
# List available models
oboyu models list

# Download specific model
oboyu models download cl-nagoya/ruri-v3-30m

# Show model info
oboyu models info cl-nagoya/ruri-v3-30m

# Clear model cache
oboyu models clear-cache
```

### `oboyu diagnose`

System diagnostics and health checks.

```bash
# Basic diagnostics
oboyu diagnose

# Detailed system info
oboyu diagnose --detailed

# Check specific component
oboyu diagnose --component indexer

# Export diagnostics
oboyu diagnose --export diagnostics.json
```

### `oboyu benchmark`

Performance benchmarking.

```bash
# Basic benchmark
oboyu benchmark

# Benchmark specific index
oboyu benchmark --index personal

# Benchmark search performance
oboyu benchmark --search-only

# Benchmark indexing
oboyu benchmark --index-only
```

## Environment Variables

Oboyu respects these environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `OBOYU_CONFIG_PATH` | Config file path | `/path/to/config.yaml` |
| `OBOYU_LOG_LEVEL` | Logging level | `DEBUG` |
| `OBOYU_CACHE_DIR` | Cache directory | `~/.cache/oboyu` |
| `OBOYU_MEMORY_LIMIT` | Memory limit | `4GB` |
| `OBOYU_THREADS` | Thread count | `8` |

### Configuration via Environment

All config options can be set via environment variables:

```bash
# Pattern: OBOYU_<SECTION>_<OPTION>
export OBOYU_INDEXER_CHUNK_SIZE=2048
export OBOYU_QUERY_DEFAULT_MODE=hybrid
export OBOYU_CRAWLER_INCLUDE_PATTERNS="*.md,*.txt"
```

## Exit Codes

Oboyu uses standard exit codes:

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |
| `3` | File not found |
| `4` | Permission denied |
| `5` | Configuration error |

## Examples by Use Case

### Personal Knowledge Base
```bash
# Setup
oboyu index ~/Notes --name personal --chunk-size 1536
oboyu config set indexer.use_reranker true

# Daily usage
oboyu query "project ideas" --index personal
oboyu query "meeting with john" --days 30
```

### Software Development
```bash
# Setup
oboyu index ~/code --include "*.py" --include "*.md" --name code
oboyu index ~/docs --name documentation

# Usage
oboyu query "authentication implementation" --index code
oboyu query "API documentation" --index documentation --mode hybrid
```

### Research Papers
```bash
# Setup
oboyu index ~/papers --include "*.pdf" --name papers --chunk-size 2048
oboyu config set indexer.chunk_overlap 512

# Usage
oboyu query "machine learning optimization" --index papers --mode vector
oboyu query "methodology section" --index papers --context 1000
```

### Team Documentation
```bash
# Setup with filters
oboyu index ~/team-docs \
  --include "*.md" \
  --exclude "*/archive/*" \
  --name team

# Scheduled updates
oboyu index update --name team --modified-after "1 day ago"
```

## Tips and Best Practices

### Command Aliases
```bash
# Add to ~/.bashrc
alias oq='oboyu query'
alias oi='oboyu index'
alias oc='oboyu config'
```

### Scripting with Oboyu
```bash
# Check exit codes
if oboyu query "test" --quiet; then
    echo "Found results"
fi

# Parse JSON output
oboyu query "search" --format json | jq '.[] | .path'

# Use in pipelines
oboyu query "TODO" --show-paths | xargs grep -l "urgent"
```

## Getting Help

```bash
# General help
oboyu --help

# Command-specific help
oboyu index --help
oboyu query --help

# Subcommand help
oboyu config set --help
oboyu mcp serve --help

# Show examples
oboyu query --examples
oboyu index --examples
```