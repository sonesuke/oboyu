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
| `--config FILE` | Specify config file | `--config ~/.oboyu/work.yaml` |
| `--db-path PATH` | Path to database file | `--db-path ~/indexes/personal.db` |
| `--verbose` | Enable verbose output | `--verbose` |
| `--help` | Show help | `--help` |

## Commands Overview

Available commands:

- `oboyu index` - Index documents for search
- `oboyu query` - Search indexed documents  
- `oboyu manage` - Manage the index database
- `oboyu health` - Health monitoring and diagnostics
- `oboyu mcp` - Run MCP server for AI assistant integration
- `oboyu clear` - Clear all data from the index database
- `oboyu version` - Display version information

## Index Commands

### `oboyu index`

Index documents for search.

#### Basic Usage
```bash
oboyu index DIRECTORIES... [OPTIONS]
```

#### Examples
```bash
# Index a directory
oboyu index ~/Documents

# Index multiple directories
oboyu index ~/docs ~/notes ~/projects

# Index with custom database path
oboyu index ~/work --db-path ~/indexes/work-docs.db

# Index with file patterns
oboyu index ~/code --include-patterns "*.py" --include-patterns "*.md"

# Index with exclusions
oboyu index ~/all --exclude-patterns "*.tmp" --exclude-patterns "*/cache/*"

# Force reindexing
oboyu index ~/docs --force

# Quiet progress output
oboyu index ~/docs --quiet-progress
```

#### Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--config FILE` | Configuration file path | None | `--config ~/.oboyu/config.yaml` |
| `--recursive / --no-recursive` | Recursive directory traversal | `--recursive` | `--no-recursive` |
| `--include-patterns TEXT` | Include file patterns | None | `--include-patterns "*.md"` |
| `--exclude-patterns TEXT` | Exclude file patterns | None | `--exclude-patterns "*.tmp"` |
| `--max-depth INTEGER` | Maximum directory depth | None | `--max-depth 3` |
| `--force / --no-force` | Force reindexing | `--no-force` | `--force` |
| `--encoding-detection / --no-encoding-detection` | Auto-detect file encoding | `--encoding-detection` | `--no-encoding-detection` |
| `--chunk-size INTEGER` | Chunk size in tokens | Default from config | `--chunk-size 512` |
| `--chunk-overlap INTEGER` | Chunk overlap size | Default from config | `--chunk-overlap 128` |
| `--embedding-model TEXT` | Embedding model to use | Default from config | `--embedding-model cl-nagoya/ruri-v3-30m` |
| `--db-path PATH` | Database file path | Default location | `--db-path ~/indexes/personal.db` |
| `--change-detection TEXT` | Change detection strategy | `smart` | `--change-detection hash` |
| `--cleanup-deleted / --no-cleanup-deleted` | Remove deleted files from index | Default from config | `--cleanup-deleted` |
| `--verify-integrity` | Verify file integrity using hashes | False | `--verify-integrity` |
| `--quiet-progress` | Minimal progress output | False | `--quiet-progress` |

#### Change Detection Strategies

| Strategy | Description | Performance | Accuracy |
|----------|-------------|-------------|----------|
| `timestamp` | Use file modification time | Fast | Good |
| `hash` | Use content hash | Slower | Excellent |
| `smart` | Hybrid approach | Balanced | Very Good |

## Query Commands

### `oboyu query`

Search through indexed documents.

#### Basic Usage
```bash
oboyu query [OPTIONS]
```

#### Examples
```bash
# Basic search (will prompt for query)
oboyu query --query "machine learning"

# Search with specific mode
oboyu query --query "python tutorial" --mode vector

# Search specific database
oboyu query --query "meeting notes" --db-path ~/indexes/work.db

# Different search modes
oboyu query --query "semantic search" --mode vector
oboyu query --query "exact phrase" --mode bm25
oboyu query --query "hybrid search" --mode hybrid

# Interactive mode
oboyu query --interactive

# JSON output
oboyu query --query "test" --format json

# More results
oboyu query --query "test" --top-k 20

# With reranking
oboyu query --query "test" --rerank

# Show explanations
oboyu query --query "test" --explain
```

#### Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--query TEXT` | Search query text | None (prompts if missing) | `--query "machine learning"` |
| `--mode TEXT` | Search mode | `hybrid` | `--mode vector` |
| `--top-k INTEGER` | Maximum number of results | Default from config | `--top-k 20` |
| `--explain / --no-explain` | Show match explanations | `--no-explain` | `--explain` |
| `--format TEXT` | Output format | `text` | `--format json` |
| `--rrf-k INTEGER` | RRF parameter for hybrid search | Default from config | `--rrf-k 10` |
| `--db-path PATH` | Database file path | Default location | `--db-path ~/indexes/work.db` |
| `--rerank / --no-rerank` | Enable result reranking | Default from config | `--rerank` |
| `--interactive / --no-interactive` | Interactive search mode | `--no-interactive` | `--interactive` |

#### Search Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `bm25` | Keyword-based search | Exact terms, technical content |
| `vector` | Semantic vector search | Concepts, natural language |
| `hybrid` | Combined BM25 + vector | General use, best quality |

#### Output Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `text` | Human-readable text | Interactive use |
| `json` | JSON format | Scripting, automation |

## Management Commands

### `oboyu manage`

Manage the index database.

#### Subcommands

##### `oboyu manage clear`
Clear all data from the index database:
```bash
# Clear default database
oboyu manage clear

# Clear specific database
oboyu manage clear --db-path ~/indexes/old.db
```

##### `oboyu manage status`
Show indexing status for directories:
```bash
# Show status for default database
oboyu manage status

# Show status for specific database
oboyu manage status --db-path ~/indexes/work.db
```

##### `oboyu manage diff`
Show what would be updated if indexing were run:
```bash
# Show diff for default database
oboyu manage diff

# Show diff for specific database
oboyu manage diff --db-path ~/indexes/work.db
```

## Health Commands

### `oboyu health`

Health monitoring and diagnostics.

#### Subcommands

##### `oboyu health check`
Quick health check of the system:
```bash
oboyu health check
```

##### `oboyu health events`
Show recent events for debugging:
```bash
oboyu health events
```

##### `oboyu health timeline`
Show timeline of events for a specific operation:
```bash
oboyu health timeline
```

##### `oboyu health operations`
Show recent operations for debugging:
```bash
oboyu health operations
```

## MCP Commands

### `oboyu mcp`

Run an MCP (Model Context Protocol) server for AI assistant integration.

#### Basic Usage
```bash
oboyu mcp [OPTIONS]
```

#### Examples
```bash
# Basic MCP server with stdio transport
oboyu mcp

# MCP server with specific database
oboyu mcp --db-path ~/indexes/work.db

# MCP server with HTTP transport
oboyu mcp --transport streamable-http --port 8080

# Debug mode
oboyu mcp --debug --verbose
```

#### Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--db-path PATH` | Database file path | Default location | `--db-path ~/indexes/work.db` |
| `--verbose / --no-verbose` | Verbose logging | `--no-verbose` | `--verbose` |
| `--debug / --no-debug` | Debug mode | `--no-debug` | `--debug` |
| `--transport TEXT` | Transport mechanism | `stdio` | `--transport streamable-http` |
| `--port INTEGER` | Port number (for HTTP transports) | None | `--port 8080` |

#### Transport Options

| Transport | Description | Use Case |
|-----------|-------------|----------|
| `stdio` | Standard input/output | Claude Desktop integration |
| `streamable-http` | HTTP with streaming | Web-based integrations |
| `sse` | Server-sent events | Real-time web apps |

## Utility Commands

### `oboyu clear`

Clear all data from the index database while preserving structure.

```bash
# Clear default database
oboyu clear

# Clear specific database
oboyu clear --db-path ~/indexes/old.db
```

### `oboyu version`

Display version information.

```bash
oboyu version
```

## Environment Variables

Oboyu respects these environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `OBOYU_DB_PATH` | Default database path | `~/indexes/default.db` |

## Exit Codes

Oboyu uses standard exit codes:

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |

## Examples by Use Case

### Personal Knowledge Base
```bash
# Setup
oboyu index ~/Notes --db-path ~/indexes/personal.db --chunk-size 1536

# Daily usage
oboyu query --query "project ideas" --db-path ~/indexes/personal.db
oboyu query --interactive --db-path ~/indexes/personal.db
```

### Software Development
```bash
# Setup
oboyu index ~/code --include-patterns "*.py" --include-patterns "*.md" --db-path ~/indexes/code.db

# Usage
oboyu query --query "authentication implementation" --db-path ~/indexes/code.db --mode hybrid
oboyu query --query "API documentation" --db-path ~/indexes/code.db --rerank
```

### Research Papers
```bash
# Setup
oboyu index ~/papers --include-patterns "*.pdf" --db-path ~/indexes/papers.db --chunk-size 2048

# Usage
oboyu query --query "machine learning optimization" --db-path ~/indexes/papers.db --mode vector
oboyu query --interactive --db-path ~/indexes/papers.db
```

## Tips and Best Practices

### Command Aliases
```bash
# Add to ~/.bashrc or ~/.zshrc
alias oq='oboyu query --interactive'
alias oi='oboyu index'
alias om='oboyu manage'
```

### Scripting with Oboyu
```bash
# Check exit codes
if oboyu query --query "test" > /dev/null 2>&1; then
    echo "Found results"
fi

# Parse JSON output
oboyu query --query "search" --format json | jq '.[].document.path'

# Automated indexing
oboyu index ~/docs --force --quiet-progress
```

## Getting Help

```bash
# General help
oboyu --help

# Command-specific help
oboyu index --help
oboyu query --help
oboyu manage --help
oboyu health --help
oboyu mcp --help
```