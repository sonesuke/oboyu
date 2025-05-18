# Oboyu CLI Commands

This document describes the available command-line interface (CLI) commands for Oboyu.

## Global Options

The following options are available for all commands:

- `--config`, `-c`: Path to configuration file
- `--db-path`: Path to database file
- `--verbose`, `-v`: Enable verbose output

## Main Commands

### `oboyu version`

Display the current version of Oboyu.

```bash
oboyu version
```

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
```

Options:
- `--recursive/--no-recursive`, `-r/-nr`: Process directories recursively (default: recursive)
- `--include-patterns`, `-i`: File patterns to include (e.g., `*.txt,*.md`)
- `--exclude-patterns`, `-e`: File patterns to exclude (e.g., `*/node_modules/*`)
- `--max-depth`, `-d`: Maximum recursion depth
- `--force`, `-f`: Force re-index of all documents
- `--encoding-detection/--no-encoding-detection`: Enable/disable Japanese encoding detection
- `--japanese-encodings`, `-j`: Japanese encodings to detect (e.g., `utf-8,shift-jis,euc-jp`)
- `--chunk-size`: Chunk size in characters
- `--chunk-overlap`: Chunk overlap in characters
- `--embedding-model`: Embedding model to use
- `--db-path`: Path to database file

### `oboyu index manage clear`

Clear all data from the index database (subcommand of index manage).

```bash
oboyu index manage clear
```

Options:
- `--force`, `-f`: Force clearing without confirmation
- `--db-path`: Path to database file to clear

### `oboyu query`

Search indexed documents.

```bash
# Basic query
oboyu query "search term"

# Specify search mode
oboyu query "search term" --mode vector

# Get more results
oboyu query "search term" --top-k 10
```

Options:
- `--mode`, `-m`: Search mode (vector, bm25, hybrid)
- `--top-k`, `-k`: Number of results to return
- `--language`, `-l`: Filter by language
- `--explain`, `-e`: Show detailed explanation of results
- `--json`: Output results in JSON format
- `--db-path`: Path to database file