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

# Start interactive search session
oboyu query --interactive

# Interactive mode with specific settings
oboyu query --interactive --mode hybrid --use-reranker
```

Options:
- `--mode`, `-m`: Search mode (vector, bm25, hybrid)
- `--top-k`, `-k`: Number of results to return
- `--language`, `-l`: Filter by language
- `--explain`, `-e`: Show detailed explanation of results
- `--json`: Output results in JSON format
- `--db-path`: Path to database file
- `--interactive`, `-i`: Start interactive search session for continuous queries
- `--rerank/--no-rerank`: Enable or disable reranking of search results
- `--vector-weight`: Weight for vector scores in hybrid search (0.0-1.0)
- `--bm25-weight`: Weight for BM25 scores in hybrid search (0.0-1.0)

#### Interactive Mode

The interactive mode allows you to perform multiple searches without reloading the models and database. This is particularly useful when using the reranker, which has a significant initialization time.

**Interactive Commands:**

| Command | Description | Example |
|---------|-------------|---------|
| `<query>` | Search for documents | `machine learning algorithms` |
| `help` | Show available commands | `help` |
| `exit`, `quit`, `q` | Exit interactive mode | `exit` |
| `mode <mode>` | Change search mode | `mode vector` |
| `topk <number>` | Change number of results | `topk 10` |
| `weights <v> <b>` | Change hybrid weights | `weights 0.8 0.2` |
| `rerank on/off` | Toggle reranker | `rerank on` |
| `settings` | Show current settings | `settings` |
| `clear` | Clear screen | `clear` |
| `stats` | Show index statistics | `stats` |

**Example Interactive Session:**

```bash
$ oboyu query --interactive --use-reranker

🔍 Oboyu Interactive Search
📊 Mode: hybrid | Top-K: 5 | Vector: 0.7 | BM25: 0.3 | Reranker: enabled

✅ Ready for search!
Type your search query (or 'help' for commands, 'exit' to quit):

> machine learning algorithms
🔍 Searching...
📊 Found 3 results in 0.12 seconds

[Results displayed here...]

> mode vector
✅ Search mode changed to: vector

> topk 10
✅ Top-K changed to: 10

> exit
👋 Goodbye!
```