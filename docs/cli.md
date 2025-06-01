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
📊 Mode: hybrid | Top-K: 5 | Vector: 0.7 | BM25: 0.3 | Reranker: enabled

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
- **Parameter Tuning**: Testing different search modes and weights
- **Large Datasets**: When model loading time becomes significant
- **Iterative Refinement**: Progressively refining search queries