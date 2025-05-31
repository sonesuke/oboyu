# Oboyu (覚ゆ)

> A Japanese-enhanced semantic search system for your local documents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

## What is Oboyu?

**Oboyu** (覚ゆ - meaning "to remember" or "to memorize" in ancient Japanese) is a semantic search system that creates an intelligent index of your text-based documents. Oboyu indexes directories of text files, embeds their content using vector representations, and retrieves the most relevant documents for your queries.

With specialized support for Japanese language documents, Oboyu excels at processing content in both Japanese and English, making it an ideal solution for multilingual document collections.

The system provides both a comprehensive command-line interface for direct queries and an MCP server mode for AI assistant integration. While Oboyu works with any text-based documents, it is particularly optimized for Japanese language content, making it unique among semantic search tools.


## Key Features

- **Local Directory Processing**: Index any directory of text-based documents on your local system with incremental updates
- **Text Format Support**: Process plain text, markdown, code files, configuration files, Jupyter notebooks, and more
- **Japanese Language Excellence**: First-class support for Japanese text with specialized tokenization, encoding detection, and optimized models
- **Multiple Search Modes**: Choose between vector, BM25, or hybrid search with configurable weighting
- **Advanced Reranking**: Improve search accuracy with lightweight Ruri Cross-Encoder reranker (enabled by default)
- **Interactive Query Mode**: Persistent REPL interface with command history, auto-suggestions, and real-time configuration
- **ONNX Optimization**: 2-4x faster inference with automatic ONNX conversion for both embedding and reranker models
- **Incremental Indexing**: Smart change detection with timestamp, hash, or hybrid strategies for efficient updates
- **MCP Server Integration**: Model Context Protocol server for seamless AI assistant integration
- **Rich Command-Line Interface**: Comprehensive CLI with hierarchical progress display and extensive options
- **Privacy-Focused**: Your documents stay on your machine - no data sent to external services by default

## Installation

```bash
# Install from PyPI
pip install oboyu

# Or install from source
git clone https://github.com/sonesuke/oboyu.git
cd oboyu
pip install -e .
```

### Requirements

Oboyu requires several dependencies that are automatically installed:

1. **sentencepiece**: Required for the Japanese tokenization used by the Ruri embedding model. This is pre-built on PyPI but may require compilation on some systems.
2. **torch**: Required for running the embedding model.
3. **duckdb**: Required for storing and retrieving the vector search database.

> **Note**: On the first run, Oboyu will download the Ruri v3 model (~90MB) and its required components from the Hugging Face model hub.

### System Requirements

**Minimum Requirements:**
- CPU: Any modern x86_64 or ARM64 processor
- Memory: 2GB RAM (for embedding model + lightweight reranker)
- Storage: 500MB free space (for models and cache)

**Recommended Requirements:**
- CPU: Multi-core processor (4+ cores)
- Memory: 4GB+ RAM
- Storage: 2GB+ free space

**Memory Usage by Component:**
- Ruri v3 embedding model: ~300MB
- Lightweight reranker (default): ~400MB
- Heavy reranker (optional): ~1.2GB
- DuckDB + indexes: Variable based on document count

#### SentencePiece Installation

SentencePiece is a critical dependency for processing Japanese text. If you encounter issues during installation:

```bash
# On Ubuntu/Debian
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev

# On macOS with Homebrew
brew install cmake

# On Windows
# Install Visual Studio Build Tools with C++ support
```

## Quick Start

```bash
# Index a directory (incremental by default)
oboyu index /path/to/your/documents

# Index with specific file patterns and Japanese encoding detection
oboyu index /path/to/documents --include-patterns "*.txt,*.md" --japanese-encodings "utf-8,shift-jis,euc-jp"

# Force complete re-indexing (non-incremental)
oboyu index /path/to/documents --force

# Query your documents in Japanese (hybrid search with reranking enabled by default)
oboyu query "ドキュメント内の重要な概念は何ですか？"

# Query in English with specific search mode and number of results
oboyu query "What are the key concepts in the documents?" --mode vector --top-k 10

# Use hybrid search with custom weights
oboyu query "database optimization techniques" --vector-weight 0.8 --bm25-weight 0.2

# Query with reranking explicitly enabled (default behavior)
oboyu query "技術的な実装の詳細" --rerank

# Query without reranking for faster results
oboyu query "Quick overview of the project" --no-rerank

# Get detailed explanation of search results
oboyu query "Important design principles" --explain

# Start interactive query session for continuous searches
oboyu query --interactive

# Interactive mode with reranker pre-loaded
oboyu query --interactive --rerank --mode hybrid

# Clear the entire index database (requires confirmation unless --force is used)
oboyu clear

# Clear with a specific database path
oboyu clear --db-path custom.db --force

# Check indexing status for a directory
oboyu index manage status /path/to/documents

# See what would be updated (dry-run)
oboyu index manage diff /path/to/documents

# Start MCP server for AI assistant integration
oboyu mcp

# Check the current version
oboyu version
```

## Configuration

Create a configuration file at the XDG-compliant location `~/.config/oboyu/config.yaml`:

```yaml
# Crawler settings for document discovery
crawler:
  depth: 10
  include_patterns:
    - "*.txt"
    - "*.md"
    - "*.html"
    - "*.py"
    - "*.java"
    - "*.js"
    - "*.ts"
    - "*.yaml"
    - "*.yml"
    - "*.json"
    - "*.toml"
    - "*.rst"
    - "*.ipynb"
  exclude_patterns:
    - "*/node_modules/*"
    - "*/.venv/*"
    - "*/__pycache__/*"
    - "*/.git/*"
  max_file_size: 10485760  # 10MB
  follow_symlinks: false
  japanese_encodings:
    - "utf-8"
    - "shift-jis"
    - "euc-jp"
  max_workers: 4
  respect_gitignore: true

# Indexer settings for embedding and storage
indexer:
  # Text processing
  chunk_size: 1024
  chunk_overlap: 256
  
  # Embedding model
  embedding_model: "cl-nagoya/ruri-v3-30m"
  embedding_device: "cpu"
  batch_size: 8
  
  # Database
  db_path: "~/.oboyu/index.db"
  
  # Reranker settings (enabled by default)
  use_reranker: true
  reranker_model: "cl-nagoya/ruri-reranker-small"
  reranker_use_onnx: true
  reranker_top_k_multiplier: 3
  reranker_score_threshold: 0.5
  
  # Change detection for incremental indexing
  change_detection_strategy: "smart"  # timestamp, hash, or smart
  cleanup_deleted_files: true
  enable_onnx_optimization: true

# Query settings for search behavior
query:
  default_mode: "hybrid"
  vector_weight: 0.7
  bm25_weight: 0.3
  top_k: 5
  use_reranker: true
  snippet_length: 160
  highlight_matches: true
  show_scores: false
```

Oboyu will create a default configuration file with these settings if none exists. You can override any of these settings via command-line options or by editing the configuration file.

See the [configuration documentation](docs/configuration.md) for complete details and advanced options.

## Contributing

Contributions are welcome! Please see our [Development Guide](docs/development.md) for detailed instructions.

### Quick Start for Contributors

1. **Fork the repository** and clone your fork
2. **Install dependencies**: `uv sync`
3. **Run tests**: `uv run pytest -m "not slow"`
4. **Make your changes** with tests and documentation
5. **Submit a pull request** with clear description

### Development Commands

```bash
# Install dependencies
uv sync

# Run fast tests
uv run pytest -m "not slow" -k "not integration"

# Format and lint code
uv run ruff check --fix

# Type checking
uv run mypy
```

For comprehensive development guidelines, testing procedures, and architecture information, see the [Development Guide](docs/development.md).

## Documentation

### User Guides
- [CLI Commands](docs/cli.md) - Complete command-line interface reference
- [Configuration Options](docs/configuration.md) - YAML configuration and settings
- [Japanese Language Support](docs/japanese.md) - Specialized Japanese text processing
- [MCP Server Integration](docs/mcp_server.md) - AI assistant integration guide
- [Reranker Guide](docs/reranker.md) - Advanced reranking for better accuracy

### Technical Documentation
- [Architecture Overview](docs/architecture.md) - System design and components
- [Query Engine](docs/query_engine.md) - Search algorithms and modes
- [Indexer](docs/indexer.md) - Document processing and embedding generation
- [Crawler](docs/crawler.md) - Document discovery and extraction

### Development
- [Development Guide](docs/development.md) - Setup, testing, and contribution guidelines

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The name "Oboyu" (覚ゆ) comes from ancient Japanese, meaning "to remember" or "to memorize"
- Thanks to the open-source Japanese NLP community for their excellent tools and resources
- Built with the inspiration of making knowledge more accessible regardless of language
