# Oboyu (覚ゆ)

> A Japanese-enhanced semantic search system for your local documents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/downloads/)

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

#### System Requirements

- **OS**: Linux, macOS (Windows not officially supported)
- **CPU**: x86_64 or ARM64 processor
- **Memory**: 2GB+ RAM
- **Storage**: 1GB free space

## Quick Start

```bash
# Index a directory (automatic encoding detection for Japanese files)
oboyu index /path/to/your/documents

# Query your documents
oboyu query "What are the key concepts?"

# Interactive query mode
oboyu query --interactive

# Start MCP server for AI integration
oboyu mcp
```

For more examples, see the [CLI documentation](docs/cli.md).

## Configuration

Oboyu works out of the box with sensible defaults. For customization, create `~/.config/oboyu/config.yaml`:

```yaml
indexer:
  chunk_size: 1024
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true

query:
  default_mode: "hybrid"
  top_k: 5
```

See the [configuration documentation](docs/configuration.md) for all available options.

## Contributing

We love your contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Start

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/oboyu.git
cd oboyu

# Install dependencies
uv sync

# Run fast tests
uv run pytest -m "not slow"

# Make your changes and submit a PR!
```

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


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- The name "Oboyu" (覚ゆ) comes from ancient Japanese, meaning "to remember" or "to memorize"
- Thanks to the open-source Japanese NLP community for their excellent tools and resources
- Built with the inspiration of making knowledge more accessible regardless of language
