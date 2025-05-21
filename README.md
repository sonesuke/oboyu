# Oboyu (覚ゆ)

> A Japanese-enhanced semantic search system for your local documents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

## What is Oboyu?

**Oboyu** (覚ゆ - meaning "to remember" or "to memorize" in ancient Japanese) is a semantic search system that creates an intelligent index of your text-based documents. Oboyu indexes directories of text files, embeds their content using vector representations, and retrieves the most relevant documents for your queries.

With specialized support for Japanese language documents, Oboyu excels at processing content in both Japanese and English, making it an ideal solution for multilingual document collections.

The system provides both command-line interface for direct queries and an MCP server mode for network-accessible document search. While Oboyu works with any text-based documents, it is particularly optimized for Japanese language content, making it unique among semantic search tools.

![Oboyu Concept](docs/images/oboyu_concept.png)

## Key Features

- **Local Directory Processing**: Index any directory of text-based documents on your local system
- **Text Format Support**: Process plain text, markdown, code files, configuration files, and more
- **Japanese Language Excellence**: First-class support for Japanese text with built-in specialized tokenization and encoding detection
- **Semantic Search**: Retrieve the most relevant documents using vector embeddings with the Ruri v3 model
- **Multiple Search Modes**: Choose between vector, BM25, or hybrid search depending on your needs
- **Document-Focused Results**: Get top matching documents with URIs, titles, and relevant snippets
- **Rich Command-Line Interface**: Powerful CLI with extensive options and colorized output
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
# Index a directory
oboyu index /path/to/your/documents

# Index with specific file patterns and Japanese encoding detection
oboyu index /path/to/documents --include-patterns "*.txt,*.md" --japanese-encodings "utf-8,shift-jis,euc-jp"

# Query your documents in Japanese (returns top matching documents with snippets)
oboyu query "ドキュメント内の重要な概念は何ですか？"

# Query in English with specific search mode and number of results
oboyu query "What are the key concepts in the documents?" --mode vector --top-k 10

# Get detailed explanation of search results
oboyu query "Important design principles" --explain

# Clear the entire index database (requires confirmation unless --force is used)
oboyu clear

# Clear with a specific database path
oboyu clear --db-path custom.db --force

# Check the current version
oboyu version
```

## Configuration

Create a configuration file at `~/.oboyu/config.yaml`:

```yaml
# Crawler settings
crawler:
  depth: 10
  include_patterns:
    - "*.txt"
    - "*.md"
    - "*.html"
    - "*.py"
    - "*.java"
  exclude_patterns:
    - "*/node_modules/*"
    - "*/venv/*"
  max_file_size: 10485760  # 10MB
  follow_symlinks: false
  japanese_encodings:
    - "utf-8"
    - "shift-jis"
    - "euc-jp"
  max_workers: 4

# Indexer settings
indexer:
  chunk_size: 1024
  chunk_overlap: 256
  embedding_model: "cl-nagoya/ruri-v3-30m"
  embedding_device: "cpu"
  batch_size: 8
  db_path: "oboyu.db"

# Query settings
query:
  default_mode: "hybrid"
  vector_weight: 0.7
  bm25_weight: 0.3
  top_k: 5
  snippet_length: 160
  highlight_matches: true
```

Oboyu will create a default configuration file with these settings if none exists. You can override any of these settings via command-line options or by editing the configuration file.

See the [configuration documentation](docs/configuration.md) for more options.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Documentation

- [Configuration Options](docs/configuration.md)
- [CLI Commands](docs/cli.md)
- [API Reference](docs/api.md)
- [Japanese Support Details](docs/japanese.md)
- [Architecture Overview](docs/architecture.md)
- [MCP Server Guide](docs/mcp_server.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The name "Oboyu" (覚ゆ) comes from ancient Japanese, meaning "to remember" or "to memorize"
- Thanks to the open-source Japanese NLP community for their excellent tools and resources
- Built with the inspiration of making knowledge more accessible regardless of language
