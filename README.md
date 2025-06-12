# Oboyu (覚ゆ)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/oboyu.svg)](https://pypi.org/project/oboyu/)

> Lightning-fast semantic search for your local documents with best-in-class Japanese support.

![demo](https://github.com/sonesuke/oboyu/blob/main/docs/assets/demo.gif?raw=true)

## What is Oboyu?

**Oboyu** (覚ゆ - "to remember" in ancient Japanese) is a powerful local semantic search engine that helps you instantly find information in your documents using natural language queries. Unlike traditional keyword search, Oboyu understands the meaning behind your questions, making it perfect for finding relevant content even when you don't know the exact terms.

### Why Oboyu?

- 🚀 **Fast**: Indexes thousands of documents in seconds, searches in milliseconds
- 🎯 **Accurate**: Semantic search finds what you mean, not just what you type
- 🇯🇵 **Japanese Excellence**: First-class support with automatic encoding detection
- 🔒 **Private**: Everything runs locally - your documents never leave your machine
- 🤖 **AI-Ready**: Built-in MCP server for Claude, Cursor, and other AI assistants


## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip (latest version recommended)
- Operating System: Linux, macOS, or Windows with WSL
- For building from source:
  - C++ compiler (build-essential on Linux, Xcode on macOS)
  - CMake (for sentencepiece)

### Installation

Get up and running in under 5 minutes:

```bash
# Install Oboyu
pip install oboyu

# Index your documents
oboyu index ~/Documents

# Search interactively
oboyu query --interactive
```

That's it! See our [Documentation](https://sonesuke.github.io/oboyu/) for complete guides and examples.

## Key Features

### 🔍 Advanced Search Capabilities
- **Hybrid Search**: Combines semantic understanding with keyword matching for best results
- **Multiple Modes**: Switch between semantic, keyword, or hybrid search modes
- **Smart Reranking**: Built-in AI reranker improves result accuracy
- **Interactive Mode**: Real-time search with command history and auto-suggestions

### 📚 Document Support
- **Rich Format Support**: PDF documents, plain text (.txt), Markdown (.md), HTML (.html), and source code files (.py, .java, etc.)
- **PDF Processing**: Full text extraction with metadata preservation from PDF documents
- **Incremental Indexing**: Only process new or changed files for lightning-fast updates
- **Smart Chunking**: Intelligent document splitting for optimal search results
- **Automatic Encoding**: Handles various text encodings seamlessly (UTF-8, Shift-JIS, EUC-JP, and more)

### 🇯🇵 Japanese Language Excellence
- **Native Support**: Purpose-built for Japanese text processing
- **Automatic Detection**: Detects and handles Shift-JIS, EUC-JP, and UTF-8
- **Specialized Models**: Optimized embedding models for Japanese content
- **Mixed Language**: Seamlessly handles Japanese and English in the same document

### 🚀 Performance & Integration
- **ONNX Acceleration**: 2-4x faster with automatic model optimization
- **MCP Server**: Direct integration with Claude Desktop and AI coding assistants
- **Rich CLI**: Beautiful terminal interface with progress tracking
- **Low Memory**: Efficient processing even on modest hardware

## Installation

### Using UV (Recommended)
```bash
uv tool install oboyu
```

### Using pip
```bash
pip install oboyu
```

### From Source
```bash
git clone https://github.com/sonesuke/oboyu.git
cd oboyu
pip install -e .
```

### System Requirements

- **Python**: 3.13 or higher
- **OS**: macOS, Linux (Windows via WSL)
- **Memory**: 2GB RAM minimum
- **Storage**: 1GB for models and index

> **Note**: Models are automatically downloaded on first use (~90MB).

## Usage Examples

### Basic Usage

```bash
# Index a directory
oboyu index ~/Documents/notes

# Search your documents
oboyu query "machine learning optimization techniques"

# Interactive mode (recommended!)
oboyu query --interactive
```

### Advanced Examples

```bash
# Index only specific file types
oboyu index ~/projects --include "*.md,*.txt"

# Search with filters
oboyu query "API design" --filter "docs/"

# Use semantic search mode
oboyu query "concepts similar to dependency injection" --mode semantic

# Enable reranking for better accuracy
oboyu query "complex technical topic" --rerank
```

### MCP Server for AI Assistants

```bash
# Start MCP server
oboyu mcp

# Or configure in Claude Desktop's settings
```

See our [MCP Integration Guide](https://sonesuke.github.io/oboyu/integration/mcp-integration) for detailed setup instructions.

## Documentation

### 🚀 Getting Started
- [**Installation**](https://sonesuke.github.io/oboyu/getting-started/installation) - Install and verify setup
- [**Your First Index**](https://sonesuke.github.io/oboyu/getting-started/first-index) - Create your first searchable index
- [**Your First Search**](https://sonesuke.github.io/oboyu/getting-started/first-search) - Learn to search effectively

### 💼 Real-world Usage
- [**Daily Workflows**](https://sonesuke.github.io/oboyu/usage-examples/basic-workflow) - Essential daily patterns
- [**Technical Documentation**](https://sonesuke.github.io/oboyu/real-world-scenarios/technical-docs) - Code and API docs
- [**Meeting Notes**](https://sonesuke.github.io/oboyu/real-world-scenarios/meeting-notes) - Track decisions and actions
- [**Research Papers**](https://sonesuke.github.io/oboyu/real-world-scenarios/research-papers) - Academic content search

### ⚙️ Configuration & Optimization
- [**Configuration Guide**](https://sonesuke.github.io/oboyu/configuration-optimization/configuration) - Customize for your needs
- [**Performance Tuning**](https://sonesuke.github.io/oboyu/configuration-optimization/performance-tuning) - Optimize speed and quality
- [**Japanese Support**](https://sonesuke.github.io/oboyu/reference-troubleshooting/japanese-support) - Japanese language features

### 🔗 Integration & Reference
- [**Claude MCP Integration**](https://sonesuke.github.io/oboyu/integration/mcp-integration) - AI-powered search
- [**CLI Reference**](https://sonesuke.github.io/oboyu/reference-troubleshooting/cli-reference) - All commands and options
- [**Troubleshooting**](https://sonesuke.github.io/oboyu/reference-troubleshooting/troubleshooting) - Solutions to common issues

**[📖 View Full Documentation →](https://sonesuke.github.io/oboyu/)**

## Common Use Cases

### 📚 Academic Research
Index and search through research notes and references:
```bash
oboyu index ~/research --include "*.md,*.txt"
oboyu query "transformer architecture improvements"
```

### 💻 Code Documentation
Search through project documentation and code comments:
```bash
oboyu index ~/projects/myapp --include "*.md,*.py"
oboyu query "authentication implementation"
```

### 📝 Personal Knowledge Base
Organize and search your notes and documents:
```bash
oboyu index ~/Documents/notes
oboyu query "meeting notes from last week"
```

### 🌏 Multilingual Documents
Perfect for mixed Japanese and English content:
```bash
oboyu index ~/Documents/bilingual
oboyu query "プロジェクト管理 best practices"
```

## Testing

### Unit and Integration Tests

```bash
# Run fast tests (recommended for development)
uv run pytest -m "not slow"

# Run all tests with coverage
uv run pytest --cov=src
```

### E2E Display Testing

Oboyu includes comprehensive E2E display testing using Claude Code SDK:

```bash
# Run all E2E display tests
python e2e/run_tests.py

# Run specific test category
python e2e/run_tests.py --test search
```

See our [Full Documentation](https://sonesuke.github.io/oboyu/) for more details.

## Contributing

We welcome contributions! See our [Contributing Guidelines](CONTRIBUTING.md) for details.

```bash
# Quick start for contributors
git clone https://github.com/YOUR_USERNAME/oboyu.git
cd oboyu
uv sync
uv run pytest -m "not slow"
```

## Support

- 📋 [GitHub Issues](https://github.com/sonesuke/oboyu/issues) - Report bugs or request features
- 📖 [Documentation](https://sonesuke.github.io/oboyu/) - Comprehensive guides and references
- 💬 [Discussions](https://github.com/sonesuke/oboyu/discussions) - Ask questions and share ideas

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- The name "Oboyu" (覚ゆ) comes from ancient Japanese, meaning "to remember"
- Built with ❤️ for the Japanese NLP community
- Inspired by the goal of making knowledge accessible across languages

---

<p align="center">
  Made with 🇯🇵 by <a href="https://github.com/sonesuke">sonesuke</a>
</p>
