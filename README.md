# Oboyu (覚ゆ)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/oboyu.svg)](https://pypi.org/project/oboyu/)

> ドキュメントを知識に、知識を価値に変える日本語特化型インテリジェンス・プラットフォーム  
> Transform documents into knowledge, knowledge into value - Japanese-optimized Intelligence Platform

![demo](https://github.com/sonesuke/oboyu/blob/main/docs/assets/demo.gif?raw=true)

## What is Oboyu?

**Oboyu** (覚ゆ - "to remember" in ancient Japanese) is a comprehensive Knowledge Intelligence Platform that transforms your documents into actionable insights. Going beyond traditional RAG (Retrieval-Augmented Generation), Oboyu combines advanced semantic search, knowledge graph generation, and AI-powered data enrichment to unlock the full potential of your information assets.

### Beyond Traditional RAG

While most solutions stop at document retrieval, Oboyu creates a living knowledge ecosystem:
- **Knowledge Graph Generation**: Automatically extracts entities, relationships, and concepts from your documents
- **GraphRAG Search**: Leverages knowledge graphs for deeper, more contextual search results
- **Data Enrichment**: Enhances CSV files and structured data with insights from your knowledge base
- **Multi-dimensional Intelligence**: Combines vector search, graph traversal, and semantic analysis

### Why Oboyu?

- 🧠 **Knowledge Intelligence**: Automatically generates knowledge graphs and extracts insights from your documents
- 📊 **Data Enrichment**: Enhances CSV files and structured data with AI-powered content from your knowledge base
- 🚀 **Lightning Fast**: Indexes thousands of documents in seconds, searches in milliseconds with GraphRAG acceleration
- 🎯 **Beyond Accurate**: Multi-layered search combining semantic understanding, knowledge graphs, and contextual reasoning
- 🇯🇵 **Japanese Excellence**: Built specifically for Japanese business environments with automatic encoding detection
- 🔒 **Enterprise Private**: Everything runs locally - your sensitive documents never leave your infrastructure
- 🤖 **AI-Native**: Built-in MCP server for Claude, Cursor, and other AI assistants with GraphRAG capabilities


## Quick Start

### Prerequisites

- Python 3.13 or higher (3.11+ supported)
- pip (latest version recommended)
- Operating System: Linux, macOS, or Windows with WSL

#### System Dependencies (for building from source)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libfreetype6-dev \
    libfontconfig1-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    libssl-dev
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install -y \
    git \
    curl \
    gcc-c++ \
    cmake \
    pkg-config \
    freetype-devel \
    fontconfig-devel \
    libjpeg-devel \
    libpng-devel \
    zlib-devel \
    openssl-devel
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install additional dependencies via Homebrew
brew install cmake pkg-config
```

### Installation

Get up and running in under 5 minutes:

```bash
# Install Oboyu
pip install oboyu

# Index your documents
oboyu index ~/Documents

# Search your documents
oboyu search "your search term"
```

That's it! See our [Documentation](https://sonesuke.github.io/oboyu/) for complete guides and examples.

## Key Features

### 🧠 Knowledge Intelligence
- **Automatic Knowledge Graph Generation**: Extracts entities, relationships, and concepts from your documents
- **GraphRAG Search**: Leverages knowledge graphs for deeper, contextual search results
- **Multi-dimensional Associations**: Discovers hidden connections between documents and concepts
- **Semantic Entity Recognition**: Identifies and links key entities across your knowledge base
- **Relationship Mapping**: Automatically maps relationships between concepts, people, and ideas

### 📊 Data Enrichment & Enhancement
- **CSV Auto-Enhancement**: Enriches CSV files with relevant information from your knowledge base
- **Schema-Driven Processing**: Uses JSON schema to define enrichment rules and data transformation
- **Semantic Data Completion**: Fills missing information using AI-powered content matching
- **Business Value Creation**: Transforms raw data into actionable business insights
- **Batch Processing**: Efficiently processes large datasets with configurable batch sizes

### 🔍 Advanced Search Capabilities
- **Hybrid Search**: Combines semantic understanding with keyword matching and graph traversal
- **Multiple Search Modes**: Vector search, keyword search, GraphRAG, and hybrid modes
- **AI-Powered Reranking**: Built-in reranker improves result accuracy and relevance
- **Contextual Understanding**: Uses knowledge graphs to provide more relevant results
- **Flexible Output**: Command-line search with JSON, plain text, and structured formats

### 📚 Comprehensive Document Support
- **Rich Format Support**: PDF, plain text (.txt), Markdown (.md), HTML (.html), and source code files
- **PDF Intelligence**: Advanced text extraction with metadata preservation and structure understanding
- **Incremental Indexing**: Only processes new or changed files for lightning-fast updates
- **Smart Chunking**: Intelligent document splitting optimized for knowledge extraction
- **Automatic Encoding**: Seamlessly handles UTF-8, Shift-JIS, EUC-JP, and other encodings

### 🇯🇵 Japanese Business Excellence
- **Native Japanese Support**: Purpose-built for Japanese business environments and content
- **Automatic Encoding Detection**: Handles legacy Japanese encodings (Shift-JIS, EUC-JP) automatically
- **Specialized Language Models**: Optimized embedding and processing models for Japanese text
- **Mixed Language Intelligence**: Seamlessly processes Japanese-English bilingual documents
- **Business Context Understanding**: Trained on Japanese business terminology and concepts

### 🚀 Enterprise Performance & Integration
- **ONNX Acceleration**: 2-4x faster processing with automatic model optimization
- **MCP Server Integration**: Native support for Claude Desktop and AI coding assistants
- **GraphRAG API**: RESTful API for knowledge graph queries and data enrichment
- **Rich CLI Interface**: Beautiful terminal interface with real-time progress tracking
- **Resource Efficient**: Low memory footprint suitable for edge computing and local deployment

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

- **Python**: 3.13 or higher (3.11+ supported)
- **OS**: macOS, Linux (Windows via WSL)
- **Memory**: 2GB RAM minimum (4GB recommended)
- **Storage**: 1GB for models and index
- **Build Tools**: See system dependencies above if building from source

> **Note**: Models are automatically downloaded on first use (~90MB).
> For installation from PyPI, most system dependencies are not required as we provide pre-built wheels.

## Usage Examples

### Basic Usage

```bash
# Index a directory
oboyu index ~/Documents/notes

# Search your documents
oboyu search "machine learning optimization techniques"

# Get results in JSON format for processing
oboyu search "machine learning" --format json
```

### Knowledge Intelligence & GraphRAG

```bash
# Build knowledge graph from your documents
oboyu build-kg

# Search using GraphRAG for deeper insights
oboyu search "project management methodologies" --mode graphrag

# Find related concepts and entities
oboyu search "agile development" --rerank --max-results 10
```

### Data Enrichment Workflows

**Schema Configuration (`enrichment_schema.json`):**
```json
{
  "input_schema": {
    "columns": {
      "company_name": {"type": "string", "description": "Company name"}
    }
  },
  "enrichment_schema": {
    "columns": {
      "description": {
        "type": "string",
        "source_strategy": "search_content",
        "query_template": "{company_name} company overview business model"
      },
      "industry": {
        "type": "string",
        "source_strategy": "search_content",
        "query_template": "{company_name} industry sector business domain"
      }
    }
  }
}
```

**Enrichment Commands:**
```bash
# Enrich CSV with knowledge from your documents
oboyu enrich companies.csv enrichment_schema.json

# Custom output location and batch processing
oboyu enrich data.csv schema.json -o enriched_data.csv --batch-size 5

# Disable GraphRAG for faster processing
oboyu enrich simple_data.csv schema.json --no-graph
```

### Advanced Search Examples

```bash
# Index only specific file types
oboyu index ~/projects --include-patterns "*.md,*.txt,*.pdf"

# GraphRAG search with relationship traversal
oboyu search "API design patterns" --mode graphrag --confidence 0.7

# Hybrid search combining multiple approaches
oboyu search "microservices architecture" --mode hybrid --rerank

# Search with custom result limits and confidence
oboyu search "database optimization" --max-results 15 --confidence 0.6
```

### MCP Server for AI Assistants

```bash
# Start MCP server with GraphRAG capabilities
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

## 🛠️ Technology Stack

Learn about the cutting-edge technologies that power Oboyu's intelligence:

- **[📚 Technology Stack Overview](docs/technology-stack/index.md)** - Complete stack architecture and philosophy
- **[🗄️ DuckDB: The Analytics Engine](docs/technology-stack/duckdb.md)** - Why DuckDB powers our knowledge intelligence
- **[🤖 HuggingFace: Japanese AI Excellence](docs/technology-stack/huggingface.md)** - Specialized Japanese language models and embeddings
- **[🔗 GraphRAG: Beyond Simple RAG](docs/technology-stack/graphrag.md)** - Graph-enhanced retrieval and knowledge understanding
- **[⚡ ONNX: Optimization Without Compromise](docs/technology-stack/onnx.md)** - 3x faster inference with maintained quality
- **[⚖️ Our Decision Framework](docs/technology-stack/decision-framework.md)** - How we evaluate and choose technologies

We believe in transparency and sharing our technical journey. These deep-dives include performance benchmarks, implementation insights, and honest assessments of alternatives.

## Common Use Cases

### 🏢 Enterprise Knowledge Management
Transform organizational documents into a searchable knowledge graph:
```bash
# Index company documents and build knowledge graph
oboyu index ~/company_docs --include "*.pdf,*.md,*.docx"
oboyu build-kg

# Search for strategic insights
oboyu search "competitive analysis market positioning" --mode graphrag
```

### 📊 Business Data Enhancement
Enrich customer or product data with insights from your knowledge base:
```bash
# Enhance customer list with company information
oboyu enrich customers.csv customer_enrichment_schema.json

# Add product descriptions from documentation
oboyu enrich products.csv product_schema.json --batch-size 10
```

### 📚 Research & Academic Intelligence
Create a comprehensive research knowledge base:
```bash
# Index research papers and notes
oboyu index ~/research --include "*.pdf,*.md,*.txt"
oboyu build-kg

# Find related concepts and methodologies
oboyu search "neural network optimization techniques" --mode graphrag
```

### 💻 Technical Documentation Intelligence
Make your codebase and documentation more discoverable:
```bash
# Index code and documentation
oboyu index ~/projects/myapp --include "*.md,*.py,*.js,*.java"

# Find implementation patterns and examples
oboyu search "authentication middleware patterns" --rerank
```

### 📋 Meeting & Decision Intelligence
Transform meeting notes into actionable insights:
```bash
# Index meeting notes and decisions
oboyu index ~/meetings --include "*.md,*.txt"

# Search for decisions and action items
oboyu search "budget approval Q4 initiatives" --mode hybrid
```

### 🌏 Multilingual Business Operations
Perfect for Japanese-English business environments:
```bash
# Index multilingual business documents
oboyu index ~/business_docs --include "*.pdf,*.md"

# Search across languages seamlessly
oboyu search "プロジェクト管理 project management methodology" --mode graphrag
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
- Built with ❤️ for the Japanese business and NLP community
- Inspired by the goal of making knowledge accessible and actionable across languages
- Special thanks to the TinySwallow model for Japanese language understanding and knowledge extraction
- GraphRAG implementation inspired by Microsoft's GraphRAG research and methodology

---

<p align="center">
  Made with 🇯🇵 by <a href="https://github.com/sonesuke">sonesuke</a>
</p>
