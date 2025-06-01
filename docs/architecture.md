# Oboyu Architecture Overview

## System Design Philosophy

Oboyu (覚ゆ) is designed with a clear architectural vision: to provide powerful semantic search for local documents with exceptional Japanese language support. The system embraces simplicity, privacy, and efficiency while offering advanced search capabilities.

## Core Architectural Principles

- **Local-First**: All processing occurs on the user's machine with no data sent externally
- **Modular Design**: Clean separation of concerns with distinct components
- **Japanese Excellence**: First-class support for Japanese throughout the system
- **Flexibility**: Support for multiple search methodologies (vector, BM25, hybrid)
- **Minimal Dependencies**: Self-contained system with few external requirements

## Component Architecture

Oboyu is built around three primary components, each with distinct responsibilities:

1. **Crawler**: Discovers and extracts documents from the file system
2. **Indexer**: Processes documents and builds search indexes
3. **Query Engine**: Handles search requests and returns relevant results

## Component Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Crawler     │────│     Indexer     │────│  Query Engine   │
│                 │    │                 │    │                 │
│ • Discovery     │    │ • Processing    │    │ • Vector Search │
│ • Extraction    │    │ • Embedding     │    │ • BM25 Search   │
│ • Japanese      │    │ • Storage       │    │ • Hybrid Search │
│   Processing    │    │ • Change        │    │ • Reranking     │
│                 │    │   Detection     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DuckDB Database                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ file_metadata│ │    chunks    │ │      embeddings          │ │
│  │              │ │              │ │                          │ │
│  │ • path       │ │ • content    │ │ • vector                 │ │
│  │ • metadata   │ │ • language   │ │ • similarity search      │ │
│  │ • checksums  │ │ • metadata   │ │   (VSS extension)        │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

The system follows a straightforward data flow:

```
Document Sources → Crawler → Indexer → Database
                                ↑
                                ↓
                    User Query → Query Engine → Results
```

## Technology Stack

- **Core Language**: Python 3.8+ for cross-platform compatibility
- **Database**: DuckDB with VSS extension for vector similarity search and full-text indexing
- **Embedding Models**: Ruri v3 (cl-nagoya/ruri-v3-30m) with Japanese optimization
- **Reranker Models**: Ruri Cross-Encoder (cl-nagoya/ruri-reranker-small) for result refinement
- **Japanese Processing**: MeCab morphological analyzer via fugashi library
- **Search Algorithms**: Vector search (HNSW), BM25, and hybrid approaches
- **ONNX Optimization**: Automatic model conversion for 2-4x inference speedup
- **CLI Framework**: Typer with Rich for interactive command-line interface
- **MCP Integration**: Model Context Protocol server for AI assistant integration

## Database Schema Overview

Oboyu uses a carefully designed DuckDB schema optimized for semantic search:

### Core Tables

- **`file_metadata`**: File information, checksums, processing metadata
- **`chunks`**: Document segments with content, language detection, and metadata
- **`embeddings`**: Vector representations with VSS extension for similarity search

### BM25 Search Tables  

- **`vocabulary`**: Term vocabulary with IDF scores
- **`inverted_index`**: Term-to-document mappings with TF scores
- **`document_stats`**: Document length and term count statistics
- **`collection_stats`**: Collection-wide statistics for BM25 scoring

### Meta Tables

- **`schema_version`**: Database schema versioning for safe migrations

### Key Features

- **VSS Extension**: Vector similarity search with HNSW indexing
- **Full-Text Search**: Native DuckDB FTS for exact term matching
- **Incremental Updates**: Change detection prevents redundant processing
- **Schema Migrations**: Version-controlled database schema evolution
- **Transaction Safety**: ACID compliance for reliable updates

## Interface Architecture

### Command-Line Interface

Oboyu provides a rich CLI with multiple interaction modes:

- **Single Commands**: Direct file indexing and one-shot queries
- **Interactive Mode**: Persistent REPL for continuous searching with session state
- **Management Commands**: Index status checking, differential updates, clearing

### MCP Server Mode

The Model Context Protocol (MCP) server enables AI assistant integration:

- **Transport Options**: stdio, Server-Sent Events (SSE), streamable-http
- **Tool Exposure**: Search, indexing, index management via standardized protocol
- **Session Management**: Persistent database connections for multiple queries
- **Error Handling**: Robust error reporting and recovery

### API Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interfaces                         │
├─────────────────────┬─────────────────────┬─────────────────┤
│    CLI Commands     │  Interactive Mode   │   MCP Server    │
├─────────────────────┼─────────────────────┼─────────────────┤
│ • index             │ • /search           │ • search_tool   │
│ • query             │ • /mode             │ • index_tool    │
│ • clear             │ • /settings         │ • clear_tool    │
│ • mcp               │ • /stats            │ • status_tool   │
└─────────────────────┴─────────────────────┴─────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     Core Engine                            │
├─────────────────────┬─────────────────────┬─────────────────┤
│     Crawler         │      Indexer        │  Query Engine   │
└─────────────────────┴─────────────────────┴─────────────────┘
```

## Configuration System

The system is configured through a YAML file located at `~/.oboyu/config.yaml`, providing extensive customization options while maintaining sensible defaults.

## Integration Points

- **Command Line Interface**: Direct document indexing and querying
- **MCP Server**: Standard stdio interface for integration with other tools

For detailed information on each component, see:
- [Crawler Architecture](crawler.md)
- [Indexer Architecture](indexer.md)
- [Query Engine Architecture](query_engine.md)