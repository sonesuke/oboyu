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

![Oboyu Component Architecture](images/oboyu_components.png)

## Data Flow

The system follows a straightforward data flow:

```
Document Sources → Crawler → Indexer → Database
                                ↑
                                ↓
                    User Query → Query Engine → Results
```

## Technology Stack

- **Core Language**: Python for cross-platform compatibility and rapid development
- **Database**: DuckDB for serverless operation with vector and text search capabilities
- **Embedding Models**: Multilingual models with Japanese language optimization
- **Japanese Processing**: Specialized tokenizers for Japanese text handling

## Database Schema Overview

Oboyu uses a carefully designed database schema with separate tables for:

- Documents (metadata and content)
- Chunks (document segments)
- Embeddings (vector representations)

This separation allows for flexible indexing and search strategies.

## Configuration System

The system is configured through a YAML file located at `~/.oboyu/config.yaml`, providing extensive customization options while maintaining sensible defaults.

## Integration Points

- **Command Line Interface**: Direct document indexing and querying
- **MCP Server**: Standard stdio interface for integration with other tools

For detailed information on each component, see:
- [Crawler Architecture](crawler.md)
- [Indexer Architecture](indexer.md)
- [Query Engine Architecture](query.md)