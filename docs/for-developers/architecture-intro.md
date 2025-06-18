# Architecture & Internals

Welcome to the deep dive into Oboyu's architecture and internal workings! This section provides detailed technical information about how the system is designed and implemented.

## Who is this for?

This section is designed for:
- Developers curious about the system's internal design
- Technical users wanting to understand how Oboyu works
- Contributors needing architectural context
- Anyone interested in the technical implementation details

## What's included?

### System Architecture
- **[Architecture Overview](architecture.md)** - High-level system design and components
- **[Query Engine](architecture/query-engine.md)** - How searches are processed and optimized

### Core Components
- **[Crawler](crawler.md)** - Document discovery and extraction system
- **[Indexer](indexer.md)** - How documents are processed and indexed
- **[MCP Server](mcp_server.md)** - Model Context Protocol server implementation

### Advanced Topics
- **[CLI Architecture](cli.md)** - Command-line interface design
- **[Reranker](reranker.md)** - Result ranking and relevance optimization
- **[Japanese Support](japanese.md)** - Japanese language processing capabilities

## Understanding the Flow

1. **Document Ingestion**: The crawler discovers and extracts documents
2. **Processing**: The indexer processes and stores document content
3. **Querying**: The query engine handles search requests
4. **Ranking**: The reranker optimizes result relevance
5. **Serving**: The MCP server provides integration capabilities

## Key Design Principles

- **Modularity**: Each component has a single, well-defined responsibility
- **Extensibility**: Easy to add new document types and processing capabilities
- **Performance**: Optimized for fast indexing and searching
- **Accuracy**: Advanced ranking algorithms for relevant results

## Next Steps

- Start with the [Architecture Overview](architecture.md) for a high-level understanding
- Dive into specific components based on your interests
- Check the [For Contributors](contributors-intro.md) section if you want to contribute

Happy exploring! 🏗️