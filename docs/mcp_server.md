# Oboyu MCP Server

The Oboyu MCP (Model Context Protocol) Server enables AI assistants to access Oboyu's Japanese-enhanced semantic search capabilities using a standardized protocol.

## What is MCP?

Model Context Protocol (MCP) is a standard protocol that enables AI assistants to interact with external tools and services. By implementing an MCP server, Oboyu allows AI assistants to:

- Search through your indexed documents
- Retrieve semantic search results with Japanese language optimization
- Perform both vector and keyword-based searches
- Access hybrid search combining both approaches

## Getting Started

### Prerequisites

To use the Oboyu MCP server, you need:

1. An existing Oboyu index (create one using `oboyu index <directory>`)
2. Oboyu installed with MCP dependencies (included by default)

### Running the MCP Server

Start the MCP server with:

```bash
oboyu mcp
```

By default, this runs the server using stdio transport, which is suitable for direct integration with AI assistant platforms like Claude Desktop.

### Command Options

The MCP server command supports several options:

| Option | Description |
|--------|-------------|
| `--db-path PATH` | Path to the database file (default: `~/.oboyu/index.db`) |
| `--transport, -t TYPE` | Transport mechanism: stdio, sse, streamable-http (default: stdio) |
| `--port, -p NUMBER` | Port number for SSE or streamable-http transport (required for non-stdio transports) |
| `--debug` | Enable debug mode with additional logging |

### Transport Types

- **stdio** (default): Standard input/output, ideal for Claude Desktop and similar integrations
- **sse**: Server-Sent Events over HTTP, useful for web-based integrations
- **streamable-http**: HTTP with streaming support, for advanced use cases

### Examples

Start the MCP server with stdio transport (default):

```bash
oboyu mcp
```

Start with SSE transport on port 8080:

```bash
oboyu mcp --transport sse --port 8080
```

Start with custom database path:

```bash
oboyu mcp --db-path /path/to/custom.db
```

## MCP Tools Provided

The Oboyu MCP server provides the following tool:

### search_documents

Search indexed documents using semantic search with Japanese language optimization.

**Parameters:**
- `query` (string, required): The search query in any language
- `limit` (integer, optional): Number of results to return (default: 5)
- `mode` (string, optional): Search mode - "vector", "bm25", or "hybrid" (default: "hybrid")
- `vector_weight` (float, optional): Weight for vector scores in hybrid mode (default: 0.7)
- `bm25_weight` (float, optional): Weight for BM25 scores in hybrid mode (default: 0.3)
- `snippet_config` (object, optional): Configuration for snippet generation and context control

**Snippet Configuration Options:**
- `length` (integer): Maximum snippet length in characters (default: 300)
- `context_window` (integer): Characters before/after match for context (default: 50)
- `max_snippets_per_result` (integer): Maximum snippets per search result (default: 1)
- `highlight_matches` (boolean): Whether to highlight search matches (default: true)
- `strategy` (string): Snippet boundary strategy - "fixed_length", "sentence_boundary", or "paragraph_boundary" (default: "sentence_boundary")
- `prefer_complete_sentences` (boolean): Try to end snippets at sentence boundaries (default: true)
- `include_surrounding_context` (boolean): Include context around matches (default: true)
- `japanese_aware` (boolean): Consider Japanese sentence boundaries (default: true)
- `levels` (array): Multi-level snippet configurations with type and length

**Returns:**
List of search results, each containing:
- `path`: File path of the document
- `title`: Document or chunk title
- `content`: Relevant text snippet (processed according to snippet_config)
- `score`: Relevance score (0-1)

**Example Response:**
```json
[
  {
    "path": "/docs/ml-guide.md",
    "title": "Machine Learning Guide",
    "content": "機械学習の基本的なアルゴリズムには、教師あり学習、教師なし学習、強化学習があります...",
    "score": 0.92
  }
]
```

## Integration with Claude Desktop

The MCP server is designed to work seamlessly with Claude Desktop:

### Configuration

1. Configure Claude Desktop settings
2. Add the Oboyu MCP server to the configuration:

```json
{
  "servers": {
    "oboyu": {
      "command": "oboyu",
      "args": ["mcp"],
      "env": {}
    }
  }
}
```

3. Restart Claude Desktop
4. Use natural language to search your indexed documents:
   - "Search for information about machine learning algorithms"
   - "Find documentation about database design patterns"
   - "Show me examples of Python async programming"

### Custom Database Path

If your index is not in the default location:

```json
{
  "servers": {
    "oboyu": {
      "command": "oboyu",
      "args": ["mcp", "--db-path", "/path/to/your/index.db"],
      "env": {}
    }
  }
}
```

## Search Examples

The MCP server provides flexible search capabilities through AI assistants:

### Hybrid Search (Default)
```
# Through MCP tools in Claude or other AI assistants
search_documents("機械学習の基本的なアルゴリズム", limit=5)
```

### Vector Search
```
search_documents("design patterns in software architecture", mode="vector", limit=10)
```

### BM25 Search
```
search_documents("データベース正規化", mode="bm25")
```

### Custom Hybrid Weights
```
search_documents("REST API best practices", 
                mode="hybrid", 
                vector_weight=0.6, 
                bm25_weight=0.4, 
                limit=8)
```

### Snippet Context Control
```
# Basic snippet configuration
search_documents("機械学習の原則", 
                snippet_config={
                  "length": 200,
                  "highlight_matches": true,
                  "strategy": "sentence_boundary"
                })

# Japanese-aware snippet processing
search_documents("システム設計の考え方", 
                snippet_config={
                  "length": 150,
                  "japanese_aware": true,
                  "prefer_complete_sentences": true,
                  "context_window": 30
                })

# Multi-level snippets
search_documents("database design patterns", 
                snippet_config={
                  "levels": [
                    {"type": "summary", "length": 100},
                    {"type": "detailed", "length": 300}
                  ],
                  "highlight_matches": false
                })
```

## Use Cases with AI Assistants

The MCP server enables powerful search workflows:

**Code Documentation Search:**
- "How do I implement authentication in this project?"
  → Uses hybrid search to find auth-related docs and code examples

**Technical Reference Lookup:**
- "What are the database migration patterns used here?"
  → Uses BM25 search for exact terminology matching

**Conceptual Queries:**
- "Explain the architecture design principles in this codebase"
  → Uses vector search for semantic understanding

**Japanese Content Search:**
- "プロジェクトでの非同期処理の実装方法を教えて"
  → Leverages Japanese language optimization for accurate results

**Snippet Context Control:**
- "Show me brief summaries of design patterns in this codebase"
  → Uses short snippet length (50-100 chars) for overview
- "I need detailed explanations of the authentication flow"
  → Uses longer snippets (300+ chars) with complete sentences
- "Find Japanese documentation about API usage"
  → Uses Japanese-aware sentence boundaries for natural text flow

## Performance Considerations

- The MCP server maintains a persistent connection to the database for fast responses
- Search operations typically complete in 50-200ms depending on index size
- Japanese text processing is optimized with MeCab tokenization caching
- Vector similarity search uses HNSW index for efficient approximate nearest neighbor lookups
- Hybrid search runs vector and BM25 searches in parallel for optimal performance

## Troubleshooting

### Common Issues

**MCP server won't start:**
- Ensure you have an existing index (run `oboyu index <directory>` first)
- Check that the database path exists and is readable
- Verify the transport type and port combination is valid

**No results returned:**
- Verify documents have been indexed properly with `oboyu index manage status`
- Check if the query language matches document language
- Try different search modes (vector vs BM25 vs hybrid)

**Connection errors with AI assistants:**
- Ensure the MCP server is running (`oboyu mcp`)
- Check the configuration JSON syntax in your AI assistant settings
- Verify the command path if Oboyu is installed in a virtual environment

### Debug Mode

Enable debug mode to see detailed server operations:

```bash
oboyu mcp --debug
```

This will output:
- MCP protocol messages
- Search query processing details
- Database operations
- Performance timing information

## Implementation Details

For advanced use cases, you can customize the MCP server by modifying the source code in the `src/oboyu/mcp/` directory:

- `server.py`: Main server implementation with tool definitions
- `context.py`: Search context management and result formatting