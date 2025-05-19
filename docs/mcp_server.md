# Oboyu MCP Server

The Oboyu MCP (Model Context Protocol) Server enables AI assistants to access Oboyu's Japanese-enhanced semantic search capabilities using a standardized protocol.

## What is MCP?

Model Context Protocol (MCP) is a standard protocol that enables AI assistants to interact with external tools and services. By implementing an MCP server, Oboyu allows AI assistants to:

- Search through your indexed documents
- Retrieve semantic search results with Japanese language optimization
- Access information about your document index

## Getting Started

### Prerequisites

To use the Oboyu MCP server, you need:

1. An existing Oboyu index (create one using `oboyu index <directory>`)
2. The `mcp[cli]` package installed (included in Oboyu dependencies)

### Running the MCP Server

Start the MCP server with:

```bash
oboyu mcp
```

By default, this runs the server using stdio transport, which is suitable for direct integration with AI assistant platforms.

### Command Options

The MCP server command supports several options:

| Option | Description |
|--------|-------------|
| `--db-path PATH` | Path to the database file (default: `~/.oboyu/index.db`) |
| `--transport, -t TYPE` | Transport mechanism: stdio, http, websocket (default: stdio) |
| `--port, -p NUMBER` | Port number for HTTP or WebSocket transport (required if using those transports) |
| `--verbose, -v` | Enable verbose output |
| `--debug, -d` | Enable debug mode with additional logging |

### Examples

Start the MCP server with stdio transport (default):

```bash
oboyu mcp
```

Start the MCP server with HTTP transport on port 8000:

```bash
oboyu mcp --transport http --port 8000
```

Start the MCP server with a specific database path:

```bash
oboyu mcp --db-path /path/to/custom/index.db
```

## Available Tools

The Oboyu MCP server provides the following tools to AI assistants:

### Search Tool

The `search` tool allows AI assistants to search for documents in your Oboyu index.

**Parameters:**

- `query` (string, required): The search query text
- `mode` (string, optional): Search mode (vector, bm25, hybrid) - default: hybrid
- `top_k` (integer, optional): Maximum number of results to return - default: 5
- `language` (string, optional): Language filter (e.g., 'ja', 'en')
- `db_path` (string, optional): Custom database path

**Example Response:**

```json
{
  "results": [
    {
      "title": "Japanese Grammar Guide",
      "content": "日本語の文法について説明します。日本語は主語-目的語-動詞の語順です。",
      "uri": "file:///path/to/document.md",
      "score": 0.89,
      "language": "ja",
      "metadata": {"source": "grammar-guide"}
    }
  ],
  "stats": {
    "count": 1,
    "query": "日本語 文法",
    "language_filter": "ja"
  }
}
```

### Index Directory Tool

The `index_directory` tool allows AI assistants to add new documents to the Oboyu index.

**Parameters:**

- `directory_path` (string, required): Path to the directory to index
- `incremental` (boolean, optional): Only index new or changed files - default: true
- `db_path` (string, optional): Custom database path

**Example Response:**

```json
{
  "success": true,
  "directory": "/path/to/documents",
  "documents_indexed": 25,
  "chunks_indexed": 142,
  "db_path": "/home/user/.oboyu/index.db"
}
```

### Clear Index Tool

The `clear_index` tool allows AI assistants to reset the Oboyu index.

**Parameters:**

- `db_path` (string, optional): Custom database path

**Example Response:**

```json
{
  "success": true,
  "message": "Index database cleared successfully",
  "db_path": "/home/user/.oboyu/index.db"
}
```

### Get Index Info Tool

The `get_index_info` tool provides information about your Oboyu index.

**Parameters:**

- `db_path` (string, optional): Custom database path

**Example Response:**

```json
{
  "document_count": 157,
  "chunk_count": 1253,
  "languages": ["ja", "en", "fr"],
  "embedding_model": "ruri-v3-30m",
  "db_path": "/home/user/.oboyu/index.db",
  "last_updated": "2025-05-18T12:34:56Z"
}
```

## Integration with AI Assistants

The Oboyu MCP server is designed to be integrated with AI assistant platforms that support the Model Context Protocol. Check your AI assistant platform's documentation for specific integration instructions.

### Example Integration with Claude

To integrate Oboyu with Claude:

1. Start the Oboyu MCP server:
   ```bash
   oboyu mcp
   ```

2. Configure Claude to use the Oboyu MCP tools via your platform's MCP integration features.

3. You can now ask Claude questions like:
   - "Search my documents for information about Japanese grammar"
   - "Find documents related to 日本の歴史 (Japanese history)"
   - "How many documents are in my Oboyu index?"

## Japanese Language Support

Oboyu MCP server preserves all the Japanese language optimizations from the core Oboyu system:

- Accurate handling of Japanese queries
- Proper tokenization of Japanese text
- Support for mixed Japanese and English content
- Proper display of Japanese characters in search results

This makes Oboyu particularly valuable for AI assistants when working with Japanese content.

## Troubleshooting

### Common Issues

1. **Missing Index**: Ensure you've created an index using `oboyu index` before starting the MCP server.

2. **Transport Errors**: If using HTTP/WebSocket transports, check that the specified port is available.

3. **Encoding Issues**: If Japanese text appears garbled, check your terminal's character encoding.

### Logs and Debugging

Enable verbose and debug mode for detailed logs:

```bash
oboyu mcp --verbose --debug
```

## Advanced Configuration

For advanced use cases, you can customize the MCP server by modifying the source code in the `src/oboyu/mcp/` directory, particularly:

- `server.py`: For customizing tool functionality
- `cli.py`: For modifying CLI behavior