---
id: mcp-integration
title: Claude MCP Integration Usage
sidebar_position: 1
---

# Claude MCP Integration Usage

Integrate Oboyu with Claude Desktop through the Model Context Protocol (MCP) for AI-powered document search and retrieval. This guide shows you how to set up and use Oboyu as an MCP server.

## What is MCP?

The Model Context Protocol (MCP) enables AI assistants like Claude to interact with external tools and data sources. With Oboyu's MCP integration, Claude can:

- Search your local documents
- Retrieve relevant context
- Answer questions based on your files
- Help with document-based tasks

## Quick Start

### 1. Install and Index Documents

```bash
# Install Oboyu (follow installation guide)
pip install oboyu

# Create an index of your documents
oboyu index ~/Documents
```

### 2. Configure Claude Desktop

Add Oboyu to your Claude Desktop configuration:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "oboyu": {
      "command": "oboyu",
      "args": ["mcp"],
      "description": "Search local documents with Oboyu"
    }
  }
}
```

### 3. Start Using

In Claude Desktop, you can now ask:
```
Search my documents for information about project deadlines
```

Claude will use Oboyu to search your indexed documents and provide relevant information.

## MCP Server Configuration

### Basic Server Options

```bash
# Start MCP server with default database
oboyu mcp

# Use specific database
oboyu mcp --db-path ~/indexes/work.db

# Enable debug mode
oboyu mcp --debug --verbose

# Use HTTP transport (for web integrations)
oboyu mcp --transport streamable-http --port 8080
```

### Available MCP Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--db-path PATH` | Database file path | Default location | `--db-path ~/indexes/work.db` |
| `--verbose / --no-verbose` | Verbose logging | `--no-verbose` | `--verbose` |
| `--debug / --no-debug` | Debug mode | `--no-debug` | `--debug` |
| `--transport TEXT` | Transport mechanism | `stdio` | `--transport streamable-http` |
| `--port INTEGER` | Port (for HTTP transports) | None | `--port 8080` |

### Transport Options

| Transport | Description | Use Case |
|-----------|-------------|----------|
| `stdio` | Standard input/output | Claude Desktop integration |
| `streamable-http` | HTTP with streaming | Web-based integrations |
| `sse` | Server-sent events | Real-time web apps |

## Claude Desktop Configuration Examples

### Basic Configuration

```json
{
  "mcpServers": {
    "oboyu": {
      "command": "oboyu",
      "args": ["mcp"],
      "description": "Search local documents"
    }
  }
}
```

### Multiple Document Collections

```json
{
  "mcpServers": {
    "oboyu-personal": {
      "command": "oboyu",
      "args": ["mcp", "--db-path", "~/indexes/personal.db"],
      "description": "Personal notes and documents"
    },
    "oboyu-work": {
      "command": "oboyu",
      "args": ["mcp", "--db-path", "~/indexes/work.db"],
      "description": "Work documents"
    }
  }
}
```

### With Environment Variables

```json
{
  "mcpServers": {
    "oboyu": {
      "command": "oboyu",
      "args": ["mcp"],
      "description": "Search local documents",
      "env": {
        "OBOYU_DB_PATH": "~/indexes/main.db"
      }
    }
  }
}
```

## Usage Examples

### Basic Document Search

In Claude Desktop:
```
Find all documents about machine learning
```

Oboyu MCP will search your indexed documents and return relevant results.

### Code Documentation Search

```
Show me documentation about authentication in my codebase
```

### Research Assistant

```
What do my notes say about quantum computing applications?
```

### Project Management

```
Find all meeting notes from last month about the new project
```

## Best Practices

### Index Organization

Create separate indices for different purposes:

```bash
# Personal knowledge base
oboyu index ~/Personal --db-path ~/indexes/personal.db

# Work documents  
oboyu index ~/Work --db-path ~/indexes/work.db

# Research papers
oboyu index ~/Research --db-path ~/indexes/research.db
```

### Regular Updates

Keep your index current:

```bash
# Update your index regularly
oboyu index ~/Documents

# Check what would be updated
oboyu manage diff
```

### Optimize for Search Quality

```bash
# Index with optimized settings
oboyu index ~/Documents --chunk-size 1024 --chunk-overlap 256
```

## Troubleshooting

### Connection Issues

```bash
# Test MCP server manually
oboyu mcp --debug --verbose

# Check if Oboyu is properly installed
oboyu version

# Verify your index exists
oboyu manage status
```

### Common Problems

**"No results found"**
- Check that documents are indexed: `oboyu manage status`
- Verify the database path is correct
- Try broader search terms

**"MCP server not starting"**
- Check Claude Desktop configuration syntax
- Verify Oboyu installation: `oboyu --help`
- Check file permissions for database path

**"Slow responses"**
- Update your index: `oboyu index ~/Documents`
- Use more specific search queries
- Consider smaller index sizes

### Debug Mode

Enable debug mode for troubleshooting:

```json
{
  "mcpServers": {
    "oboyu": {
      "command": "oboyu", 
      "args": ["mcp", "--debug", "--verbose"],
      "description": "Search local documents (debug mode)"
    }
  }
}
```

## Advanced Usage

### HTTP Transport for Web Integration

```bash
# Start HTTP MCP server
oboyu mcp --transport streamable-http --port 8080
```

Configure for web applications:
```json
{
  "mcpServers": {
    "oboyu-web": {
      "command": "oboyu",
      "args": ["mcp", "--transport", "streamable-http", "--port", "8080"],
      "description": "Web-accessible document search"
    }
  }
}
```

### Multiple Database Support

You can run multiple MCP servers for different document collections:

```json
{
  "mcpServers": {
    "oboyu-docs": {
      "command": "oboyu",
      "args": ["mcp", "--db-path", "~/indexes/docs.db"],
      "description": "Documentation search"
    },
    "oboyu-notes": {
      "command": "oboyu", 
      "args": ["mcp", "--db-path", "~/indexes/notes.db"],
      "description": "Personal notes search"
    },
    "oboyu-code": {
      "command": "oboyu",
      "args": ["mcp", "--db-path", "~/indexes/code.db"], 
      "description": "Code and technical docs"
    }
  }
}
```

## Security Considerations

### File Access

Oboyu MCP server only accesses:
- The configured database file
- Documents already indexed in that database

The server cannot:
- Access files outside the indexed collection
- Modify or delete files
- Execute commands on your system

### Network Access

When using HTTP transport:
- Server runs only on localhost by default
- No automatic internet access
- Consider firewall rules for production use

## Next Steps

- Learn about [CLI Workflows](cli-workflows.md) for managing multiple indices
- Explore [Search Patterns](../usage-examples/search-patterns.md) for better queries
- Configure [Automation](automation.md) for keeping indices updated