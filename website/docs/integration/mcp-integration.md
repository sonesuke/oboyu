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

### 1. Install Oboyu with MCP Support

```bash
# Install Oboyu
pip install oboyu

# Verify MCP support
oboyu mcp --version
```

### 2. Configure Claude Desktop

Add Oboyu to your Claude Desktop configuration:

```json
{
  "mcp-servers": {
    "oboyu": {
      "command": "oboyu",
      "args": ["mcp", "serve", "--index", "~/.oboyu/index.db"],
      "description": "Search local documents"
    }
  }
}
```

### 3. Start Using

In Claude Desktop, you can now:
```
Search my documents for information about project deadlines
```

Claude will use Oboyu to search your indexed documents and provide relevant information.

## Detailed Setup

### Creating an Index for MCP

```bash
# Index your documents
oboyu index ~/Documents --name main

# Optimize for MCP usage
oboyu index ~/Documents \
  --chunk-size 1024 \
  --use-reranker \
  --name mcp-index
```

### MCP Server Configuration

```yaml
# ~/.config/oboyu/mcp-config.yaml
mcp:
  index_path: "~/.oboyu/mcp-index.db"
  max_results: 10
  snippet_length: 200
  enable_highlighting: true
  
  # Security settings
  allowed_paths:
    - "~/Documents"
    - "~/Notes"
  
  # Performance settings
  cache_queries: true
  timeout: 30
```

### Starting MCP Server

```bash
# Start with default settings
oboyu mcp serve

# Start with custom config
oboyu mcp serve --config ~/.config/oboyu/mcp-config.yaml

# Start with specific index
oboyu mcp serve --index ~/research/index.db

# Debug mode
oboyu mcp serve --debug
```

## Claude Desktop Configuration

### Basic Configuration

Edit Claude Desktop settings:

**macOS**: `~/Library/Application Support/Claude/config.json`
**Windows**: `%APPDATA%\Claude\config.json`
**Linux**: `~/.config/claude/config.json`

```json
{
  "mcp-servers": {
    "oboyu": {
      "command": "oboyu",
      "args": ["mcp", "serve"],
      "description": "Search local documents",
      "env": {
        "OBOYU_INDEX_PATH": "/path/to/index.db"
      }
    }
  }
}
```

### Advanced Configuration

```json
{
  "mcp-servers": {
    "oboyu-personal": {
      "command": "oboyu",
      "args": [
        "mcp", 
        "serve",
        "--index", "~/.oboyu/personal.db",
        "--max-results", "20"
      ],
      "description": "Personal notes and documents"
    },
    "oboyu-work": {
      "command": "oboyu",
      "args": [
        "mcp",
        "serve", 
        "--index", "~/.oboyu/work.db",
        "--filter", "path:~/work/**"
      ],
      "description": "Work documents"
    }
  }
}
```

## Usage Examples

### Basic Document Search

In Claude Desktop:
```
Find all documents about machine learning from the past month
```

Oboyu MCP will:
1. Search for "machine learning"
2. Filter by date (past month)
3. Return relevant documents
4. Claude provides summary

### Code Documentation Search

```
Show me the API documentation for the authentication module
```

Oboyu searches for:
- Files containing "API" and "authentication"
- Documentation files (*.md, *.rst)
- Returns code examples and descriptions

### Research Assistant

```
What do my notes say about quantum computing applications?
```

Oboyu:
- Searches notes for "quantum computing"
- Focuses on application mentions
- Returns relevant passages
- Claude synthesizes information

## MCP Features

### Search Capabilities

```python
# MCP exposes these search methods
tools = [
    {
        "name": "search",
        "description": "Search documents",
        "parameters": {
            "query": "search terms",
            "mode": "hybrid|vector|bm25",
            "limit": 10,
            "filters": {
                "file_type": ["md", "txt"],
                "date_range": "7d"
            }
        }
    },
    {
        "name": "get_document",
        "description": "Retrieve full document",
        "parameters": {
            "path": "/path/to/file"
        }
    }
]
```

### Context Window Management

```yaml
# Configure context limits
mcp:
  max_context_length: 8000  # tokens
  truncation_strategy: "middle"  # start|middle|end
  prioritize_relevance: true
```

### Security Features

```yaml
mcp:
  # Path restrictions
  allowed_paths:
    - "~/Documents"
    - "~/Notes"
  denied_paths:
    - "~/Private"
    - "*.key"
    - "*.secret"
  
  # Content filtering
  filter_sensitive: true
  redact_patterns:
    - "ssn:\s*\d{3}-\d{2}-\d{4}"
    - "api[_-]?key:\s*\w+"
```

## Advanced Integration

### Custom MCP Handlers

```python
# Create custom MCP tool
from oboyu.mcp import MCPTool

class CustomSearchTool(MCPTool):
    def execute(self, query, **kwargs):
        # Custom search logic
        results = self.search_with_context(query)
        return self.format_for_claude(results)

# Register with Oboyu
oboyu mcp register-tool CustomSearchTool
```

### Workflow Integration

```bash
# Pre-process documents for MCP
oboyu mcp prepare ~/Documents \
  --extract-summaries \
  --generate-metadata

# Create specialized indices
oboyu mcp create-index \
  --type "technical-docs" \
  --optimize-for "question-answering"
```

### Multi-Index Support

```json
{
  "mcp-servers": {
    "oboyu-multi": {
      "command": "oboyu",
      "args": [
        "mcp",
        "serve",
        "--indices", 
        "personal:~/.oboyu/personal.db",
        "work:~/.oboyu/work.db",
        "research:~/.oboyu/research.db"
      ]
    }
  }
}
```

## Best Practices

### Index Organization

1. **Separate Indices by Purpose**
```bash
# Personal knowledge base
oboyu index ~/Personal --name personal-mcp

# Work documents
oboyu index ~/Work --name work-mcp

# Research papers
oboyu index ~/Research --name research-mcp
```

2. **Optimize for Q&A**
```bash
oboyu index ~/Documents \
  --chunk-size 512 \
  --overlap 128 \
  --extract-qa-pairs
```

### Performance Optimization

```yaml
mcp:
  # Cache configuration
  cache:
    enabled: true
    size: 1000
    ttl: 3600
  
  # Preload common queries
  preload_queries:
    - "meeting notes"
    - "project status"
    - "documentation"
  
  # Background indexing
  auto_update: true
  update_interval: 3600
```

### Security Best Practices

1. **Limit Access**
```yaml
mcp:
  allowed_paths:
    - "~/Documents/Shared"
  denied_patterns:
    - "*.password"
    - "*.key"
    - "**/private/**"
```

2. **Audit Logging**
```bash
# Enable MCP audit log
oboyu mcp serve --audit-log ~/.oboyu/mcp-audit.log

# Review access
oboyu mcp audit-report --days 7
```

## Troubleshooting

### Connection Issues

```bash
# Test MCP server
oboyu mcp test

# Check server status
oboyu mcp status

# Debug mode
oboyu mcp serve --debug --verbose
```

### Performance Issues

```bash
# Optimize index for MCP
oboyu mcp optimize --index personal

# Monitor performance
oboyu mcp monitor --metrics

# Adjust settings
oboyu mcp config set max_results 5
oboyu mcp config set timeout 60
```

### Common Problems

**"No results found"**
- Check index is up-to-date
- Verify paths are allowed
- Try broader search terms

**"Timeout errors"**
- Reduce max_results
- Optimize index
- Increase timeout setting

**"Permission denied"**
- Check allowed_paths configuration
- Verify file permissions
- Review security settings

## MCP Command Reference

```bash
# Server management
oboyu mcp serve           # Start MCP server
oboyu mcp stop           # Stop MCP server
oboyu mcp restart        # Restart server
oboyu mcp status         # Check server status

# Configuration
oboyu mcp config show    # Show configuration
oboyu mcp config set     # Set config value
oboyu mcp config reset   # Reset to defaults

# Testing
oboyu mcp test           # Test connection
oboyu mcp query-test     # Test search functionality
oboyu mcp benchmark      # Performance test

# Maintenance
oboyu mcp optimize       # Optimize for MCP usage
oboyu mcp clean-cache    # Clear query cache
oboyu mcp update-index   # Update search index
```

## Integration Examples

### Personal Assistant
```json
{
  "mcp-servers": {
    "personal-assistant": {
      "command": "oboyu",
      "args": ["mcp", "serve", "--profile", "assistant"],
      "env": {
        "OBOYU_ASSISTANT_MODE": "true"
      }
    }
  }
}
```

### Research Helper
```json
{
  "mcp-servers": {
    "research-helper": {
      "command": "oboyu",
      "args": [
        "mcp", 
        "serve",
        "--index", "~/.oboyu/papers.db",
        "--enable-citations",
        "--extract-methodology"
      ]
    }
  }
}
```

### Code Documentation
```json
{
  "mcp-servers": {
    "code-docs": {
      "command": "oboyu",
      "args": [
        "mcp",
        "serve", 
        "--index", "~/.oboyu/code.db",
        "--language-aware",
        "--include-symbols"
      ]
    }
  }
}
```

## Next Steps

- Set up [CLI Workflows](cli-workflows.md) for index management
- Configure [Automation](automation.md) for automatic updates
- Explore [Search Patterns](../usage-examples/search-patterns.md) for better queries