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

### search

Execute high-precision semantic search with Japanese language optimization and comprehensive parameter guidance.

**✨ Enhanced User Experience Features:**
- 🔍 **Search Mode Optimization Guide**: Clear guidance on when to use vector, BM25, or hybrid modes
- 💡 **Usage Examples & Best Practices**: Comprehensive examples for Japanese, English, and mixed queries
- 🎯 **Optimization Tips**: Practical advice for query formulation and parameter tuning
- ❌ **Troubleshooting Section**: Built-in guidance for common search issues
- 🌏 **Language Support Details**: Explicit Japanese/English multilingual capabilities

**Parameters:**

The `search` tool now provides comprehensive documentation and optimization guidance directly in its annotation. Key parameters include:

- `query` (string, required): Search query text (Japanese, English, or mixed supported)
  - Examples: "machine learning algorithms", "機械学習アルゴリズム", "REST API design"
- `top_k` (integer, 1-100): Number of results to return (recommended: 5-10, default: 5)
  - 1-5: Highly curated results
  - 6-15: Broader candidate pool  
  - 16-100: Comprehensive search coverage
- `mode` (string, optional): Search algorithm mode (default: "hybrid")
  - "vector": Semantic similarity focus (best for conceptual queries)
  - "bm25": Keyword matching focus (best for technical searches)  
  - "hybrid": Balanced combination (recommended for general use)
- `language` (string, optional): Language filter for results ("ja", "en", or None for all)
- `snippet_config` (object, optional): Configuration for snippet generation and context control
- `filters` (object, optional): Search filters for date range and path filtering

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

**Search Filter Options:**
- `date_range` (object, optional): Filter by document timestamps
  - `start` (string, optional): Start date in ISO format (e.g., "2024-01-01" or "2024-01-01T12:00:00")
  - `end` (string, optional): End date in ISO format
  - `field` (string, optional): Date field to filter on - "created_at" or "modified_at" (default: "modified_at")
- `path_filter` (object, optional): Filter by file path patterns
  - `include_patterns` (array, optional): List of shell-style patterns to include (e.g., ["*/docs/*", "*.md"])
  - `exclude_patterns` (array, optional): List of shell-style patterns to exclude (e.g., ["*/test/*", "*.log"])

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

## Client Integration Examples

Oboyu's MCP server can be integrated with various AI coding assistants beyond Claude Desktop. Here are setup examples for popular platforms:

### Cursor IDE

Cursor supports MCP servers through its AI features configuration:

1. Open Cursor Settings (Cmd/Ctrl + ,)
2. Navigate to "AI" → "MCP Servers"
3. Add the following configuration:

```json
{
  "oboyu": {
    "command": "oboyu",
    "args": ["mcp"],
    "env": {},
    "description": "Japanese-enhanced semantic search for your codebase"
  }
}
```

4. Restart Cursor IDE
5. Use Cursor Chat with queries like:
   - "Search my docs for authentication implementation"
   - "Find Japanese documentation about API設計"

### VS Code with Continue.dev

Continue.dev is an open-source AI coding assistant that supports MCP:

1. Install the Continue extension in VS Code
2. Open Continue settings (`~/.continue/config.json`)
3. Add Oboyu to the tools section:

```json
{
  "tools": [
    {
      "name": "oboyu",
      "type": "mcp",
      "config": {
        "command": "oboyu",
        "args": ["mcp"],
        "transport": "stdio"
      }
    }
  ]
}
```

4. Reload VS Code window
5. Access through Continue chat with @oboyu mentions

### Cody AI

Cody supports external tools through its extension API:

1. Install Cody extension in your IDE
2. Configure Cody settings file (`~/.cody/mcp-servers.json`):

```json
{
  "servers": {
    "oboyu": {
      "command": "oboyu",
      "args": ["mcp", "--db-path", "${CODY_WORKSPACE}/.oboyu/index.db"],
      "env": {
        "OBOYU_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

3. Enable MCP support in Cody settings
4. Use natural language queries in Cody chat

### Other AI Coding Assistants

For AI assistants that support MCP or stdio-based tools:

**Generic stdio configuration:**
```json
{
  "command": "oboyu",
  "args": ["mcp"],
  "transport": "stdio",
  "timeout": 30000
}
```

**HTTP-based configuration (for web integrations):**
```json
{
  "url": "http://localhost:8080",
  "transport": "sse",
  "headers": {
    "Authorization": "Bearer ${OBOYU_API_KEY}"
  }
}
```

### Platform-Specific Setup Instructions

#### macOS

1. **Install Oboyu:**
   ```bash
   # Using pip
   pip install oboyu
   
   # Using pipx (recommended for global install)
   pipx install oboyu
   
   # Using uv
   uv tool install oboyu
   ```

2. **Configure shell environment:**
   ```bash
   # Add to ~/.zshrc or ~/.bash_profile
   export OBOYU_DB_PATH="$HOME/.oboyu/index.db"
   export OBOYU_LOG_LEVEL="INFO"
   ```

3. **Set up MCP server auto-start (optional):**
   ```bash
   # Create LaunchAgent plist
   cat > ~/Library/LaunchAgents/com.oboyu.mcp.plist << EOF
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
       <key>Label</key>
       <string>com.oboyu.mcp</string>
       <key>ProgramArguments</key>
       <array>
           <string>/usr/local/bin/oboyu</string>
           <string>mcp</string>
           <string>--transport</string>
           <string>sse</string>
           <string>--port</string>
           <string>8080</string>
       </array>
       <key>RunAtLoad</key>
       <true/>
       <key>KeepAlive</key>
       <true/>
   </dict>
   </plist>
   EOF
   
   # Load the service
   launchctl load ~/Library/LaunchAgents/com.oboyu.mcp.plist
   ```

#### Linux

1. **Install Oboyu:**
   ```bash
   # Using pip in virtual environment
   python -m venv oboyu-env
   source oboyu-env/bin/activate
   pip install oboyu
   
   # System-wide with pipx
   pipx install oboyu
   ```

2. **Configure environment:**
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export OBOYU_DB_PATH="$HOME/.oboyu/index.db"
   export OBOYU_LOG_DIR="$HOME/.oboyu/logs"
   ```

3. **Create systemd service (optional):**
   ```bash
   # Create service file
   sudo tee /etc/systemd/system/oboyu-mcp.service << EOF
   [Unit]
   Description=Oboyu MCP Server
   After=network.target
   
   [Service]
   Type=simple
   User=$USER
   ExecStart=/usr/local/bin/oboyu mcp --transport sse --port 8080
   Restart=always
   RestartSec=10
   Environment="OBOYU_DB_PATH=$HOME/.oboyu/index.db"
   
   [Install]
   WantedBy=multi-user.target
   EOF
   
   # Enable and start the service
   sudo systemctl enable oboyu-mcp
   sudo systemctl start oboyu-mcp
   ```

#### Windows

1. **Install Oboyu:**
   ```powershell
   # Using pip
   pip install oboyu
   
   # Or using pipx
   pipx install oboyu
   ```

2. **Configure environment variables:**
   ```powershell
   # Set user environment variables
   [System.Environment]::SetEnvironmentVariable("OBOYU_DB_PATH", "$env:USERPROFILE\.oboyu\index.db", "User")
   [System.Environment]::SetEnvironmentVariable("OBOYU_LOG_LEVEL", "INFO", "User")
   ```

3. **Create Windows service (optional):**
   ```powershell
   # Using NSSM (Non-Sucking Service Manager)
   # First install NSSM: choco install nssm
   
   nssm install oboyu-mcp "C:\Python311\Scripts\oboyu.exe" "mcp --transport sse --port 8080"
   nssm set oboyu-mcp AppDirectory "$env:USERPROFILE"
   nssm set oboyu-mcp DisplayName "Oboyu MCP Server"
   nssm set oboyu-mcp Description "Japanese-enhanced semantic search MCP server"
   nssm start oboyu-mcp
   ```

### Docker Setup

For containerized deployments:

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    && rm -rf /var/lib/apt/lists/*

# Install Oboyu
RUN pip install oboyu

# Create data directory
RUN mkdir -p /data

# Expose MCP port
EXPOSE 8080

# Run MCP server
CMD ["oboyu", "mcp", "--transport", "sse", "--port", "8080", "--db-path", "/data/index.db"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  oboyu-mcp:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./data:/data
      - ./documents:/documents:ro
    environment:
      - OBOYU_LOG_LEVEL=INFO
    restart: unless-stopped
```

## Configuration Examples

### Development Environment

Complete MCP configuration for a development setup:

```json
{
  "servers": {
    "oboyu-main": {
      "command": "oboyu",
      "args": ["mcp", "--db-path", "${HOME}/projects/main/.oboyu/index.db"],
      "env": {
        "OBOYU_LOG_LEVEL": "DEBUG",
        "OBOYU_CACHE_DIR": "${HOME}/.cache/oboyu"
      },
      "description": "Main project documentation"
    },
    "oboyu-docs": {
      "command": "oboyu",
      "args": ["mcp", "--db-path", "${HOME}/docs/.oboyu/index.db"],
      "env": {
        "OBOYU_LOG_LEVEL": "INFO"
      },
      "description": "Personal documentation and notes"
    }
  }
}
```

### Team Environment

Shared configuration for team usage:

```json
{
  "servers": {
    "oboyu-shared": {
      "command": "oboyu",
      "args": ["mcp", "--transport", "sse", "--port", "8080"],
      "env": {
        "OBOYU_DB_PATH": "/shared/knowledge-base/index.db",
        "OBOYU_READONLY": "true",
        "OBOYU_AUTH_TOKEN": "${TEAM_AUTH_TOKEN}"
      },
      "description": "Team knowledge base"
    }
  }
}
```

### Multi-Repository Setup

Configuration for searching across multiple repositories:

```json
{
  "servers": {
    "oboyu-frontend": {
      "command": "oboyu",
      "args": ["mcp", "--db-path", "${WORKSPACE}/frontend/.oboyu/index.db"],
      "description": "Frontend repository search"
    },
    "oboyu-backend": {
      "command": "oboyu",
      "args": ["mcp", "--db-path", "${WORKSPACE}/backend/.oboyu/index.db"],
      "description": "Backend repository search"
    },
    "oboyu-docs": {
      "command": "oboyu",
      "args": ["mcp", "--db-path", "${WORKSPACE}/documentation/.oboyu/index.db"],
      "description": "Documentation repository search"
    }
  }
}
```

### Advanced Configuration with Custom Models

```json
{
  "servers": {
    "oboyu-custom": {
      "command": "oboyu",
      "args": [
        "mcp",
        "--db-path", "${PROJECT_ROOT}/.oboyu/index.db"
      ],
      "env": {
        "OBOYU_EMBEDDING_MODEL": "intfloat/multilingual-e5-large",
        "OBOYU_RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
        "OBOYU_CACHE_SIZE": "2048",
        "OBOYU_MAX_WORKERS": "8",
        "HF_HOME": "${HOME}/.cache/huggingface"
      }
    }
  }
}
```

## Security Considerations

When deploying Oboyu MCP server, consider these security aspects:

### Authentication and Authorization

1. **Token-based authentication for HTTP transports:**
   ```bash
   # Generate secure token
   export OBOYU_AUTH_TOKEN=$(openssl rand -hex 32)
   
   # Start server with authentication
   oboyu mcp --transport sse --port 8080 --auth-token $OBOYU_AUTH_TOKEN
   ```

2. **Client configuration with auth:**
   ```json
   {
     "servers": {
       "oboyu-secure": {
         "command": "oboyu",
         "args": ["mcp"],
         "env": {
           "OBOYU_AUTH_TOKEN": "${SECRET_AUTH_TOKEN}"
         }
       }
     }
   }
   ```

### Network Security

1. **Use TLS for HTTP transports:**
   ```bash
   # With self-signed certificate
   oboyu mcp --transport sse --port 8443 \
     --tls-cert /path/to/cert.pem \
     --tls-key /path/to/key.pem
   ```

2. **Restrict network access:**
   ```bash
   # Bind to localhost only
   oboyu mcp --transport sse --port 8080 --host 127.0.0.1
   
   # Use firewall rules
   sudo ufw allow from 192.168.1.0/24 to any port 8080
   ```

### Data Protection

1. **Read-only mode for shared deployments:**
   ```bash
   export OBOYU_READONLY=true
   oboyu mcp --db-path /shared/index.db
   ```

2. **Encrypt database at rest:**
   ```bash
   # Use encrypted filesystem
   # On macOS
   hdiutil create -size 10g -fs APFS -encryption -volname "OboyuData" oboyu-encrypted.dmg
   
   # On Linux
   cryptsetup luksFormat /dev/sdb1
   cryptsetup open /dev/sdb1 oboyu-data
   mkfs.ext4 /dev/mapper/oboyu-data
   ```

3. **Audit logging:**
   ```json
   {
     "env": {
       "OBOYU_AUDIT_LOG": "/var/log/oboyu/audit.log",
       "OBOYU_LOG_QUERIES": "true",
       "OBOYU_LOG_RESULTS": "false"
     }
   }
   ```

### Access Control

1. **Path-based restrictions:**
   ```json
   {
     "env": {
       "OBOYU_ALLOWED_PATHS": "/docs,/public",
       "OBOYU_DENIED_PATHS": "/private,/secrets"
     }
   }
   ```

2. **Query filtering:**
   ```json
   {
     "env": {
       "OBOYU_MAX_RESULTS": "50",
       "OBOYU_QUERY_TIMEOUT": "30",
       "OBOYU_BLOCK_PATTERNS": "password,secret,token"
     }
   }
   ```

### Best Practices

1. **Principle of Least Privilege:**
   - Run MCP server with minimal required permissions
   - Use read-only database access when possible
   - Restrict file system access to indexed directories

2. **Regular Security Updates:**
   ```bash
   # Keep Oboyu updated
   pipx upgrade oboyu
   
   # Monitor for vulnerabilities
   pip-audit
   ```

3. **Monitoring and Alerting:**
   - Set up logging for authentication failures
   - Monitor resource usage for DoS prevention
   - Alert on unusual query patterns

4. **Secure Configuration Management:**
   - Store sensitive configuration in environment variables
   - Use secret management tools (e.g., HashiCorp Vault)
   - Rotate authentication tokens regularly

## Advanced MCP Usage Patterns

### Contextual Search Workflows

1. **Progressive Refinement Pattern:**
   ```python
   # Start with broad search
   results = search("authentication", top_k=20, mode="hybrid")
   
   # Refine with specific mode based on initial results
   if needs_exact_match(results):
       refined = search("OAuth2 implementation", mode="bm25", top_k=5)
   else:
       refined = search("authentication concepts", mode="vector", top_k=10)
   ```

2. **Multi-Stage Search Pipeline:**
   ```python
   # Stage 1: Broad conceptual search
   concepts = search("security patterns", mode="vector", top_k=30)
   
   # Stage 2: Filter by date and path
   recent = search("security patterns", 
                  filters={
                      "date_range": {"start": "2024-01-01"},
                      "path_filter": {"include_patterns": ["*/security/*"]}
                  })
   
   # Stage 3: Deep dive with snippets
   detailed = search("JWT implementation",
                    snippet_config={
                        "levels": [
                            {"type": "overview", "length": 100},
                            {"type": "code", "length": 500}
                        ]
                    })
   ```

### Language-Aware Search Strategies

1. **Mixed Language Queries:**
   ```python
   # Search with mixed Japanese/English
   results = search("REST API設計のベストプラクティス", 
                   mode="hybrid",
                   snippet_config={"japanese_aware": true})
   ```

2. **Language-Specific Optimization:**
   ```python
   # Japanese technical documentation
   jp_results = search("システムアーキテクチャ",
                      language="ja",
                      mode="bm25",  # Better for Japanese exact matches
                      snippet_config={
                          "strategy": "sentence_boundary",
                          "japanese_aware": true
                      })
   
   # English conceptual search
   en_results = search("system architecture patterns",
                      language="en", 
                      mode="vector",  # Better for conceptual English
                      snippet_config={
                          "strategy": "paragraph_boundary"
                      })
   ```

### Custom Integration Patterns

1. **Batch Search Operations:**
   ```python
   # Search multiple related topics
   topics = ["authentication", "authorization", "security tokens"]
   results = {}
   
   for topic in topics:
       results[topic] = search(topic, 
                              top_k=5,
                              snippet_config={"length": 150})
   ```

2. **Search Result Aggregation:**
   ```python
   # Combine results from different search modes
   vector_results = search(query, mode="vector", top_k=10)
   bm25_results = search(query, mode="bm25", top_k=10)
   
   # Custom merge logic based on score distribution
   combined = merge_results(vector_results, bm25_results, 
                           strategy="score_weighted")
   ```

3. **Dynamic Parameter Adjustment:**
   ```python
   # Adjust search parameters based on query characteristics
   def adaptive_search(query):
       # Short queries: prefer BM25
       if len(query.split()) <= 3:
           return search(query, mode="bm25", top_k=10)
       
       # Japanese queries: use specific configuration
       if contains_japanese(query):
           return search(query, mode="hybrid",
                        snippet_config={"japanese_aware": true})
       
       # Long English queries: use vector search
       return search(query, mode="vector", top_k=15)
   ```

### Performance Optimization Patterns

1. **Caching Strategy:**
   ```json
   {
     "env": {
       "OBOYU_CACHE_ENABLED": "true",
       "OBOYU_CACHE_SIZE": "1024",
       "OBOYU_CACHE_TTL": "3600"
     }
   }
   ```

2. **Parallel Search Execution:**
   ```python
   # Execute multiple searches in parallel
   from concurrent.futures import ThreadPoolExecutor
   
   queries = ["design patterns", "アーキテクチャ", "best practices"]
   
   with ThreadPoolExecutor(max_workers=3) as executor:
       futures = [executor.submit(search, q, top_k=5) for q in queries]
       results = [f.result() for f in futures]
   ```

3. **Resource Management:**
   ```json
   {
     "env": {
       "OBOYU_MAX_CONNECTIONS": "10",
       "OBOYU_CONNECTION_TIMEOUT": "30",
       "OBOYU_QUERY_TIMEOUT": "60",
       "OBOYU_MAX_MEMORY": "2048"
     }
   }
   ```

## Search Examples

The MCP server provides flexible search capabilities through AI assistants:

### Hybrid Search (Default)
```
# Through MCP tools in Claude or other AI assistants
search("機械学習の基本的なアルゴリズム", top_k=5)
```

### Vector Search
```
search("design patterns in software architecture", mode="vector", top_k=10)
```

### BM25 Search
```
search("データベース正規化", mode="bm25")
```

### Language Filtering
```
search("機械学習アルゴリズム", language="ja", top_k=10)
```

### Snippet Context Control
```
# Basic snippet configuration
search("機械学習の原則", 
       snippet_config={
         "length": 200,
         "highlight_matches": true,
         "strategy": "sentence_boundary"
       })

# Japanese-aware snippet processing
search("システム設計の考え方", 
       snippet_config={
         "length": 150,
         "japanese_aware": true,
         "prefer_complete_sentences": true,
         "context_window": 30
       })

# Multi-level snippets
search("database design patterns", 
       snippet_config={
         "levels": [
           {"type": "summary", "length": 100},
           {"type": "detailed", "length": 300}
         ],
         "highlight_matches": false
       })
```

### Search Filtering

The MCP server now supports advanced filtering to narrow down search results:

#### Date Range Filtering
```
# Find recent documentation updates
search("システム設計", 
       filters={
         "date_range": {
           "start": "2024-05-01",
           "field": "modified_at"
         }
       })

# Find documents created in a specific time period
search("機械学習アルゴリズム", 
       filters={
         "date_range": {
           "start": "2024-01-01",
           "end": "2024-12-31",
           "field": "created_at"
         }
       })
```

#### Path Pattern Filtering
```
# Search only in documentation directories
search("API documentation", 
       filters={
         "path_filter": {
           "include_patterns": ["*/docs/*", "*/api/*", "*.md"],
           "exclude_patterns": ["*/test/*", "*.log"]
         }
       })

# Focus search on specific project areas
search("API implementation", 
       filters={
         "path_filter": {
           "include_patterns": ["*/backend/*", "*/api/*"],
           "exclude_patterns": ["*/test/*", "*/deprecated/*"]
         }
       })
```

#### Combined Filtering
```
# Search recent documentation in specific directories
search("設計パターン", 
       filters={
         "date_range": {
           "start": "2024-06-01",
           "field": "modified_at"
         },
         "path_filter": {
           "include_patterns": ["*/documentation/*"],
           "exclude_patterns": ["*/archived/*"]
         }
       })
```

#### Pattern Matching Rules
- **Wildcard patterns**: Use `*` for any characters, `?` for single character
- **Directory matching**: `*/docs/*` matches any file in a docs directory at any level
- **File extension**: `*.md` matches all Markdown files
- **Case sensitivity**: Pattern matching is case-insensitive for better usability
- **Multiple patterns**: Include patterns are OR-ed together, exclude patterns are applied after includes

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

**Search Filtering:**
- "Find recent changes to the API documentation"
  → Uses date range filtering with modified_at field + path filtering for docs
- "Show me configuration examples, but exclude test files"
  → Uses path filtering to include config files and exclude test directories
- "What documentation was created this year?"
  → Uses date range filtering with created_at field for temporal search
- "Search for database patterns only in the backend code"
  → Uses path filtering to focus on specific project areas

## Performance Considerations

- The MCP server maintains a persistent connection to the database for fast responses
- Search operations typically complete in 50-200ms depending on index size
- Japanese text processing is optimized with MeCab tokenization caching
- Vector similarity search uses HNSW index for efficient approximate nearest neighbor lookups
- Hybrid search runs vector and BM25 searches in parallel for optimal performance

## Troubleshooting

### Common Issues and Solutions

#### MCP Server Startup Errors

**Error: "Database not found at path"**
```
Error: Database file not found: /home/user/.oboyu/index.db
```
**Solution:**
- Create an index first: `oboyu index <directory>`
- Verify the database path: `ls -la ~/.oboyu/index.db`
- Use explicit path: `oboyu mcp --db-path /path/to/existing.db`

**Error: "Port already in use"**
```
Error: Address already in use: 0.0.0.0:8080
```
**Solution:**
- Check for existing processes: `lsof -i :8080` (macOS/Linux) or `netstat -ano | findstr :8080` (Windows)
- Kill the process: `kill -9 <PID>`
- Use a different port: `oboyu mcp --transport sse --port 8081`

**Error: "Permission denied"**
```
PermissionError: [Errno 13] Permission denied: '/var/oboyu/index.db'
```
**Solution:**
- Check file permissions: `ls -la /var/oboyu/index.db`
- Fix ownership: `sudo chown $USER:$USER /var/oboyu/index.db`
- Use a user-writable location: `oboyu mcp --db-path ~/oboyu-data/index.db`

#### Search and Query Issues

**Error: "No documents in index"**
```
Warning: Index contains 0 documents
```
**Solution:**
- Verify index status: `oboyu index manage status`
- Re-index if needed: `oboyu index <directory> --force`
- Check for indexing errors in logs: `tail -n 100 ~/.oboyu/logs/indexing.log`

**Error: "Model not found"**
```
Error: Could not load embedding model: intfloat/multilingual-e5-base
```
**Solution:**
- Let the model download: First run may take time to download models
- Check internet connection for Hugging Face access
- Clear cache and retry: `rm -rf ~/.cache/huggingface && oboyu mcp`
- Use offline mode with pre-downloaded models

**Error: "Out of memory"**
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Use CPU instead: `export OBOYU_DEVICE=cpu`
- Reduce batch size: `export OBOYU_BATCH_SIZE=8`
- Use smaller model: `export OBOYU_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`

#### Connection and Integration Issues

**Error: "MCP handshake failed"**
```
Error: MCP protocol handshake failed: Invalid message format
```
**Solution:**
- Verify MCP server is running: `ps aux | grep "oboyu mcp"`
- Check transport type matches client expectation
- Enable debug logging: `oboyu mcp --debug`
- Ensure client supports MCP protocol version

**Error: "Authentication failed"**
```
Error: Authentication failed: Invalid or missing token
```
**Solution:**
- Set auth token: `export OBOYU_AUTH_TOKEN=your-secret-token`
- Verify token in client config matches server
- Check token format (no extra spaces or quotes)
- Regenerate token if compromised

**Error: "Timeout waiting for response"**
```
Error: Request timeout after 30 seconds
```
**Solution:**
- Increase timeout in client: `"timeout": 60000`
- Check server performance: `oboyu mcp --debug`
- Optimize query: Use fewer results (`top_k=5`)
- Check network latency if using remote server

#### Platform-Specific Issues

**macOS: "Library not loaded: libmecab.dylib"**
```
Error: dlopen(libmecab.dylib): Library not loaded
```
**Solution:**
```bash
# Install MeCab
brew install mecab mecab-ipadic

# Set library path
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
```

**Linux: "MeCab dictionary not found"**
```
Error: MeCab dictionary not found at /usr/lib/mecab/dic/mecab-ipadic-neologd
```
**Solution:**
```bash
# Install dictionaries
sudo apt-get install mecab-ipadic-utf8

# Or download manually
git clone https://github.com/neologd/mecab-ipadic-neologd.git
cd mecab-ipadic-neologd
./bin/install-mecab-ipadic-neologd -n
```

**Windows: "DLL load failed"**
```
Error: DLL load failed while importing MeCab
```
**Solution:**
- Install Visual C++ Redistributable
- Use WSL2 for better compatibility
- Or use Docker container approach

#### Performance Issues

**Slow search responses (>5 seconds)**
**Solution:**
- Check index optimization: `oboyu index manage optimize`
- Reduce result count: Use `top_k=5` instead of `top_k=50`
- Enable caching: `export OBOYU_CACHE_ENABLED=true`
- Monitor resource usage: `htop` or `top` during searches

**High memory usage**
**Solution:**
- Limit cache size: `export OBOYU_CACHE_SIZE=512`
- Use smaller models for limited resources
- Enable memory mapping: `export OBOYU_USE_MMAP=true`
- Set max memory limit: `export OBOYU_MAX_MEMORY=2048`

### Debug Mode

Enable comprehensive debug logging:

```bash
# Maximum verbosity
oboyu mcp --debug

# With specific log file
OBOYU_LOG_FILE=/tmp/oboyu-debug.log oboyu mcp --debug

# Debug specific components
export OBOYU_DEBUG_COMPONENTS="mcp,search,embedding"
oboyu mcp --debug
```

Debug output includes:
- MCP protocol messages (JSON-RPC requests/responses)
- Search query parsing and transformation
- Embedding generation timings
- Database query execution plans
- Memory usage statistics
- Cache hit/miss rates

### Diagnostic Commands

**Check system compatibility:**
```bash
# Verify Oboyu installation
oboyu --version

# Test MeCab installation
echo "テスト" | mecab

# Check Python dependencies
pip show oboyu

# Verify database integrity
oboyu index manage check
```

**Monitor MCP server health:**
```bash
# Check if server is responding (stdio mode)
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | oboyu mcp

# Test search functionality
oboyu query "test" --db-path ~/.oboyu/index.db

# View server metrics (if enabled)
curl http://localhost:8080/metrics
```

### Getting Help

If issues persist:

1. **Collect diagnostic information:**
   ```bash
   oboyu index manage status --verbose > oboyu-diagnostics.txt
   oboyu --version >> oboyu-diagnostics.txt
   python --version >> oboyu-diagnostics.txt
   pip freeze | grep oboyu >> oboyu-diagnostics.txt
   ```

2. **Enable trace logging:**
   ```bash
   export OBOYU_LOG_LEVEL=TRACE
   export OBOYU_LOG_FILE=oboyu-trace.log
   oboyu mcp --debug
   ```

3. **Report issues with:**
   - Error messages and stack traces
   - Steps to reproduce
   - System information (OS, Python version)
   - Diagnostic output
   - Relevant configuration files

## Implementation Details

For advanced use cases, you can customize the MCP server by modifying the source code in the `src/oboyu/mcp/` directory:

- `server.py`: Main server implementation with tool definitions
- `context.py`: Search context management and result formatting