# Oboyu Query Engine Architecture

## Overview

The Query Engine is the user-facing component of Oboyu, handling search requests, executing queries against the indexes, and delivering relevant results. It supports multiple search modes and provides specialized processing for Japanese queries.

## Design Goals

- Process search queries with language-appropriate handling
- Support multiple search modes (vector, BM25, hybrid)
- Deliver relevant results with appropriate ranking
- Provide effective snippet generation for context
- Offer flexible integration through CLI and MCP interfaces

## Component Structure

### Query Processor

The Query Processor handles:

- Query analysis and normalization
- Language detection for query text
- Japanese-specific query processing
- Query expansion and refinement

```python
def process_query(query_text):
    # Implementation details
    # Returns processed query ready for search
```

### Search Engine

The Search Engine implements multiple search strategies:

- Vector search using HNSW index
- BM25 search using full-text index
- Hybrid search combining both approaches
- Result ranking and scoring

```python
def search(query, mode="hybrid", **options):
    # Implementation details
    # Returns ranked search results
```

### Result Formatter

The Result Formatter prepares search results for presentation:

- Formats documents with metadata
- Generates contextual snippets
- Highlights matching terms
- Provides relevance explanations

```python
def format_results(results, query):
    # Implementation details
    # Returns formatted results for presentation
```

### MCP Server

The MCP Server provides integration with external tools:

- Implements stdio-based MCP protocol
- Handles JSON request/response formatting
- Manages persistent connection state
- Provides debugging and error handling

```python
def start_mcp_server():
    # Implementation details
    # Runs the MCP server loop
```

## Search Modes

### Vector Search

Vector search performs semantic similarity matching:

- Converts query to vector representation
- Performs approximate nearest neighbor search via HNSW
- Ranks results by vector similarity
- Excels at understanding conceptual meaning

```python
def vector_search(query_text, model="default", top_k=5):
    # Implementation details
    # Returns vector search results
```

### BM25 Search

BM25 search performs keyword-based matching:

- Tokenizes query with language-appropriate processing
- Executes BM25 ranking algorithm
- Scores documents based on term frequency and importance
- Excels at keyword matching and specific terms

```python
def bm25_search(query_text, top_k=5):
    # Implementation details
    # Returns BM25 search results
```

### Hybrid Search

Hybrid search combines the strengths of both approaches:

- Executes both vector and BM25 searches
- Normalizes and combines relevance scores
- Applies configurable weighting between approaches
- Delivers superior results for complex queries

```python
def hybrid_search(query_text, vector_weight=0.7, bm25_weight=0.3, top_k=5):
    # Implementation details
    # Returns combined search results
```

## Japanese Query Support

The Query Engine provides specialized processing for Japanese queries:

- Japanese-aware tokenization
- Character normalization
- Handling of query variations
- Mixed Japanese-English query support

```python
def process_japanese_query(query_text):
    # Implementation details
    # Returns processed Japanese query
```

## Data Flow

1. Receive query from user via CLI or MCP
2. Process query with language-specific handling
3. Execute search using appropriate mode
4. Rank and format results
5. Return formatted results to user

## Configuration Options

The Query Engine is configured through the following settings in `config.yaml`:

```yaml
query:
  default_mode: "hybrid"         # Default search mode
  vector_weight: 0.7             # Weight for vector scores in hybrid search
  bm25_weight: 0.3               # Weight for BM25 scores in hybrid search
  top_k: 5                       # Number of results to return
  snippet_length: 160            # Character length for snippets
  highlight_matches: true        # Whether to highlight matching terms
```

## Command Line Interface

The Query Engine provides a command-line interface:

```bash
# Basic query using default settings
oboyu query "システムの設計原則について教えてください"

# Specify search mode and options
oboyu query --mode vector --top-k 10 "What are the key concepts?"

# Get detailed result information
oboyu query --explain "important design principles"
```

## MCP Protocol

The MCP server mode follows a simple JSON-based protocol:

```json
// Request
{
  "type": "query",
  "query": "ドキュメント内の重要な概念は何ですか？",
  "mode": "hybrid",
  "options": {
    "top_k": 5,
    "vector_weight": 0.7,
    "bm25_weight": 0.3
  }
}

// Response
{
  "results": [
    {
      "id": "doc123",
      "title": "システム設計の基本原則",
      "snippet": "...当システムの設計は「シンプルさ」「モジュール性」「拡張性」の三つの原則に基づいています...",
      "uri": "file:///projects/docs/設計/principles.md",
      "score": 0.91
    },
    // More results...
  ],
  "stats": {
    "time_ms": 120,
    "total_matches": 28
  }
}
```

## Performance Considerations

- Implements caching for frequent queries
- Optimizes vector operations for speed
- Uses parallel processing where appropriate
- Balances accuracy and performance based on configuration

## Integration with Other Components

- Accesses indexed data via the database created by the Indexer
- Maintains clean separation from indexing and crawling processes
- Provides well-defined interfaces for external integration