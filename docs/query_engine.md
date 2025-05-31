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

Oboyu supports three distinct search modes, each optimized for different types of queries:

### Vector Search

Vector search performs semantic similarity matching using embeddings:

- Converts query to high-dimensional vector representation
- Performs approximate nearest neighbor search via HNSW index
- Ranks results by cosine similarity
- **Best for**: conceptual queries, semantic understanding, synonyms

**Example CLI usage:**
```bash
# Search for conceptually similar content
oboyu query --mode vector "機械学習の基本概念"
oboyu query --mode vector "What are design patterns?"
```

**When to use:**
- Queries about concepts rather than specific terms
- When looking for semantically related content
- Cross-lingual or synonym matching needs

### BM25 Search

BM25 search performs keyword-based matching with term frequency analysis:

- Tokenizes query using language-appropriate processing (MeCab for Japanese)
- Executes BM25 ranking algorithm (k1=1.2, b=0.75)
- Scores documents based on term frequency and document length normalization
- **Best for**: exact keyword matching, specific terminology, precise queries

**Example CLI usage:**
```bash
# Search for specific keywords
oboyu query --mode bm25 "データベース設計"
oboyu query --mode bm25 "REST API implementation"
```

**When to use:**
- Looking for specific terms or keywords
- Technical documentation with precise terminology
- When exact word matching is important

### Hybrid Search (Default)

Hybrid search combines both approaches for optimal results by leveraging the strengths of both vector and BM25 search:

- **Parallel Execution**: Executes both vector and BM25 searches simultaneously for efficiency
- **Score Normalization**: Uses min-max normalization to ensure fair combination of different scoring systems
- **Configurable Weights**: Combines results with adjustable weights (default: 70% vector, 30% BM25)
- **Result Fusion**: Merges and re-ranks results from both methods to provide comprehensive coverage
- **Best for**: most general queries, balanced precision and recall, complex information needs

**How Hybrid Search Works:**

1. **Query Processing**: The same query is processed for both search methods
2. **Parallel Search**: Vector and BM25 searches execute simultaneously
3. **Score Normalization**: Each method's scores are normalized to 0-1 range using min-max scaling
4. **Weight Application**: Normalized scores are multiplied by their respective weights
5. **Result Combination**: Final score = (vector_score × vector_weight) + (bm25_score × bm25_weight)
6. **Ranking**: Results are sorted by combined score and top-k selected

**Weight Configuration Strategies:**

```bash
# Default hybrid search (recommended for most use cases)
oboyu query "Pythonでの非同期処理の実装方法"

# Semantic-focused: Better for conceptual queries
oboyu query --vector-weight 0.8 --bm25-weight 0.2 "database optimization techniques"

# Keyword-focused: Better for specific term searches
oboyu query --vector-weight 0.3 --bm25-weight 0.7 "REST API status codes"

# Balanced approach: Equal weight to both methods
oboyu query --vector-weight 0.5 --bm25-weight 0.5 "システム設計の原則"
```

**Interactive Weight Tuning:**
```bash
# Start interactive session for weight experimentation
oboyu query --interactive --mode hybrid

> /weights 0.8 0.2
✅ Weights changed to: Vector=0.8, BM25=0.2

> machine learning algorithms
# See results with semantic focus

> /weights 0.3 0.7
✅ Weights changed to: Vector=0.3, BM25=0.7

> machine learning algorithms
# See results with keyword focus
```

**When to use different weight configurations:**

- **Vector-heavy (0.8/0.2)**: Conceptual queries, cross-language search, synonym matching
- **Balanced (0.5/0.5)**: Mixed queries with both semantic and keyword requirements
- **BM25-heavy (0.2/0.8)**: Precise technical terms, specific API names, exact matches
- **Default (0.7/0.3)**: General purpose searches, most common scenario

**Performance Benefits:**
- **Comprehensive Coverage**: Finds both semantically similar and keyword-matching documents
- **Robustness**: Reduces risk of missing relevant results due to single-method limitations
- **Flexibility**: Easily tunable for different content types and query styles
- **Efficiency**: Parallel execution means minimal performance penalty over single methods

## Japanese Query Support

Oboyu provides advanced Japanese language support through specialized processing:

### Tokenization

Japanese text is processed using MeCab morphological analyzer:

- **MeCab with fugashi**: Primary tokenizer for Japanese text
- **Part-of-speech filtering**: Extracts content words (nouns, verbs, adjectives)
- **Stop word removal**: Filters common particles and auxiliary words
- **Fallback tokenizer**: Simple character-based tokenization when MeCab is unavailable

**Example tokenization:**
```
Input:  "機械学習ではPythonがよく使われています"
Output: ["機械学習", "Python", "使わ", "れる"]
```

### Character Normalization

Text normalization ensures consistent matching:

- **Unicode normalization**: Converts to NFKC form
- **Character variant handling**: ひらがな/カタカナ conversion when needed
- **Width normalization**: Full-width ↔ half-width character conversion

### Query Processing Examples

```bash
# Natural Japanese queries
oboyu query "機械学習のアルゴリズムについて教えて"
oboyu query "Pythonでのデータ処理方法"

# Mixed Japanese-English queries
oboyu query "REST APIの設計パターン"
oboyu query "データベースのNormalization理論"

# Technical terminology
oboyu query "非同期処理とPromise"
oboyu query "マイクロサービスアーキテクチャの利点"
```

### Search Mode Recommendations for Japanese

- **Vector search**: Best for conceptual Japanese queries
- **BM25 search**: Excellent for specific Japanese technical terms
- **Hybrid search**: Optimal balance for most Japanese content

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

The Query Engine provides a comprehensive command-line interface:

### Basic Usage

```bash
# Default hybrid search
oboyu query "システムの設計原則について教えてください"

# Specify search mode
oboyu query --mode vector "What are the key concepts?"
oboyu query --mode bm25 "データベース設計"
oboyu query --mode hybrid "machine learning algorithms"
```

### Advanced Options

```bash
# Control number of results
oboyu query --limit 10 "Python programming best practices"

# Adjust hybrid search weights
oboyu query --vector-weight 0.8 --bm25-weight 0.2 "システム設計"

# Use different language settings
oboyu query --language ja "英語ドキュメントの日本語検索"

# Enable reranker for improved results
oboyu query --use-reranker "complex technical concepts"
```

### Complete Options

```bash
oboyu query [OPTIONS] QUERY

Options:
  --mode [vector|bm25|hybrid]     Search mode (default: hybrid)
  --limit INTEGER                 Number of results (default: 10)
  --vector-weight FLOAT          Weight for vector scores (default: 0.7)
  --bm25-weight FLOAT            Weight for BM25 scores (default: 0.3)
  --language TEXT                Language hint for processing
  --use-reranker / --no-reranker Enable reranking (default: auto)
  --help                         Show this message and exit
```

### Real-world Examples

```bash
# Research query with semantic understanding
oboyu query --mode vector "distributed systems consistency models"

# Exact terminology lookup
oboyu query --mode bm25 "REST API status codes 404"

# Balanced search for documentation
oboyu query "Pythonでの例外処理のベストプラクティス"

# Technical documentation with custom weights
oboyu query --vector-weight 0.3 --bm25-weight 0.7 --limit 15 "database normalization rules"
```

## MCP Protocol

The MCP server provides a standardized interface for external tools and AI assistants:

### Starting MCP Server

```bash
# Start MCP server for integration with Claude/other AI tools
oboyu mcp

# Specify custom options
oboyu mcp --log-level debug
```

### Search Request Examples

**Hybrid Search (Default):**
```bash
# Through MCP tools in Claude or other AI assistants
search_documents("機械学習の基本的なアルゴリズム", limit=5)
```

**Vector Search:**
```bash
search_documents("design patterns in software architecture", mode="vector", limit=10)
```

**BM25 Search:**
```bash
search_documents("データベース正規化", mode="bm25")
```

**Custom Hybrid Weights:**
```bash
search_documents("REST API best practices", 
                mode="hybrid", 
                vector_weight=0.6, 
                bm25_weight=0.4, 
                limit=8)
```

### Integration Examples

**With Claude Code:**
The MCP server integrates seamlessly with Claude for code analysis and documentation queries:

```bash
# Claude can search your codebase for relevant documentation
"How do I implement authentication in this project?"
# → Uses hybrid search to find auth-related docs and code examples

# Claude can find specific technical details
"What are the database migration patterns used here?"
# → Uses BM25 search for exact terminology matching
```

**With Custom Tools:**
```python
import mcp_client

# Connect to Oboyu MCP server
client = mcp_client.connect("oboyu")

# Perform searches programmatically
results = client.search_documents(
    query="Python async programming patterns",
    mode="hybrid",
    vector_weight=0.7,
    bm25_weight=0.3,
    limit=5
)
```

## Performance Considerations

### Search Mode Performance

- **Vector search**: Fast for small-medium datasets (< 100K documents)
- **BM25 search**: Scales well with large datasets, fast keyword lookups
- **Hybrid search**: Slightly slower but provides best quality results

### Optimization Strategies

- **Parallel processing**: Vector and BM25 searches run in parallel during hybrid mode
- **Index optimization**: HNSW parameters tuned for Japanese content
- **Tokenization caching**: MeCab tokenization results cached for common queries
- **Score normalization**: Efficient min-max normalization prevents score dominance

### Japanese-specific Optimizations

- **Tokenizer selection**: Automatic fallback from MeCab to simple tokenizer
- **Character normalization**: Preprocessing reduces search space
- **Stop word filtering**: Removes common Japanese particles for better performance
- **Memory management**: Efficient handling of Japanese Unicode strings

### Tuning Recommendations

For **large Japanese document collections** (>50K docs):
```bash
# Favor BM25 for better scaling
oboyu query --vector-weight 0.4 --bm25-weight 0.6 "検索クエリ"
```

For **multilingual content**:
```bash
# Favor vector search for cross-language understanding
oboyu query --vector-weight 0.8 --bm25-weight 0.2 "technical concepts"
```

For **precise technical documentation**:
```bash
# Balanced approach works best
oboyu query --mode hybrid "API documentation patterns"
```

## Integration with Other Components

- Accesses indexed data via the database created by the Indexer
- Maintains clean separation from indexing and crawling processes
- Provides well-defined interfaces for external integration