---
id: query-engine
title: Oboyu Query Engine Architecture
sidebar_position: 50
---

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

Hybrid search combines both approaches for optimal results using **RRF (Reciprocal Rank Fusion)**, a proven rank-based fusion method:

- **Parallel Execution**: Executes both vector and BM25 searches simultaneously for efficiency
- **Rank-based Fusion**: Uses RRF algorithm instead of score-based weighting for more robust results
- **Configurable RRF Parameter**: Uses a configurable `k` parameter (default: 60) to control fusion behavior
- **Result Fusion**: Merges and re-ranks results using ranks rather than scores for better handling of different scoring systems
- **Best for**: most general queries, balanced precision and recall, complex information needs, proper nouns and technical terms

**How RRF Hybrid Search Works:**

1. **Query Processing**: The same query is processed for both search methods
2. **Parallel Search**: Vector and BM25 searches execute simultaneously  
3. **Rank Assignment**: Each method assigns ranks to documents (1st, 2nd, 3rd, etc.)
4. **RRF Calculation**: Final score = 1/(k + rank_vector) + 1/(k + rank_bm25)
5. **Result Combination**: Documents are scored using RRF formula where k=60 by default
6. **Ranking**: Results are sorted by RRF score (higher is better) and top-k selected

**RRF Configuration:**

```bash
# Default RRF hybrid search (recommended for most use cases)
oboyu query "Pythonでの非同期処理の実装方法"

# Custom RRF parameter for different fusion behavior
oboyu query --rrf-k 30 "database optimization techniques"  # More aggressive fusion
oboyu query --rrf-k 100 "REST API status codes"           # More conservative fusion
```

**RRF Parameter Effects:**

- **Lower k (e.g., 30)**: More aggressive fusion, higher weight to top-ranked results from each method
- **Higher k (e.g., 100)**: More conservative fusion, more balanced contribution from all ranks
- **Default k=60**: Optimal balance for most content types and query patterns

**Performance Benefits:**
- **Comprehensive Coverage**: Finds both semantically similar and keyword-matching documents
- **Robustness**: Rank-based fusion is more stable than score-based weighting across different content types
- **Better Term Handling**: Superior performance on proper nouns and technical terminology
- **Efficiency**: Parallel execution means minimal performance penalty over single methods
- **Parameter Simplicity**: Single `k` parameter is easier to tune than dual weight system

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
  rrf_k: 60                      # RRF parameter for hybrid search (default: 60)
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

# Adjust RRF parameter for hybrid search
oboyu query --rrf-k 30 "システム設計"  # More aggressive fusion

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
  --rrf-k INTEGER                RRF parameter for hybrid search (default: 60)
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

# Technical documentation with custom RRF parameter
oboyu query --rrf-k 30 --limit 15 "database normalization rules"
```


## Performance Considerations

### Search Mode Performance

- **Vector search**: Fast for small-medium datasets (&lt; 100K documents)
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
# Use conservative RRF for better scaling
oboyu query --rrf-k 100 "検索クエリ"
```

For **precise technical documentation**:
```bash
# Use aggressive RRF for better top-result fusion
oboyu query --rrf-k 30 "API documentation patterns"
```

For **general purpose searches**:
```bash
# Default RRF parameter works best
oboyu query --mode hybrid "technical concepts"
```

## Integration with Other Components

- Accesses indexed data via the database created by the Indexer
- Maintains clean separation from indexing and crawling processes
- Provides well-defined interfaces for external integration
