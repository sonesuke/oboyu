# Oboyu Indexer Architecture

## Overview

The Indexer is the core processing component of Oboyu, responsible for analyzing documents, generating search indexes, and managing the database. It implements specialized processing for Japanese text and creates both vector and BM25 indexes for powerful search capabilities.

## Design Goals

- Process documents efficiently with language-specific handling
- Generate high-quality embeddings for semantic search
- Create effective keyword indexes for BM25 search
- Maintain a flexible database schema
- Support incremental updates and multiple embedding models

## Component Structure

### Document Processor

The Document Processor handles:

- Language-specific text preparation
- Document chunking with configurable size and overlap
- Special content handling (code blocks, tables, etc.)
- Text normalization and cleaning

```python
def process_document(document):
    # Implementation details
    # Returns document chunks ready for indexing
```

### Japanese Processor

The Japanese Processor provides specialized handling for Japanese text:

- Tokenization using MeCab/Sudachi
- Japanese-specific text normalization
- Character variation handling
- Mixed Japanese-English content processing

```python
def process_japanese_text(text):
    # Implementation details
    # Returns tokenized and normalized Japanese text
```

### Embedding Generator

The Embedding Generator creates vector representations:

- Loads and manages embedding models
- Generates embeddings for document chunks
- Handles batching for efficient processing
- Manages embedding caching and updates

```python
def generate_embeddings(chunks, model_name="default"):
    # Implementation details
    # Returns vectors for document chunks
```

### Database Manager

The Database Manager handles:

- Database initialization and schema management
- DuckDB connection and extension loading
- Transaction management for data consistency
- Index creation and maintenance

```python
def initialize_database():
    # Implementation details
    # Sets up DuckDB with required extensions and schema
```

## Database Schema

The Indexer creates and maintains the following schema:

```sql
-- Documents table stores the original document metadata
CREATE TABLE documents (
    id VARCHAR PRIMARY KEY,
    path VARCHAR,
    title VARCHAR,
    content TEXT,
    language VARCHAR,
    created_at TIMESTAMP,
    modified_at TIMESTAMP,
    metadata JSONB
);

-- Chunks table stores document segments for granular search
CREATE TABLE chunks (
    id VARCHAR PRIMARY KEY,
    doc_id VARCHAR,
    content TEXT,
    chunk_index INTEGER,
    FOREIGN KEY (doc_id) REFERENCES documents (id)
);

-- Embeddings table stores vector representations for semantic search
CREATE TABLE embeddings (
    id VARCHAR PRIMARY KEY,
    chunk_id VARCHAR,
    model VARCHAR,
    vector ARRAY,
    created_at TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES chunks (id)
);

-- Create HNSW index for vector search
CREATE INDEX vector_idx ON embeddings 
USING HNSW (vector) 
WITH (metric = 'cosine');

-- Create FTS index for BM25 search
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    chunk_id UNINDEXED,
    tokenize='porter unicode61'
);
```

## Index Types

### Vector Index (HNSW)

- Uses DuckDB's VSS extension for vector similarity search
- Creates Hierarchical Navigable Small Worlds (HNSW) index
- Supports multiple distance metrics (cosine, L2, inner product)
- Configurable for accuracy vs. performance tradeoffs

### BM25 Index

- Uses DuckDB's FTS extension for full-text search
- Implements BM25 ranking algorithm
- Enhanced with Japanese-aware tokenization
- Optimized for keyword-based retrieval

## Data Flow

1. Receive documents from the Crawler
2. Process documents into chunks
3. Generate embeddings for each chunk
4. Store documents, chunks, and embeddings in the database
5. Create and maintain search indexes

## Configuration Options

The Indexer is configured through the following settings in `config.yaml`:

```yaml
indexer:
  chunk_size: 512                 # Size of document chunks
  chunk_overlap: 50               # Overlap between chunks
  embedding_model: "intfloat/multilingual-e5-large"
  japanese_tokenizer: "sudachi"   # Japanese tokenizer to use
  batch_size: 32                  # Batch size for embedding generation
  cache_embeddings: true          # Whether to cache embeddings
```

## Performance Considerations

- Uses batch processing for efficient embedding generation
- Implements parallel processing where appropriate
- Optimizes database operations for large document collections
- Manages memory usage for embedding models

## Integration with Other Components

- Receives documents from the Crawler in standardized format
- Provides indexed data to the Query Engine via the database
- Maintains clean separation of concerns for system modularity