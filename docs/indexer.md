# Oboyu Indexer Architecture

## Overview

The Indexer is the core processing component of Oboyu, responsible for analyzing documents, generating search indexes, and managing the database. It implements specialized processing for Japanese text and creates vector indexes for powerful semantic search capabilities.

## Design Goals

- Process documents efficiently with language-specific handling
- Generate high-quality embeddings for semantic search with Ruri v3 model
- Maintain a flexible database schema with DuckDB and VSS extension
- Support incremental updates and embedding caching
- Optimize for Japanese text processing

## Component Structure

### Document Processor

The Document Processor handles:

- Language-specific text preparation
- Document chunking with configurable size and overlap
- Text normalization and cleaning
- Prefix handling for Ruri v3 embedding model

```python
def process_document(path, content, title, language, metadata=None):
    # Split content into chunks
    chunks = _chunk_text(content)
    # Apply prefix to each chunk
    prefixed_chunks = [f"検索文書: {chunk}" for chunk in chunks]
    # Returns document chunks ready for indexing
    return chunks_with_metadata
```

### Japanese Processor

The Japanese Processor provides specialized handling for Japanese text:

- Integration with crawler's Japanese text processing
- Character normalization (full-width to half-width conversion)
- Line ending standardization
- Encoding detection and handling

```python
def process_japanese_text(text, encoding="utf-8"):
    # Normalize Japanese text
    normalized = _normalize_japanese(text)
    # Standardize line endings
    standardized = _standardize_line_endings(normalized)
    # Returns processed Japanese text
    return standardized
```

### Embedding Generator

The Embedding Generator creates vector representations:

- Uses SentenceTransformer with Ruri v3-30m model
- Generates 256-dimensional embeddings for document chunks
- Handles batching for efficient processing
- Implements persistent embedding cache
- Applies specialized prefix scheme for different text types
- **NEW**: Supports ONNX optimization for faster CPU inference

```python
def generate_embeddings(chunks):
    # Apply document prefix
    prefixed_chunks = [f"検索文書: {chunk.content}" for chunk in chunks]
    # Generate embeddings in batches
    embeddings = model.encode(prefixed_chunks, batch_size=8)
    # Returns vectors for document chunks
    return embeddings
```

#### ONNX Optimization

Oboyu supports ONNX (Open Neural Network Exchange) optimization for accelerated inference:

- **Automatic Conversion**: Models are automatically converted to ONNX format on first use
- **Optimized Runtime**: Uses ONNX Runtime with graph optimization for better CPU performance
- **Transparent Switching**: Can toggle between PyTorch and ONNX backends via configuration
- **Model Caching**: ONNX models are cached to avoid re-conversion overhead
- **Performance Benefits**: Typically 2-4x faster inference on CPU compared to PyTorch
- **Lazy Loading**: Models are loaded only when first needed, improving startup time

To enable ONNX optimization (enabled by default for CPU):

```yaml
indexer:
  use_onnx: true  # Enable ONNX optimization
  embedding_device: cpu  # ONNX is most beneficial for CPU
```

#### Lazy Loading

The EmbeddingGenerator implements lazy loading for optimal performance:

- **Fast Initialization**: Creating EmbeddingGenerator instances is nearly instantaneous
- **On-Demand Loading**: Models are only loaded when first accessed (e.g., during embedding generation)
- **Global Caching**: Multiple instances with the same configuration share cached models
- **Memory Efficiency**: Reduces memory usage when models aren't immediately needed

### Database Manager

The Database Manager handles:

- DuckDB setup with VSS extension
- Schema creation and management
- HNSW index for vector similarity search
- Efficient vector storage and search
- Transaction management for data consistency
- Index clearing and maintenance

```python
def setup():
    # Install and load VSS extension
    conn.execute("INSTALL vss; LOAD vss;")
    # Create database schema
    _create_schema()
    # Create HNSW index
    _create_hnsw_index()

def clear():
    # Clear all data from the database
    conn.execute("DELETE FROM embeddings")
    conn.execute("DELETE FROM chunks")
    # Recompact index after clearing
    recompact_index()
```

## Database Schema

The Indexer creates and maintains the following schema:

```sql
-- Chunks table stores document segments for granular search
CREATE TABLE chunks (
    id VARCHAR PRIMARY KEY,
    path VARCHAR,             -- Path to file
    title VARCHAR,            -- Chunk title (or original filename)
    content TEXT,             -- Chunk text content
    chunk_index INTEGER,      -- Chunk position in original document
    language VARCHAR,         -- Language code
    created_at TIMESTAMP,     -- Creation timestamp
    modified_at TIMESTAMP,    -- Modification timestamp
    metadata JSONB            -- Additional metadata
);

-- Embeddings table stores vector representations for semantic search
CREATE TABLE embeddings (
    id VARCHAR PRIMARY KEY,
    chunk_id VARCHAR,         -- Related chunk ID
    model VARCHAR,            -- Embedding model used
    vector FLOAT[256],        -- 256-dimensional vector (ruri-v3-30m specific)
    created_at TIMESTAMP,     -- Embedding generation timestamp
    FOREIGN KEY (chunk_id) REFERENCES chunks (id)
);

-- Create HNSW index for vector search
CREATE INDEX vector_idx ON embeddings 
USING HNSW (vector) 
WITH (
    metric = 'cosine',
    ef_construction = 128,
    ef_search = 64,
    M = 16
);
```

## Index Types

### Vector Index (HNSW)

- Uses DuckDB's VSS extension for vector similarity search
- Creates Hierarchical Navigable Small Worlds (HNSW) index
- Uses cosine similarity as distance metric
- Configurable parameters for accuracy vs. performance tradeoffs:
  - `ef_construction`: Controls index build quality (default: 128)
  - `ef_search`: Controls search quality (default: 64)
  - `M`: Number of bidirectional links (default: 16)
  - `M0`: Level-0 connections (default: 2*M)

## Ruri v3-30m Embedding Model

Oboyu uses the [Ruri v3-30m](https://huggingface.co/cl-nagoya/ruri-v3-30m) embedding model by Cyberagent Nagoya for Japanese and multilingual support.

### Key Features

- **256-dimensional embeddings**: Compact but powerful representations
- **Multilingual Support**: Optimized for Japanese, English, and other languages
- **Maximum Sequence Length**: 8192 tokens (much longer than most models)
- **Symmetric Similarity**: Performance optimized for document-to-query matching

### Prefix Scheme

Ruri v3 uses a 1+3 prefix scheme for different embedding purposes:

- **Document Prefix** (`検索文書: `): Added to document chunks for indexing
- **Query Prefix** (`検索クエリ: `): Added to search queries
- **Topic Prefix** (`トピック: `): Used for topic information
- **General Prefix** (empty string): Used for general semantic encoding

## Data Flow

1. Receive documents from the Crawler
2. Process documents into chunks with appropriate prefix
3. Generate embeddings for each chunk
4. Store chunks and embeddings in the database
5. Create and maintain HNSW index
6. Process search queries with matching prefix

## Configuration Options

The Indexer is configured through the IndexerConfig class:

```python
DEFAULT_CONFIG = {
    "indexer": {
        # Document processing settings
        "chunk_size": 1024,  # Default chunk size in characters
        "chunk_overlap": 256,  # Default overlap between chunks
        
        # Embedding settings
        "embedding_model": "cl-nagoya/ruri-v3-30m",  # Default embedding model
        "embedding_device": "cpu",  # Default device for embeddings
        "batch_size": 8,  # Default batch size for embedding generation
        "max_seq_length": 8192,  # Maximum sequence length
        "use_onnx": true,  # Use ONNX optimization for faster inference
        
        # Prefix scheme settings (Ruri v3's 1+3 prefix scheme)
        "document_prefix": "検索文書: ",  # Prefix for documents
        "query_prefix": "検索クエリ: ",  # Prefix for search queries
        "topic_prefix": "トピック: ",  # Prefix for topic information
        "general_prefix": "",  # Prefix for general semantic encoding
        
        # Database settings
        "db_path": "oboyu.db",  # Default database path
        
        # VSS (Vector Similarity Search) settings
        "ef_construction": 128,  # Index construction parameter
        "ef_search": 64,  # Search time parameter
        "m": 16,  # Number of bidirectional links in HNSW graph
        "m0": None,  # Level-0 connections (None means use 2*M)
        
        # Processing settings
        "max_workers": 4,  # Maximum number of worker threads
    }
}
```

## Performance Considerations

- Uses ThreadPoolExecutor for parallel document processing
- Implements batch processing for efficient embedding generation
- Persistent embedding cache to avoid regenerating embeddings
- Periodic HNSW index recompaction for better search performance
- Intelligent chunk boundary detection at sentence/paragraph level
- ONNX optimization provides 2-4x faster CPU inference for embeddings

## Integration with Other Components

- Directly accepts CrawlerResult objects from the Crawler component
- Provides search functionality via the Indexer.search method
- Maintains backwards compatibility with the crawler's Japanese text processing
- Supports incremental updates for new documents