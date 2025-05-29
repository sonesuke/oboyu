# Oboyu Indexer Architecture

## Overview

The Indexer is the core processing component of Oboyu, responsible for analyzing documents, generating search indexes, and managing the database. It implements specialized processing for Japanese text and creates both vector indexes for semantic search and BM25 indexes for keyword-based search capabilities.

## Design Goals

- Process documents efficiently with language-specific handling
- Generate high-quality embeddings for semantic search with Ruri v3 model
- Create BM25 indexes for precise keyword matching with Japanese tokenization
- Maintain a flexible database schema with DuckDB and VSS extension
- Support incremental updates and embedding caching
- Optimize for Japanese text processing with MeCab morphological analysis
- Provide hybrid search combining vector and BM25 approaches

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

### BM25 Indexer

The BM25 Indexer creates keyword-based search indexes:

- Japanese tokenization using MeCab morphological analyzer with fugashi
- Fallback to simple character-based tokenization when MeCab unavailable
- Inverted index construction for fast keyword lookups
- Document statistics calculation for BM25 scoring
- Support for incremental index updates

```python
def build_bm25_index(chunks):
    # Tokenize documents using Japanese-aware tokenizer
    tokenized_docs = [tokenizer.tokenize(chunk.content) for chunk in chunks]
    # Build inverted index
    inverted_index = _build_inverted_index(tokenized_docs)
    # Calculate document statistics
    doc_stats = _calculate_document_statistics(tokenized_docs)
    # Store in database
    store_bm25_index(inverted_index, doc_stats)
```

### Japanese Tokenizer

The Tokenizer provides Japanese text analysis:

- **MeCab Integration**: Uses fugashi wrapper for morphological analysis
- **Part-of-Speech Filtering**: Extracts content words (nouns, verbs, adjectives)
- **Stop Word Removal**: Filters particles and auxiliary words
- **Text Normalization**: Unicode and character width normalization
- **Fallback Support**: Simple tokenization when Japanese tools unavailable

```python
def tokenize_japanese(text):
    if HAS_JAPANESE_TOKENIZER:
        # Use MeCab for morphological analysis
        return japanese_tokenizer.tokenize(text)
    else:
        # Fallback to simple tokenization
        return fallback_tokenizer.tokenize(text)
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

##### ONNX Quantization Support

**NEW**: Oboyu now supports ONNX dynamic quantization for enhanced performance:

- **Dynamic Quantization**: Weights are quantized to INT8 while activations remain in FP32
- **Automatic Application**: Quantization is enabled by default for optimal performance
- **Performance Benefits**: 20-50% additional speedup on top of ONNX optimization
- **Memory Reduction**: 50-75% reduction in model memory usage
- **Minimal Accuracy Loss**: Typically <5% accuracy degradation for Japanese semantic search

Configuration options:

```yaml
indexer:
  use_onnx: true  # Enable ONNX optimization
  onnx_quantization:
    enabled: true       # Enable quantization (default: true)
    method: "dynamic"   # Quantization method (dynamic/static/fp16)
    weight_type: "uint8"  # Weight quantization type (uint8/int8)
```

Key features:
- **Zero Configuration**: Dynamic quantization requires no calibration dataset
- **Graceful Fallback**: Falls back to non-quantized ONNX if quantization fails
- **Transparent Caching**: Quantized models are cached separately for quick reuse

##### ONNX Model Cache Directory Structure

ONNX models are cached following the XDG Base Directory specification:

```
$XDG_CACHE_HOME/oboyu/embedding/cache/     # ~/.cache/oboyu/embedding/cache/
├── models/                                # ONNX converted models
│   └── onnx/                              # ONNX model subdirectory
│       ├── cl-nagoya_ruri-v3-30m/
│       │   ├── model.onnx                 # Converted ONNX model
│       │   ├── model_optimized.onnx       # Optimized ONNX model (if optimization succeeds)
│       │   ├── model_quantized.onnx       # Quantized ONNX model (NEW)
│       │   ├── tokenizer_config.json      # Tokenizer configuration
│       │   ├── special_tokens_map.json    # Special tokens mapping
│       │   ├── vocab.txt                  # Vocabulary file
│       │   └── onnx_config.json           # ONNX-specific configuration
│       └── other_model_name/
│           └── ...
└── [embedding cache files]                # Regular embedding cache (*.pkl files)
```

The ONNX models are stored in the cache directory because they can be regenerated from the original models, making them appropriate for cache storage according to XDG specifications.

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
- HNSW index for vector similarity search with persistence support
- Efficient vector storage and search
- Transaction management for data consistency
- Index clearing and maintenance

```python
def setup():
    # Install and load VSS extension
    conn.execute("INSTALL vss; LOAD vss;")
    # Enable experimental persistence for HNSW indexes
    conn.execute("SET hnsw_enable_experimental_persistence=true")
    # Create database schema
    _create_schema()
    # Check if HNSW index exists before creating (NEW)
    if not _hnsw_index_exists():
        _create_hnsw_index()
    else:
        logger.info("Using existing HNSW index")

def clear():
    # Clear all data from the database
    conn.execute("DELETE FROM embeddings")
    conn.execute("DELETE FROM chunks")
    # Recreate index to ensure clean state
    recreate_hnsw_index(force=True)
```

## Database Schema

The Indexer creates and maintains the following schema for both vector and BM25 search:

### Core Tables

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
```

### BM25 Index Tables

```sql
-- Vocabulary table stores term statistics for BM25 scoring
CREATE TABLE vocabulary (
    term VARCHAR PRIMARY KEY,
    document_frequency INTEGER,    -- Number of documents containing this term
    collection_frequency INTEGER   -- Total occurrences across all documents
);

-- Inverted index table for fast keyword lookups
CREATE TABLE inverted_index (
    term VARCHAR,
    chunk_id VARCHAR,
    term_frequency INTEGER,       -- Frequency of term in this chunk
    positions JSON,               -- Term positions (for phrase queries)
    PRIMARY KEY (term, chunk_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks (id)
);

-- Document statistics for BM25 normalization
CREATE TABLE document_stats (
    chunk_id VARCHAR PRIMARY KEY,
    total_terms INTEGER,          -- Total number of terms in document
    unique_terms INTEGER,         -- Number of unique terms
    avg_term_frequency FLOAT,     -- Average term frequency
    FOREIGN KEY (chunk_id) REFERENCES chunks (id)
);

-- Collection-wide statistics for BM25 calculation
CREATE TABLE collection_stats (
    key VARCHAR PRIMARY KEY,
    value FLOAT                   -- Stores avg_document_length, total_documents, etc.
);
```

### Search Indexes

```sql
-- HNSW index for vector search
CREATE INDEX vector_idx ON embeddings 
USING HNSW (vector) 
WITH (
    metric = 'cosine',
    ef_construction = 128,
    ef_search = 64,
    M = 16
);

-- B-tree indexes for efficient BM25 lookups
CREATE INDEX idx_vocabulary_term ON vocabulary (term);
CREATE INDEX idx_inverted_index_term ON inverted_index (term);
CREATE INDEX idx_inverted_index_chunk ON inverted_index (chunk_id);
```

## Index Types

Oboyu creates two complementary index types for comprehensive search capabilities:

### Vector Index (HNSW)

The vector index enables semantic similarity search:

- Uses DuckDB's VSS extension for vector similarity search
- Creates Hierarchical Navigable Small Worlds (HNSW) index
- Uses cosine similarity as distance metric
- **Best for**: conceptual queries, cross-language understanding, semantic matching
- **NEW**: Supports index persistence (experimental) for faster startup

**Configurable parameters for accuracy vs. performance tradeoffs:**
- `ef_construction`: Controls index build quality (default: 128)
- `ef_search`: Controls search quality (default: 64)
- `M`: Number of bidirectional links (default: 16)
- `M0`: Level-0 connections (default: 2*M)

#### HNSW Index Persistence (NEW)

Oboyu now preserves HNSW indexes across database restarts:

- **Automatic Detection**: Checks for existing indexes on startup
- **Faster Queries**: No need to rebuild index after restart
- **Experimental Feature**: Enabled via `SET hnsw_enable_experimental_persistence=true`
- **Index Maintenance**: Provides `recreate_hnsw_index()` method for manual rebuilds

**Important Notes:**
- The entire index must fit in RAM (not buffer-managed)
- No incremental updates - full index is serialized on checkpoint
- WAL recovery not fully implemented - use with caution in production
- Index is loaded lazily when first accessed after restart

### BM25 Index (Inverted Index)

The BM25 index enables precise keyword matching:

- Uses inverted index structure for fast term lookups
- Implements BM25 ranking algorithm with Japanese tokenization
- Stores term statistics and document frequencies
- **Best for**: exact keyword matching, specific terminology, technical documentation

**Key components:**
- **Vocabulary**: Term statistics (document frequency, collection frequency)
- **Inverted Index**: Term-to-document mappings with frequencies
- **Document Statistics**: Per-document term counts and averages
- **Collection Statistics**: Global statistics for normalization

**BM25 Parameters:**
- `k1`: Term frequency saturation parameter (default: 1.2)
- `b`: Length normalization parameter (default: 0.75)
- `min_token_length`: Minimum token length (default: 2)

### Japanese Tokenization Features

For BM25 indexing, Japanese text undergoes specialized processing:

- **Morphological Analysis**: Uses MeCab with fugashi for accurate word segmentation
- **Part-of-Speech Filtering**: Extracts content words (nouns, verbs, adjectives, adverbs)
- **Stop Word Removal**: Filters particles (助詞), auxiliary verbs (助動詞), symbols
- **Text Normalization**: Unicode NFKC normalization, character width conversion
- **Base Form Extraction**: Uses lemmatized forms when available

**Example Japanese tokenization:**
```
Input:  "機械学習アルゴリズムの実装について"
Tokens: ["機械学習", "アルゴリズム", "実装"]
Filtered: Removes particles like "の", "について"
```

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
        
        # ONNX quantization settings (NEW)
        "onnx_quantization": {
            "enabled": true,       # Enable dynamic quantization (default: true)
            "method": "dynamic",   # Quantization method (dynamic/static/fp16)
            "weight_type": "uint8", # Weight quantization type (uint8/int8)
        },
        
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
        
        # BM25 settings
        "use_bm25": true,  # Enable BM25 indexing (enabled by default)
        "bm25_k1": 1.2,  # Term frequency saturation parameter
        "bm25_b": 0.75,  # Length normalization parameter
        "bm25_min_token_length": 2,  # Minimum token length for indexing
        "use_japanese_tokenizer": true,  # Use MeCab for Japanese tokenization
        
        # Processing settings
        "max_workers": 4,  # Maximum number of worker threads
    }
}
```

### BM25-specific Configuration

**Japanese Tokenization Settings:**
- `use_japanese_tokenizer`: Enable MeCab-based tokenization for Japanese text
- `bm25_min_token_length`: Filter out very short tokens (default: 2)

**BM25 Algorithm Parameters:**
- `bm25_k1`: Controls term frequency saturation (higher = less saturation)
- `bm25_b`: Controls document length normalization (0 = no normalization, 1 = full normalization)

**Recommended Settings by Use Case:**

**For Japanese documentation:**
```yaml
indexer:
  use_japanese_tokenizer: true
  bm25_k1: 1.2  # Standard value works well for Japanese
  bm25_b: 0.75  # Good balance for varied document lengths
```

**For technical code documentation:**
```yaml
indexer:
  bm25_k1: 1.5  # Higher saturation for technical terms
  bm25_b: 0.6   # Less length normalization for code snippets
```

**For mixed-language content:**
```yaml
indexer:
  use_japanese_tokenizer: true  # Handles Japanese, falls back for other languages
  bm25_min_token_length: 1     # Include single-character terms for code
```

## Performance Considerations

- Uses ThreadPoolExecutor for parallel document processing
- Implements batch processing for efficient embedding generation
- Persistent embedding cache to avoid regenerating embeddings
- Periodic HNSW index recompaction for better search performance
- Intelligent chunk boundary detection at sentence/paragraph level
- ONNX optimization provides 2-4x faster CPU inference for embeddings
- Dynamic quantization adds 20-50% additional speedup with minimal accuracy loss

## Integration with Other Components

- Directly accepts CrawlerResult objects from the Crawler component
- Provides search functionality via the Indexer.search method
- Maintains backwards compatibility with the crawler's Japanese text processing
- Supports incremental updates for new documents