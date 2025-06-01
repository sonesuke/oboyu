# Oboyu Indexer Architecture

## Table of Contents

- [Overview](#overview)
- [Design Goals](#design-goals)
- [Modular Architecture](#modular-architecture)
- [Component Structure](#component-structure)
- [Database Schema](#database-schema)
- [Index Types](#index-types)
- [Ruri v3-30m Embedding Model](#ruri-v3-30m-embedding-model)
- [Configuration](#configuration)
- [Data Flow](#data-flow)
- [Performance Considerations](#performance-considerations)
- [API Usage Examples](#api-usage-examples)
- [Migration from Legacy Architecture](#migration-from-legacy-architecture)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Integration with Other Components](#integration-with-other-components)

## Overview

The Indexer is the core processing component of Oboyu, responsible for analyzing documents, generating search indexes, and managing the database. It implements specialized processing for Japanese text and creates both vector indexes for semantic search and BM25 indexes for keyword-based search capabilities.

The indexer has been completely refactored to use a **modular service-oriented architecture** that provides better maintainability, testability, and flexibility.

### Key Architectural Changes (v2.0)

- **✅ Modular Services**: Database, Embedding, Tokenizer, and Reranker services are now independent
- **✅ Type-Safe Configuration**: Replaced flat dicts with typed dataclasses
- **✅ Lazy Loading**: Models load only when needed for faster startup
- **✅ Enhanced Testing**: Each service can be independently tested and mocked
- **✅ Legacy Compatibility**: Old import paths still work during transition
- **✅ Improved Error Handling**: Comprehensive logging and error reporting

## Design Goals

- Process documents efficiently with language-specific handling
- Generate high-quality embeddings for semantic search with Ruri v3 model
- Create BM25 indexes for precise keyword matching with Japanese tokenization
- Maintain a flexible database schema with DuckDB and VSS extension
- Support incremental updates and embedding caching
- Optimize for Japanese text processing with MeCab morphological analysis
- Provide hybrid search combining vector and BM25 approaches
- **NEW**: Modular architecture for independent service management
- **NEW**: Comprehensive configuration system with type safety
- **NEW**: Enhanced reranker integration for improved search accuracy

## Modular Architecture

The indexer follows a clean service-oriented architecture with clear separation of concerns:

### Core Services

```python
# Main indexer facade that coordinates services
class Indexer:
    """Lightweight facade class that coordinates modular services."""
    
    def __init__(self, config: Optional[IndexerConfig] = None):
        # Initialize configuration
        self.config = config or IndexerConfig()
        
        # Initialize modular services
        self.database_service = DatabaseService(...)
        self.embedding_service = EmbeddingService(...)
        self.tokenizer_service = TokenizerService(...)
        self.reranker_service = RerankerService(...)
        
        # Initialize search engines
        self.search_engine = SearchEngine(...)

# Database service for data management
class DatabaseService:
    """Manages database operations and schema."""
    
    def store_chunks(self, chunks: List[Chunk]) -> None
    def store_embeddings(self, chunk_ids: List[str], embeddings: List[NDArray]) -> None
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]
    def delete_chunks_by_path(self, path: Union[str, Path]) -> int
    
# Embedding service for vector generation  
class EmbeddingService:
    """Handles embedding model loading and inference."""
    
    def generate_embeddings(self, texts: List[str]) -> List[NDArray[np.float32]]
    
# Tokenizer service for text processing
class TokenizerService:
    """Provides Japanese and multilingual tokenization."""
    
    def tokenize(self, text: str) -> List[str]
    def get_term_frequencies(self, text: str) -> Dict[str, int]
    
# Reranker service for result improvement
class RerankerService:
    """Manages reranker models and scoring."""
    
    def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]
```

### Search Engines

```python
# Specialized search implementations
class VectorSearch:
    """Vector similarity search using embeddings."""
    
class BM25Search:  
    """Keyword-based search using BM25 algorithm."""
    
class HybridSearch:
    """Combines vector and BM25 with configurable weights."""

class SearchEngine:
    """Unified search interface coordinating different search modes."""
    
    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        limit: int = 10,
        **kwargs
    ) -> List[SearchResult]
```

### Configuration System

The indexer uses a modular configuration system with type safety:

```python
@dataclass 
class IndexerConfig:
    """Main indexer configuration."""
    
    model: ModelConfig           # Embedding and reranker models
    search: SearchConfig         # Search algorithm settings  
    processing: ProcessingConfig # Text processing options

@dataclass
class ModelConfig:
    """Model-related configuration."""
    
    embedding_model: str = "cl-nagoya/ruri-v3-30m"
    embedding_device: str = "cpu"
    batch_size: int = 64
    use_onnx: bool = True
    reranker_model: str = "cl-nagoya/ruri-reranker-small"
    use_reranker: bool = False

@dataclass
class ProcessingConfig:
    """Document processing configuration."""
    
    chunk_size: int = 1024
    chunk_overlap: int = 256
    max_workers: int = 2
    db_path: Union[str, Path] = "index.db"

@dataclass
class SearchConfig:
    """Search-related configuration."""
    
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    use_reranker: bool = False
    top_k_multiplier: int = 2
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
```

This modular approach enables:
- **Independent testing** of each component
- **Flexible model swapping** without code changes
- **Performance optimization** at the service level
- **Easy extension** with new search algorithms
- **Type-safe configuration** with automatic validation

## Component Structure

### Document Processor

The Document Processor handles:

- Language-specific text preparation
- Document chunking with configurable size and overlap
- Text normalization and cleaning
- Prefix handling for Ruri v3 embedding model

```python
@dataclass
class Chunk:
    """Represents a document chunk for indexing."""
    
    id: str
    path: Path
    title: str
    content: str
    chunk_index: int
    language: str
    created_at: datetime
    modified_at: datetime
    metadata: Dict[str, object]

class DocumentProcessor:
    """Processes documents into indexable chunks."""
    
    def process_document(
        self, 
        path: Union[str, Path], 
        content: str, 
        title: str, 
        language: str = "en",
        metadata: Optional[Dict[str, object]] = None
    ) -> List[Chunk]:
        """Process document into chunks with Ruri v3 prefixes."""
        chunks = self._chunk_text(content)
        return [
            Chunk(
                id=f"{path}:{i}",
                path=Path(path),
                title=title,
                content=chunk,
                chunk_index=i,
                language=language,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                metadata=metadata or {}
            )
            for i, chunk in enumerate(chunks)
        ]
```

### Change Detection

The indexer implements intelligent change detection for incremental updates:

```python
class FileChangeDetector:
    """Detects changes in files for incremental indexing."""
    
    def __init__(self, database: DatabaseService, batch_size: int = 1000):
        self.db = database
        self.batch_size = batch_size
    
    def detect_changes(
        self,
        file_paths: List[Path],
        strategy: str = "smart"
    ) -> ChangeResult:
        """Detect new, modified, and deleted files.
        
        Strategies:
        - timestamp: Fast, uses modification time
        - hash: Accurate, uses content hash  
        - smart: Balanced approach using both
        """

@dataclass
class ChangeResult:
    """Result of change detection analysis."""
    
    new_files: List[Path]
    modified_files: List[Path]
    deleted_files: List[Path]
    
    @property
    def total_changes(self) -> int:
        return len(self.new_files) + len(self.modified_files) + len(self.deleted_files)
    
    def has_changes(self) -> bool:
        return self.total_changes > 0
```

### Reranker Integration

The indexer supports advanced reranking for improved search accuracy:

```python
class Indexer:
    """Main indexer with reranking support."""
    
    def search(
        self,
        query: str,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        use_reranker: bool = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search with optional reranking."""
        # Use configuration default if not specified
        if use_reranker is None:
            use_reranker = self.config.use_reranker
        
        # Initial retrieval (multiple of limit for reranking)
        initial_limit = limit * self.config.search.top_k_multiplier if use_reranker else limit
        initial_results = self.search_engine.search(query, mode=mode, limit=initial_limit, **kwargs)
        
        if use_reranker and self.reranker_service:
            # Rerank for better accuracy
            return self.reranker_service.rerank(query, initial_results, limit)
        
        return initial_results[:limit]
```

### ONNX Optimization

Automatic model optimization for faster inference:

```python
class EmbeddingService:
    """Embedding generation with ONNX optimization."""
    
    def __init__(
        self, 
        model_name: str = "cl-nagoya/ruri-v3-30m",
        use_onnx: bool = True,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.batch_size = batch_size
        self.device = device
        self._model = None  # Lazy loading
        
    def generate_embeddings(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Generate embeddings with automatic ONNX optimization."""
        if self._model is None:
            self._model = self._load_model()
        
        return self._model.encode(texts, batch_size=self.batch_size)
```

## BM25 Indexer

The BM25 Indexer creates keyword-based search indexes:

- Japanese tokenization using MeCab morphological analyzer with fugashi
- Fallback to simple character-based tokenization when MeCab unavailable
- Inverted index construction for fast keyword lookups
- Document statistics calculation for BM25 scoring
- Support for incremental index updates

```python
class BM25Indexer:
    """BM25 indexer for building inverted index and computing statistics."""
    
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer_class: Optional[str] = None,
        use_stopwords: bool = False,
        min_doc_frequency: int = 1,
        store_positions: bool = True,
    ):
        self.k1 = k1
        self.b = b
        self.tokenizer = create_tokenizer(
            language=tokenizer_class or "ja",
            use_stopwords=use_stopwords
        )
        
    def index_chunks(
        self, 
        chunks: List[Chunk],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, int]:
        """Index a batch of chunks for BM25 search."""
        # Build inverted index with Japanese tokenization
        for chunk in chunks:
            term_frequencies = self.tokenizer.get_term_frequencies(chunk.content)
            self._update_inverted_index(chunk.id, term_frequencies)
        
        return self.get_collection_stats()
```

### Japanese Tokenizer

The Tokenizer provides Japanese text analysis:

- **MeCab Integration**: Uses fugashi wrapper for morphological analysis
- **Part-of-Speech Filtering**: Extracts content words (nouns, verbs, adjectives)
- **Stop Word Removal**: Filters particles and auxiliary words
- **Text Normalization**: Unicode and character width normalization
- **Fallback Support**: Simple tokenization when Japanese tools unavailable

```python
class TokenizerService:
    """Service for text tokenization with Japanese support."""
    
    def __init__(
        self,
        language: str = "ja",
        min_token_length: int = 2,
        use_stopwords: bool = False
    ):
        self.language = language
        self.min_token_length = min_token_length
        self.use_stopwords = use_stopwords
        self._tokenizer = None  # Lazy loading
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with language-specific processing."""
        if self._tokenizer is None:
            self._tokenizer = self._create_tokenizer()
        
        return self._tokenizer.tokenize(text)
    
    def get_term_frequencies(self, text: str) -> Dict[str, int]:
        """Get term frequency dictionary for BM25 indexing."""
        tokens = self.tokenize(text)
        return Counter(tokens)
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

-- File metadata for change detection and incremental indexing
CREATE TABLE file_metadata (
    path VARCHAR PRIMARY KEY,
    last_processed_at TIMESTAMP,
    file_modified_at TIMESTAMP,
    file_size INTEGER,
    content_hash VARCHAR,
    processing_status VARCHAR  -- 'pending', 'processing', 'completed', 'failed'
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

-- B-tree indexes for efficient lookups
CREATE INDEX idx_chunks_path ON chunks (path);
CREATE INDEX idx_embeddings_chunk_id ON embeddings (chunk_id);
CREATE INDEX idx_file_metadata_status ON file_metadata (processing_status);
```

## Index Types

Oboyu creates two complementary index types for comprehensive search capabilities:

### Vector Index (HNSW)

The vector index enables semantic similarity search:

- Uses DuckDB's VSS extension for vector similarity search
- Creates Hierarchical Navigable Small Worlds (HNSW) index
- Uses cosine similarity as distance metric
- **Best for**: conceptual queries, cross-language understanding, semantic matching

**Configurable parameters for accuracy vs. performance tradeoffs:**
- `ef_construction`: Controls index build quality (default: 128)
- `ef_search`: Controls search quality (default: 64)
- `M`: Number of bidirectional links (default: 16)
- `M0`: Level-0 connections (default: 2*M = 32)

### BM25 Index (Inverted Index)

The BM25 index enables precise keyword matching:

- Uses inverted index structure for fast term lookups
- Implements BM25 ranking algorithm with Japanese tokenization
- Stores term statistics and document frequencies
- **Best for**: exact keyword matching, specific terminology, technical documentation

**BM25 Parameters:**
- `k1`: Term frequency saturation parameter (default: 1.2)
- `b`: Length normalization parameter (default: 0.75)
- `min_token_length`: Minimum token length (default: 2)

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

## Configuration

The indexer is configured through a type-safe configuration system:

```python
# Default configuration values
IndexerConfig(
    model=ModelConfig(
        embedding_model="cl-nagoya/ruri-v3-30m",
        embedding_device="cpu",
        batch_size=64,
        max_seq_length=8192,
        use_onnx=True,
        reranker_model="cl-nagoya/ruri-reranker-small",
        use_reranker=False,
    ),
    processing=ProcessingConfig(
        chunk_size=1024,
        chunk_overlap=256,
        max_workers=2,
        db_path="index.db",
        ef_construction=128,
        ef_search=64,
        m=16,
    ),
    search=SearchConfig(
        vector_weight=0.7,
        bm25_weight=0.3,
        use_reranker=False,
        top_k_multiplier=2,
        bm25_k1=1.2,
        bm25_b=0.75,
        bm25_min_token_length=2,
    )
)
```

## Data Flow

1. **Document Input**: Receive documents from the Crawler
2. **Document Processing**: Process documents into chunks with appropriate prefix
3. **Change Detection**: Detect new, modified, and deleted files
4. **Embedding Generation**: Generate embeddings for new/modified chunks
5. **Database Storage**: Store chunks and embeddings in the database
6. **Index Creation**: Create and maintain HNSW and BM25 indexes
7. **Search Processing**: Process search queries with matching prefix and reranking

## Performance Considerations

- **Modular Services**: Independent optimization of each service
- **Lazy Loading**: Models are loaded only when needed
- **Batch Processing**: Efficient embedding generation in batches
- **ONNX Optimization**: 2-4x faster CPU inference for embeddings
- **Incremental Updates**: Only process changed files
- **Persistent Caching**: Avoid regenerating embeddings
- **Configurable Threading**: Adjustable worker count for parallel processing

## API Usage Examples

### Basic Usage

```python
from oboyu.indexer import Indexer
from oboyu.indexer.config import IndexerConfig

# Initialize with default configuration
indexer = Indexer()

# Index documents
from oboyu.crawler.crawler import CrawlerResult
crawler_result = CrawlerResult(...)
indexer.index_documents(crawler_result)

# Search with hybrid mode
results = indexer.search("検索クエリ", limit=10)
```

### Advanced Configuration

```python
from oboyu.indexer.config import IndexerConfig, ModelConfig, SearchConfig

# Custom configuration
config = IndexerConfig(
    model=ModelConfig(
        embedding_model="cl-nagoya/ruri-v3-30m",
        use_onnx=True,
        use_reranker=True,
    ),
    search=SearchConfig(
        vector_weight=0.6,
        bm25_weight=0.4,
        use_reranker=True,
    )
)

indexer = Indexer(config)

# Search with reranking
results = indexer.search(
    "機械学習の実装方法",
    limit=5,
    use_reranker=True
)
```

### Search Modes

```python
from oboyu.indexer.core.search_engine import SearchMode

# Vector search only
results = indexer.search("query", mode=SearchMode.VECTOR)

# BM25 search only  
results = indexer.search("query", mode=SearchMode.BM25)

# Hybrid search (default)
results = indexer.search("query", mode=SearchMode.HYBRID)
```

## Migration from Legacy Architecture

The indexer has been completely refactored from a monolithic design to a modular service-oriented architecture. Here are the key changes:

### Breaking Changes

- **Configuration**: Old flat configuration dict replaced with type-safe dataclasses
- **API**: Service methods now have more specific interfaces
- **Dependencies**: Some old methods deprecated in favor of new service APIs

### Legacy Compatibility

For backward compatibility, legacy import paths are maintained:

```python
# Legacy imports (still work)
from oboyu.indexer import Indexer
from oboyu.indexer.database import Database  # Points to DatabaseService
from oboyu.indexer.embedding import EmbeddingGenerator  # Points to EmbeddingService

# New preferred imports
from oboyu.indexer.indexer import Indexer
from oboyu.indexer.storage.database_service import DatabaseService
from oboyu.indexer.services.embedding import EmbeddingService
```

### Testing Improvements

The modular architecture significantly improves testability:

- **Unit Testing**: Each service can be tested independently
- **Mocking**: Services can be easily mocked for testing
- **Integration Testing**: Clear interfaces between services
- **Performance Testing**: Individual service performance can be measured

## Advanced Features

### Lazy Loading

All services implement lazy loading for optimal startup performance:

```python
# Fast initialization - no models loaded yet
indexer = Indexer()

# Models loaded only when first needed
results = indexer.search("query")  # <-- Models loaded here
```

### Service Injection

For advanced use cases, services can be customized or replaced:

```python
# Custom embedding service
custom_embedding_service = CustomEmbeddingService(...)

# Inject into indexer
indexer = Indexer(config)
indexer.embedding_service = custom_embedding_service
```

### Monitoring and Metrics

Services provide hooks for monitoring:

```python
# Progress callbacks for long operations
def progress_callback(current: int, total: int):
    print(f"Progress: {current}/{total}")

indexer.index_documents(documents, progress_callback=progress_callback)
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Check ONNX runtime installation and model cache permissions
2. **Japanese Tokenization**: Verify MeCab and fugashi installation for optimal performance
3. **Database Connection**: Ensure DuckDB and VSS extension are properly installed
4. **Memory Usage**: Adjust batch sizes if encountering out-of-memory errors

### Performance Tuning

1. **Batch Size**: Increase for better throughput, decrease for lower memory usage
2. **Worker Count**: Adjust `max_workers` based on CPU cores
3. **ONNX Optimization**: Enable for 2-4x faster CPU inference
4. **Index Parameters**: Tune HNSW parameters for accuracy vs. speed tradeoffs

### Debug Mode

Enable debug logging for detailed service information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed service logs will be printed
indexer = Indexer()
```

## Integration with Other Components

- Directly accepts CrawlerResult objects from the Crawler component
- Provides search functionality via the Indexer.search method
- Maintains backwards compatibility with the crawler's Japanese text processing
- Supports incremental updates for new documents
- **NEW**: Modular service architecture for easy integration and testing
- **NEW**: Type-safe configuration system for better maintainability
- **NEW**: Comprehensive error handling and logging throughout all services
- **NEW**: Performance monitoring capabilities built into each service