# Module Dependencies Analysis

## Overview
This analysis identifies cross-module dependencies that access internal components (submodules) rather than facades.

## Deep Dependencies by Module

### 1. CLI Module
The CLI module has extensive deep dependencies on other modules:

**Dependencies on Indexer internals:**
- `indexer.config.indexer_config.IndexerConfig`
- `indexer.config.model_config.ModelConfig`
- `indexer.config.processing_config.ProcessingConfig`
- `indexer.config.search_config.SearchConfig`

**Dependencies on Crawler internals:**
- `crawler.config.load_default_config`
- `crawler.crawler.Crawler`
- `crawler.discovery.discover_documents`
- `crawler.extractor._detect_language`

**Dependencies on Retriever internals:**
- `retriever.retriever.Retriever`
- `retriever.search.search_mode.SearchMode`
- `retriever.search.search_context.ContextBuilder, SettingSource`
- `retriever.search.search_result.SearchResult`

**Dependencies on MCP internals:**
- `mcp.context.db_path_global, mcp`

### 2. Config Module
✅ No dependencies on other modules - Good isolation!

### 3. Crawler Module
✅ No dependencies on other modules - Good isolation!

### 4. Indexer Module
The Indexer module has significant dependencies on other modules:

**Dependencies on Retriever internals:**
- `retriever.search.search_result.SearchResult`
- `retriever.search.bm25_search.BM25Search`
- `retriever.search.vector_search.VectorSearch`
- `retriever.search.hybrid_search_combiner.HybridSearchCombiner`
- `retriever.search.mode_router.SearchModeRouter`
- `retriever.search.result_merger.ResultMerger`
- `retriever.search.score_normalizer.ScoreNormalizer`
- `retriever.search.search_filters.SearchFilters`
- `retriever.search.search_mode.SearchMode`
- `retriever.services.tokenizer.TokenizerService, create_tokenizer`
- `retriever.storage.database_search_service.DatabaseSearchService`

**Dependencies on Crawler internals:**
- `crawler.crawler.CrawlerResult`
- `crawler.services.encoding_detector.EncodingDetector`

### 5. MCP Module
The MCP module has dependencies on multiple modules:

**Dependencies on Indexer internals:**
- `indexer.config.indexer_config.IndexerConfig`
- `indexer.config.model_config.ModelConfig`
- `indexer.config.processing_config.ProcessingConfig`
- `indexer.config.search_config.SearchConfig`

**Dependencies on Retriever internals:**
- `retriever.retriever.Retriever`
- `retriever.search.search_filters.SearchFilters`
- `retriever.search.snippet_processor.SnippetProcessor`
- `retriever.search.snippet_types.SnippetConfig`

**Dependencies on Crawler internals:**
- `crawler.crawler.Crawler`

### 6. Retriever Module
The Retriever module has dependencies on Indexer:

**Dependencies on Indexer internals:**
- `indexer.config.indexer_config.IndexerConfig`
- `indexer.services.embedding.EmbeddingService`
- `indexer.storage.database_service.DatabaseService`
- `indexer.storage.index_manager.HNSWIndexParams`
- `indexer.data.stop_words.DEFAULT_JAPANESE_STOP_WORDS`

## Analysis Summary

### Good Isolation
- ✅ **Config**: No dependencies on other modules
- ✅ **Crawler**: No dependencies on other modules

### Circular Dependencies
- ❌ **Indexer ↔ Retriever**: These modules have circular dependencies
  - Indexer imports retriever search components
  - Retriever imports indexer storage and services

### Heavy Coupling
- ❌ **CLI**: Depends on internals of all other modules (indexer, crawler, retriever, mcp)
- ❌ **MCP**: Depends on internals of indexer, retriever, and crawler

## Recommendations for Refactoring

### 1. Create Facade Interfaces
Each module should expose a clean facade interface:

```python
# indexer/__init__.py
from .indexer import Indexer
from .config import IndexerConfig  # Move config to module root
__all__ = ['Indexer', 'IndexerConfig']

# retriever/__init__.py
from .retriever import Retriever
from .models import SearchResult, SearchMode
__all__ = ['Retriever', 'SearchResult', 'SearchMode']
```

### 2. Move Shared Components
- Move search-related components that both indexer and retriever need to a shared location
- Consider creating a `search` module at the top level for shared search functionality

### 3. Dependency Inversion
- Use interfaces/protocols to break circular dependencies
- Indexer should not directly import retriever components
- Retriever should not directly import indexer storage

### 4. Configuration Consolidation
- Move all config classes to their module roots or to a central config module
- Avoid deep imports like `indexer.config.model_config.ModelConfig`

### 5. Service Layer
- CLI and MCP should interact with other modules through service interfaces
- Avoid direct access to internal components

## Detailed Problematic Patterns

### 1. Configuration Deep Imports
Many modules are importing configuration classes from deep paths:
```python
# Current (BAD)
from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.config.model_config import ModelConfig

# Proposed (GOOD)
from oboyu.indexer import IndexerConfig, ModelConfig
```

### 2. Circular Dependency: Indexer ↔ Retriever
**Problem**: Indexer uses retriever's search components, while retriever uses indexer's storage.

```python
# indexer/core/search_engine.py imports:
from oboyu.retriever.search.bm25_search import BM25Search
from oboyu.retriever.search.vector_search import VectorSearch

# retriever/orchestrators/service_registry.py imports:
from oboyu.indexer.storage.database_service import DatabaseService
from oboyu.indexer.services.embedding import EmbeddingService
```

**Solution**: Extract shared components to a common module or use dependency injection.

### 3. Cross-Module Service Usage
```python
# indexer/algorithm/bm25_indexer.py
from oboyu.retriever.services.tokenizer import TokenizerService

# retriever/services/tokenizer.py
from oboyu.indexer.data.stop_words import DEFAULT_JAPANESE_STOP_WORDS
```

**Issue**: Services are being shared across modules without clear ownership.

### 4. Result Type Sharing
```python
# Used by both indexer and retriever
from oboyu.retriever.search.search_result import SearchResult
```

**Issue**: Common types should be in a shared location, not owned by one module.

## Proposed Module Structure

```
src/oboyu/
├── common/          # Shared utilities
├── config/          # All configuration
├── types/           # Shared types and models
│   ├── search.py   # SearchResult, SearchMode, etc.
│   └── filters.py  # SearchFilters
├── services/        # Shared services
│   ├── tokenizer.py
│   └── embedding.py
├── crawler/
│   └── __init__.py  # Exposes: Crawler, CrawlerResult
├── indexer/
│   └── __init__.py  # Exposes: Indexer, IndexerConfig
├── retriever/
│   └── __init__.py  # Exposes: Retriever
├── cli/
│   └── __init__.py  # Uses only facades
└── mcp/
    └── __init__.py  # Uses only facades
```

## Priority Refactoring Tasks

1. **Extract shared types** (SearchResult, SearchMode, SearchFilters) to a `types` module
2. **Move shared services** (TokenizerService, EmbeddingService) to a `services` module
3. **Create facade interfaces** for each module exposing only necessary APIs
4. **Break circular dependency** between indexer and retriever
5. **Simplify configuration imports** by exposing configs at module root
