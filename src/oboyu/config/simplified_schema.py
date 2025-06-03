"""Simplified configuration schema with only essential options.

This module provides a streamlined configuration system that removes
~60% of the current configuration options while maintaining all functionality
through sensible defaults and auto-optimization.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator


class SimplifiedIndexerConfig(BaseModel):
    """Simplified indexer configuration with only essential options."""
    
    db_path: str = Field(
        default="~/.oboyu/index.db",
        description="Database path for storing index data"
    )
    chunk_size: int = Field(
        default=1024,
        ge=256,
        le=8192,
        description="Size of text chunks for processing"
    )
    chunk_overlap: int = Field(
        default=256,
        ge=0,
        description="Overlap between consecutive chunks"
    )
    embedding_model: str = Field(
        default="cl-nagoya/ruri-v3-30m",
        description="Embedding model to use for vector search"
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to use reranking for better search quality"
    )

    @field_validator('chunk_overlap')
    @classmethod
    def validate_chunk_overlap(cls, v: int) -> int:
        """Ensure chunk_overlap is less than chunk_size."""
        # Note: This validation is also done at the model level in model_validator
        return v

    @model_validator(mode='after')
    def validate_chunk_config(self) -> 'SimplifiedIndexerConfig':
        """Validate chunk configuration relationships."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self

    @field_validator('db_path')
    @classmethod
    def validate_db_path(cls, v: str) -> str:
        """Expand user home directory in path."""
        return str(Path(v).expanduser())


class SimplifiedCrawlerConfig(BaseModel):
    """Simplified crawler configuration with only essential options."""
    
    include_patterns: List[str] = Field(
        default_factory=lambda: ["*.txt", "*.md", "*.py", "*.rst", "*.ipynb"],
        description="File patterns to include in indexing"
    )
    exclude_patterns: List[str] = Field(
        default_factory=lambda: ["*/node_modules/*", "*/.git/*", "*/venv/*", "*/__pycache__/*"],
        description="File patterns to exclude from indexing"
    )

    @field_validator('include_patterns', 'exclude_patterns')
    @classmethod
    def validate_patterns(cls, v: List[str]) -> List[str]:
        """Ensure patterns are properly formatted."""
        return [pattern.strip() for pattern in v if pattern.strip()]


class SimplifiedQueryConfig(BaseModel):
    """Simplified query configuration with only essential options."""
    
    default_mode: str = Field(
        default="hybrid",
        description="Default search mode (vector, bm25, hybrid)"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return by default"
    )

    @field_validator('default_mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate search mode."""
        valid_modes = {"vector", "bm25", "hybrid"}
        if v not in valid_modes:
            raise ValueError(f"default_mode must be one of: {', '.join(valid_modes)}")
        return v


class SimplifiedConfig(BaseModel):
    """Complete simplified configuration."""
    
    indexer: SimplifiedIndexerConfig = Field(default_factory=SimplifiedIndexerConfig)
    crawler: SimplifiedCrawlerConfig = Field(default_factory=SimplifiedCrawlerConfig)
    query: SimplifiedQueryConfig = Field(default_factory=SimplifiedQueryConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimplifiedConfig":
        """Create simplified config from dictionary with validation."""
        return cls(
            indexer=SimplifiedIndexerConfig(**data.get("indexer", {})),
            crawler=SimplifiedCrawlerConfig(**data.get("crawler", {})),
            query=SimplifiedQueryConfig(**data.get("query", {}))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "indexer": self.indexer.model_dump(),
            "crawler": self.crawler.model_dump(),
            "query": self.query.model_dump()
        }


class BackwardCompatibilityMapper:
    """Maps old configuration options to new simplified structure."""
    
    # Mapping of deprecated options to their replacements or auto-optimized values
    DEPRECATED_MAPPINGS = {
        # Indexer deprecations
        "indexer.batch_size": "auto-optimized based on system memory",
        "indexer.max_workers": "auto-optimized based on CPU cores",
        "indexer.ef_construction": "auto-optimized for performance",
        "indexer.ef_search": "auto-optimized for performance",
        "indexer.m": "auto-optimized for performance",
        "indexer.bm25_k1": "uses proven default of 1.2",
        "indexer.bm25_b": "uses proven default of 0.75",
        "indexer.bm25_min_token_length": "uses proven default of 2",
        "indexer.reranker_batch_size": "auto-optimized based on system",
        "indexer.reranker_max_length": "auto-optimized for model",
        "indexer.reranker_device": "always uses CPU for consistency",
        "indexer.use_onnx": "always enabled for performance",
        "indexer.onnx_quantization": "auto-optimized for best performance",
        
        # Crawler deprecations
        "crawler.max_workers": "auto-optimized based on CPU cores",
        "crawler.timeout": "auto-optimized for file operations",
        "crawler.max_depth": "auto-optimized to prevent excessive recursion",
        "crawler.min_doc_length": "auto-optimized for meaningful content",
        "crawler.max_file_size": "hard-coded to 10MB for safety",
        "crawler.follow_symlinks": "hard-coded to false for safety",
        "crawler.encoding": "auto-detected using proven algorithms",
        "crawler.use_japanese_tokenizer": "auto-detected based on content",
        
        # Query deprecations
        "query.vector_weight": "replaced by RRF algorithm",
        "query.bm25_weight": "replaced by RRF algorithm",
        "query.rrf_k": "auto-optimized default of 60",
        "query.show_scores": "runtime option only",
        "query.interactive": "runtime option only",
        "query.rerank_model": "uses indexer.embedding_model setting",
        "query.snippet_length": "runtime option only",
        "query.highlight_matches": "runtime option only",
        "query.language_filter": "runtime option only",
        "query.reranker_threshold": "auto-optimized for best results",
    }

    @classmethod
    def migrate_config(cls, old_config: Dict[str, Any]) -> SimplifiedConfig:
        """Migrate old configuration to simplified structure with warnings."""
        found_deprecated: list[tuple[str, str]] = []
        new_config_data: dict[str, Any] = {"indexer": {}, "crawler": {}, "query": {}}
        
        cls._migrate_indexer_section(old_config, new_config_data, found_deprecated)
        cls._migrate_crawler_section(old_config, new_config_data, found_deprecated)
        cls._migrate_query_section(old_config, new_config_data, found_deprecated)
        cls._warn_deprecated_options(found_deprecated)
        
        return SimplifiedConfig.from_dict(new_config_data)
    
    @classmethod
    def _migrate_indexer_section(cls, old_config: Dict[str, Any],
                                new_config_data: Dict[str, Any],
                                found_deprecated: list[tuple[str, str]]) -> None:
        """Migrate indexer section of configuration."""
        if "indexer" not in old_config:
            return
            
        indexer_data = old_config["indexer"]
        essential_keys = ["db_path", "chunk_size", "chunk_overlap", "embedding_model", "use_reranker"]
        
        for key in essential_keys:
            if key in indexer_data:
                new_config_data["indexer"][key] = indexer_data[key]
        
        cls._check_deprecated_options(indexer_data, "indexer", found_deprecated)
    
    @classmethod
    def _migrate_crawler_section(cls, old_config: Dict[str, Any],
                                new_config_data: Dict[str, Any],
                                found_deprecated: list[tuple[str, str]]) -> None:
        """Migrate crawler section of configuration."""
        if "crawler" not in old_config:
            return
            
        crawler_data = old_config["crawler"]
        essential_keys = ["include_patterns", "exclude_patterns"]
        
        for key in essential_keys:
            if key in crawler_data:
                new_config_data["crawler"][key] = crawler_data[key]
        
        cls._check_deprecated_options(crawler_data, "crawler", found_deprecated)
    
    @classmethod
    def _migrate_query_section(cls, old_config: Dict[str, Any],
                              new_config_data: Dict[str, Any],
                              found_deprecated: list[tuple[str, str]]) -> None:
        """Migrate query section of configuration."""
        if "query" not in old_config:
            return
            
        query_data = old_config["query"]
        essential_keys = ["default_mode", "top_k"]
        
        for key in essential_keys:
            if key in query_data:
                new_config_data["query"][key] = query_data[key]
        
        cls._check_deprecated_options(query_data, "query", found_deprecated)
    
    @classmethod
    def _check_deprecated_options(cls, section_data: Dict[str, Any],
                                 section_name: str,
                                 found_deprecated: list[tuple[str, str]]) -> None:
        """Check for deprecated options in a config section."""
        for key in section_data:
            deprecated_key = f"{section_name}.{key}"
            if deprecated_key in cls.DEPRECATED_MAPPINGS:
                found_deprecated.append((deprecated_key, cls.DEPRECATED_MAPPINGS[deprecated_key]))
    
    @classmethod
    def _warn_deprecated_options(cls, found_deprecated: list[tuple[str, str]]) -> None:
        """Issue warnings for deprecated options."""
        if not found_deprecated:
            return
            
        warning_message = "The following configuration options have been deprecated and removed:\n"
        for option, replacement in found_deprecated:
            warning_message += f"  - {option}: {replacement}\n"
        warning_message += "\nYour configuration will be automatically migrated. Update your config file to remove these warnings."
        warnings.warn(warning_message, DeprecationWarning, stacklevel=2)


# Auto-optimization functions for removed parameters
class AutoOptimizer:
    """Automatically optimizes parameters that were removed from configuration."""
    
    @staticmethod
    def get_optimal_batch_size() -> int:
        """Auto-optimize batch size based on available memory."""
        import psutil
        
        # Get available memory in GB
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Optimize batch size based on memory
        if available_memory_gb >= 16:
            return 128
        elif available_memory_gb >= 8:
            return 64
        elif available_memory_gb >= 4:
            return 32
        else:
            return 16
    
    @staticmethod
    def get_optimal_max_workers() -> int:
        """Auto-optimize worker count based on CPU cores."""
        import os
        
        cpu_count = os.cpu_count() or 4
        # Use number of cores but cap at reasonable limit
        return min(cpu_count, 8)
    
    @staticmethod
    def get_optimal_hnsw_params() -> Dict[str, int]:
        """Auto-optimize HNSW parameters for best performance."""
        return {
            "ef_construction": 128,
            "ef_search": 64,
            "m": 16,
            "m0": 32
        }
    
    @staticmethod
    def get_optimal_bm25_params() -> Dict[str, float]:
        """Get proven BM25 parameters."""
        return {
            "k1": 1.2,
            "b": 0.75,
            "min_token_length": 2
        }
