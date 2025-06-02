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
    def validate_chunk_overlap(cls, v: int, values: Any) -> int:
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
        # Track which deprecated options were found
        found_deprecated = []
        
        # Extract essential values from old config
        new_config_data = {
            "indexer": {},
            "crawler": {},
            "query": {}
        }
        
        # Handle indexer section
        if "indexer" in old_config:
            indexer_data = old_config["indexer"]
            
            # Keep essential indexer options
            if "db_path" in indexer_data:
                new_config_data["indexer"]["db_path"] = indexer_data["db_path"]
            if "chunk_size" in indexer_data:
                new_config_data["indexer"]["chunk_size"] = indexer_data["chunk_size"]
            if "chunk_overlap" in indexer_data:
                new_config_data["indexer"]["chunk_overlap"] = indexer_data["chunk_overlap"]
            if "embedding_model" in indexer_data:
                new_config_data["indexer"]["embedding_model"] = indexer_data["embedding_model"]
            if "use_reranker" in indexer_data:
                new_config_data["indexer"]["use_reranker"] = indexer_data["use_reranker"]
                
            # Check for deprecated options
            for key in indexer_data:
                deprecated_key = f"indexer.{key}"
                if deprecated_key in cls.DEPRECATED_MAPPINGS:
                    found_deprecated.append((deprecated_key, cls.DEPRECATED_MAPPINGS[deprecated_key]))
        
        # Handle crawler section
        if "crawler" in old_config:
            crawler_data = old_config["crawler"]
            
            # Keep essential crawler options
            if "include_patterns" in crawler_data:
                new_config_data["crawler"]["include_patterns"] = crawler_data["include_patterns"]
            if "exclude_patterns" in crawler_data:
                new_config_data["crawler"]["exclude_patterns"] = crawler_data["exclude_patterns"]
                
            # Check for deprecated options
            for key in crawler_data:
                deprecated_key = f"crawler.{key}"
                if deprecated_key in cls.DEPRECATED_MAPPINGS:
                    found_deprecated.append((deprecated_key, cls.DEPRECATED_MAPPINGS[deprecated_key]))
        
        # Handle query section
        if "query" in old_config:
            query_data = old_config["query"]
            
            # Keep essential query options
            if "default_mode" in query_data:
                new_config_data["query"]["default_mode"] = query_data["default_mode"]
            if "top_k" in query_data:
                new_config_data["query"]["top_k"] = query_data["top_k"]
                
            # Check for deprecated options
            for key in query_data:
                deprecated_key = f"query.{key}"
                if deprecated_key in cls.DEPRECATED_MAPPINGS:
                    found_deprecated.append((deprecated_key, cls.DEPRECATED_MAPPINGS[deprecated_key]))
        
        # Issue warnings for deprecated options
        if found_deprecated:
            warning_message = "The following configuration options have been deprecated and removed:\n"
            for option, replacement in found_deprecated:
                warning_message += f"  - {option}: {replacement}\n"
            warning_message += "\nYour configuration will be automatically migrated. Update your config file to remove these warnings."
            warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
        
        return SimplifiedConfig.from_dict(new_config_data)


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
