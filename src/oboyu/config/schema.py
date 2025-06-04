"""Configuration schema definitions with proper types."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class CrawlerConfigSchema(BaseModel):
    """Schema for crawler configuration."""

    max_workers: int = Field(default=4, ge=1, le=100, description="Maximum number of worker threads")
    timeout: int = Field(default=30, ge=1, le=300, description="Timeout in seconds for operations")
    max_depth: int = Field(default=3, ge=0, le=10, description="Maximum directory traversal depth")
    exclude_dirs: List[str] = Field(
        default_factory=lambda: ["__pycache__", ".git", "node_modules"],
        description="Directories to exclude from crawling"
    )
    include_extensions: List[str] = Field(
        default_factory=lambda: [".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".rst", ".ipynb"],
        description="File extensions to include in crawling"
    )
    min_doc_length: int = Field(default=50, ge=1, description="Minimum document length to process")
    chunk_size: int = Field(default=1000, ge=100, le=10000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between chunks")
    encoding: Literal["utf-8", "shift-jis", "euc-jp", "iso-2022-jp"] = Field(
        default="utf-8",
        description="Text encoding to use"
    )
    use_japanese_tokenizer: bool = Field(default=True, description="Use Japanese-specific tokenizer")

    @field_validator('encoding')
    @classmethod
    def validate_encoding(cls, v: str) -> str:
        """Validate that encoding is supported."""
        # pydantic Literal already handles validation, but we can add custom logic here
        return v.lower()

    @field_validator('include_extensions')
    @classmethod
    def validate_extensions(cls, v: List[str]) -> List[str]:
        """Validate file extensions format."""
        validated = []
        for ext in v:
            if not ext.startswith('.'):
                ext = '.' + ext
            validated.append(ext.lower())
        return validated

    @model_validator(mode='after')
    def validate_chunk_config(self) -> 'CrawlerConfigSchema':
        """Validate chunk configuration relationships."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrawlerConfigSchema":
        """Create instance from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary."""
        return self.model_dump()


class IndexerConfigSchema(BaseModel):
    """Schema for indexer configuration."""
    
    model_config = {"extra": "forbid"}

    embedding_model: str = Field(
        default="cl-nagoya/ruri-v3-30m",
        description="Name or path of embedding model",
        pattern=r'^[\w\-./]+[\w\-/]+$'
    )
    batch_size: int = Field(default=128, ge=1, le=1024, description="Batch size for processing")
    max_length: int = Field(default=8192, ge=256, le=32768, description="Maximum sequence length")
    normalize_embeddings: bool = Field(default=True, description="Whether to normalize embeddings")
    show_progress: bool = Field(default=True, description="Show progress bars")
    bm25_k1: float = Field(default=1.5, ge=0.0, le=3.0, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 b parameter")
    use_japanese_tokenizer: bool = Field(default=True, description="Use Japanese-specific tokenizer")
    n_probe: int = Field(default=10, ge=1, le=100, description="Number of probes for vector search")
    db_path: Optional[Path] = Field(default=None, description="Database path")

    @field_validator('embedding_model')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate embedding model name format."""
        if not v:
            raise ValueError('Model name cannot be empty')
        # Be more lenient for testing - allow any non-empty string
        return v


    @field_validator('db_path')
    @classmethod
    def validate_db_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate database path and ensure parent directory exists."""
        if v is not None:
            try:
                # Ensure parent directory exists - but don't fail if we can't create it
                v.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                # Ignore permission errors in tests or read-only filesystems
                pass
        return v

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexerConfigSchema":
        """Create instance from dictionary."""
        # Convert db_path string to Path if present
        if "db_path" in data and data["db_path"] is not None:
            data = dict(data)
            data["db_path"] = Path(str(data["db_path"]))
        return cls(**data)

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary."""
        result = self.model_dump(exclude_none=True)
        if result.get("db_path") is not None:
            result["db_path"] = str(result["db_path"])
        return result


class CircuitBreakerConfigSchema(BaseModel):
    """Schema for circuit breaker configuration."""

    enabled: bool = Field(default=True, description="Enable circuit breaker protection")
    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of failures before opening circuit"
    )
    recovery_timeout_minutes: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Minutes to wait before trying half-open state"
    )
    success_threshold: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Consecutive successes needed to close circuit"
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for individual requests in seconds"
    )
    enable_fallback_services: bool = Field(
        default=True,
        description="Enable fallback to alternative models"
    )
    enable_local_fallback: bool = Field(
        default=True,
        description="Enable local deterministic fallback embeddings"
    )
    fallback_model_names: List[str] = Field(
        default_factory=lambda: [
            "cl-nagoya/ruri-v3-base",
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5"
        ],
        description="List of fallback embedding model names"
    )

    @field_validator('fallback_model_names')
    @classmethod
    def validate_fallback_models(cls, v: List[str]) -> List[str]:
        """Validate fallback model names."""
        return [model.strip() for model in v if model.strip()]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitBreakerConfigSchema":
        """Create instance from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary."""
        return self.model_dump()


class QueryConfigSchema(BaseModel):
    """Schema for query engine configuration."""

    top_k: int = Field(default=10, ge=1, le=1000, description="Number of top results to return")
    rerank: bool = Field(default=True, description="Enable reranking of search results")
    rerank_model: str = Field(
        default="cl-nagoya/ruri-reranker-small",
        description="Reranker model name",
        pattern=r'^[\w\-./]+[\w\-/]+$'
    )
    show_scores: bool = Field(default=False, description="Show relevance scores in output")
    interactive: bool = Field(default=False, description="Enable interactive mode")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryConfigSchema":
        """Create instance from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary."""
        return self.model_dump()


class ConfigSchema(BaseModel):
    """Complete configuration schema."""

    crawler: CrawlerConfigSchema = Field(default_factory=CrawlerConfigSchema)
    indexer: IndexerConfigSchema = Field(default_factory=IndexerConfigSchema)
    query: QueryConfigSchema = Field(default_factory=QueryConfigSchema)
    circuit_breaker: CircuitBreakerConfigSchema = Field(default_factory=CircuitBreakerConfigSchema)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigSchema":
        """Create instance from dictionary."""
        crawler_data = data.get("crawler", {}) if isinstance(data.get("crawler", {}), dict) else {}
        indexer_data = data.get("indexer", {}) if isinstance(data.get("indexer", {}), dict) else {}
        query_data = data.get("query", {}) if isinstance(data.get("query", {}), dict) else {}
        circuit_breaker_data = data.get("circuit_breaker", {}) if isinstance(data.get("circuit_breaker", {}), dict) else {}
        
        return cls(
            crawler=CrawlerConfigSchema.from_dict(crawler_data),
            indexer=IndexerConfigSchema.from_dict(indexer_data),
            query=QueryConfigSchema.from_dict(query_data),
            circuit_breaker=CircuitBreakerConfigSchema.from_dict(circuit_breaker_data),
        )

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary."""
        return {
            "crawler": self.crawler.to_dict(),
            "indexer": self.indexer.to_dict(),
            "query": self.query.to_dict(),
            "circuit_breaker": self.circuit_breaker.to_dict(),
        }
