"""Configuration schema definitions with proper types."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CrawlerConfigSchema:
    """Schema for crawler configuration."""

    max_workers: int = 4
    timeout: int = 30
    max_depth: int = 3
    exclude_dirs: List[str] = field(default_factory=lambda: ["__pycache__", ".git", "node_modules"])
    include_extensions: List[str] = field(default_factory=lambda: [".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".rst", ".ipynb"])
    min_doc_length: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200
    encoding: str = "utf-8"
    use_japanese_tokenizer: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrawlerConfigSchema":
        """Create instance from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_workers": self.max_workers,
            "timeout": self.timeout,
            "max_depth": self.max_depth,
            "exclude_dirs": self.exclude_dirs,
            "include_extensions": self.include_extensions,
            "min_doc_length": self.min_doc_length,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "encoding": self.encoding,
            "use_japanese_tokenizer": self.use_japanese_tokenizer,
        }


@dataclass
class IndexerConfigSchema:
    """Schema for indexer configuration."""

    embedding_model: str = "cl-nagoya/ruri-v3-30m"
    batch_size: int = 128
    max_length: int = 8192
    normalize_embeddings: bool = True
    show_progress: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    use_japanese_tokenizer: bool = True
    n_probe: int = 10
    db_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexerConfigSchema":
        """Create instance from dictionary."""
        # Convert db_path string to Path if present
        if "db_path" in data and data["db_path"] is not None:
            data = data.copy()
            data["db_path"] = Path(data["db_path"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "embedding_model": self.embedding_model,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "normalize_embeddings": self.normalize_embeddings,
            "show_progress": self.show_progress,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "use_japanese_tokenizer": self.use_japanese_tokenizer,
            "n_probe": self.n_probe,
        }
        if self.db_path is not None:
            result["db_path"] = str(self.db_path)
        return result


@dataclass
class QueryConfigSchema:
    """Schema for query engine configuration."""

    top_k: int = 10
    rerank: bool = True
    rerank_model: str = "cl-nagoya/ruri-reranker-small"
    show_scores: bool = False
    interactive: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryConfigSchema":
        """Create instance from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "top_k": self.top_k,
            "rerank": self.rerank,
            "rerank_model": self.rerank_model,
            "show_scores": self.show_scores,
            "interactive": self.interactive,
        }


@dataclass
class ConfigSchema:
    """Complete configuration schema."""

    crawler: CrawlerConfigSchema = field(default_factory=CrawlerConfigSchema)
    indexer: IndexerConfigSchema = field(default_factory=IndexerConfigSchema)
    query: QueryConfigSchema = field(default_factory=QueryConfigSchema)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigSchema":
        """Create instance from dictionary."""
        return cls(
            crawler=CrawlerConfigSchema.from_dict(data.get("crawler", {})),
            indexer=IndexerConfigSchema.from_dict(data.get("indexer", {})),
            query=QueryConfigSchema.from_dict(data.get("query", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "crawler": self.crawler.to_dict(),
            "indexer": self.indexer.to_dict(),
            "query": self.query.to_dict(),
        }
