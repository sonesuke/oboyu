"""Configuration creation utilities for index command."""

from pathlib import Path
from typing import Any, List, Optional

from oboyu.common.config import ConfigManager
from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.config.model_config import ModelConfig
from oboyu.indexer.config.processing_config import ProcessingConfig
from oboyu.indexer.config.search_config import SearchConfig


def create_crawler_config(
    config_manager: ConfigManager,
    recursive: Optional[bool],
    max_depth: Optional[int],
    include_patterns: Optional[List[str]],
    exclude_patterns: Optional[List[str]],
) -> dict[str, Any]:
    """Create crawler configuration from config data and command-line options."""
    crawler_config_dict = config_manager.get_section("crawler")

    if recursive is not None:
        crawler_config_dict["depth"] = 0 if not recursive else (max_depth or 10)
    elif max_depth is not None:
        crawler_config_dict["depth"] = max_depth

    if include_patterns:
        crawler_config_dict["include_patterns"] = include_patterns
    if exclude_patterns:
        crawler_config_dict["exclude_patterns"] = exclude_patterns

    return dict(crawler_config_dict)


def create_model_config(config_dict: dict[str, Any]) -> ModelConfig:
    """Create ModelConfig from configuration dictionary."""
    return ModelConfig(
        embedding_model=config_dict.get("embedding_model", "cl-nagoya/ruri-v3-30m"),
        use_onnx=config_dict.get("use_onnx", True),
        reranker_model=config_dict.get("reranker_model", "cl-nagoya/ruri-reranker-small"),
        reranker_use_onnx=config_dict.get("reranker_use_onnx", True),
    )


def create_indexer_config(
    config_manager: ConfigManager,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    embedding_model: Optional[str],
    db_path: Optional[Path],
) -> dict[str, Any]:
    """Create indexer configuration from config data and command-line options."""
    cli_overrides: dict[str, Any] = {}
    if chunk_size is not None:
        cli_overrides["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        cli_overrides["chunk_overlap"] = chunk_overlap
    if embedding_model is not None:
        cli_overrides["embedding_model"] = embedding_model
    if db_path is not None:
        cli_overrides["db_path"] = str(db_path)

    indexer_config_dict = config_manager.merge_cli_overrides("indexer", cli_overrides)
    if "db_path" not in indexer_config_dict:
        resolved_db_path = config_manager.resolve_db_path(None, indexer_config_dict)
        indexer_config_dict["db_path"] = str(resolved_db_path)
    if db_path:
        indexer_config_dict["database_path"] = str(db_path)
    elif "database_path" not in indexer_config_dict:
        indexer_config_dict["database_path"] = indexer_config_dict.get("db_path", str(config_manager.resolve_db_path(None, indexer_config_dict)))
    return dict(indexer_config_dict)


def build_indexer_config(indexer_config_dict: dict[str, Any]) -> IndexerConfig:
    """Build complete IndexerConfig from configuration dictionary."""
    model_config = create_model_config(indexer_config_dict)
    
    search_config = SearchConfig(
        bm25_k1=indexer_config_dict.get("bm25_k1", 1.2),
        bm25_b=indexer_config_dict.get("bm25_b", 0.75),
        use_reranker=indexer_config_dict.get("use_reranker", True),
        top_k_multiplier=indexer_config_dict.get("reranker_top_k_multiplier", 3),
    )
    
    processing_config = ProcessingConfig(
        chunk_size=indexer_config_dict.get("chunk_size", 1024),
        chunk_overlap=indexer_config_dict.get("chunk_overlap", 256),
        db_path=Path(indexer_config_dict.get("db_path", "oboyu.db")),
        max_workers=indexer_config_dict.get("max_workers", 4),
        ef_construction=indexer_config_dict.get("ef_construction", 128),
        ef_search=indexer_config_dict.get("ef_search", 64),
        m=indexer_config_dict.get("m", 16),
        m0=indexer_config_dict.get("m0"),
    )
    
    return IndexerConfig(
        model=model_config,
        search=search_config,
        processing=processing_config,
    )


def build_status_indexer_config(indexer_config_dict: dict[str, Any]) -> IndexerConfig:
    """Build IndexerConfig for status operations (simplified)."""
    model_config = create_model_config(indexer_config_dict)
    
    search_config = SearchConfig(
        bm25_k1=indexer_config_dict.get("bm25_k1", 1.2),
        bm25_b=indexer_config_dict.get("bm25_b", 0.75),
        use_reranker=indexer_config_dict.get("use_reranker", False),
    )
    
    processing_config = ProcessingConfig(
        chunk_size=indexer_config_dict.get("chunk_size", 1024),
        chunk_overlap=indexer_config_dict.get("chunk_overlap", 256),
        db_path=Path(indexer_config_dict.get("db_path", "oboyu.db")),
    )
    
    return IndexerConfig(
        model=model_config,
        search=search_config,
        processing=processing_config,
    )
