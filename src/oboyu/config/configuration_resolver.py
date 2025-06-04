"""Configuration resolver for creating immutable configurations with proper precedence."""

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from ..common.types.search_mode import SearchMode
from .configuration_builder import ConfigSource, ConfigurationBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedSearchConfig:
    """Immutable resolved search configuration."""

    query: str
    mode: SearchMode
    top_k: int
    use_reranker: bool
    reranker_model: str
    reranker_top_k: int
    
    # Source tracking for debugging
    sources: Dict[str, ConfigSource] = None
    
    def with_reranker(self, enabled: bool) -> 'ResolvedSearchConfig':
        """Create a new config with reranker explicitly set."""
        return replace(self, use_reranker=enabled)
    
    def with_top_k(self, top_k: int) -> 'ResolvedSearchConfig':
        """Create a new config with top_k explicitly set."""
        return replace(self, top_k=top_k)


@dataclass(frozen=True)
class ResolvedIndexerConfig:
    """Immutable resolved indexer configuration."""

    use_reranker: bool
    reranker_model: str
    reranker_device: str
    reranker_use_onnx: bool
    chunk_size: int
    chunk_overlap: int
    
    # Source tracking for debugging
    sources: Dict[str, ConfigSource] = None


class ConfigurationResolver:
    """Resolves configuration from multiple sources with clear precedence."""
    
    # System defaults - lowest precedence
    SYSTEM_DEFAULTS = {
        # Search/Query defaults
        "search.top_k": 10,
        "search.use_reranker": True,  # Default to True for better search quality
        "search.reranker_model": "cl-nagoya/ruri-reranker-small",
        "search.reranker_top_k": 3,
        
        # Indexer defaults
        "indexer.use_reranker": False,  # Default to False for indexing (not needed)
        "indexer.reranker_model": "cl-nagoya/ruri-reranker-small",
        "indexer.reranker_device": "cpu",
        "indexer.reranker_use_onnx": False,
        "indexer.chunk_size": 1000,
        "indexer.chunk_overlap": 200,
    }
    
    def __init__(self) -> None:
        """Initialize configuration resolver."""
        self.builder = ConfigurationBuilder()
        self._apply_system_defaults()
    
    def _apply_system_defaults(self) -> None:
        """Apply system defaults."""
        for key, value in self.SYSTEM_DEFAULTS.items():
            self.builder.set_default(key, value)
    
    def load_from_dict(self, config_dict: Dict[str, Any], source: ConfigSource = ConfigSource.FILE) -> None:
        """Load configuration from a dictionary (e.g., parsed YAML)."""
        # Handle nested configuration
        def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
            items: list[tuple[str, Any]] = []
            for k, v in d.items():
                new_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_config = flatten_dict(config_dict)
        
        # Map legacy configuration keys to new ones
        mappings = {
            # Handle both 'rerank' and 'use_reranker' for backward compatibility
            "query.rerank": "search.use_reranker",
            "query.use_reranker": "search.use_reranker",
            "query.rerank_model": "search.reranker_model",
            "query.reranker_model": "search.reranker_model",
            "query.top_k": "search.top_k",
            
            # Indexer mappings
            "indexer.use_reranker": "indexer.use_reranker",
            "indexer.reranker_model": "indexer.reranker_model",
            "indexer.reranker_device": "indexer.reranker_device",
            "indexer.reranker_use_onnx": "indexer.reranker_use_onnx",
        }
        
        for old_key, new_key in mappings.items():
            if old_key in flat_config:
                value = flat_config[old_key]
                if source == ConfigSource.FILE:
                    self.builder.set_from_file(new_key, value)
                elif source == ConfigSource.CLI:
                    self.builder.set_from_cli(new_key, value)
                    
                # Log mapping for debugging
                if old_key != new_key:
                    logger.debug(f"Mapped {old_key} -> {new_key}")
    
    def set_from_cli_args(self, **kwargs: object) -> None:
        """Set configuration from CLI arguments."""
        # Map CLI argument names to configuration keys
        cli_mappings = {
            "use_reranker": "search.use_reranker",
            "rerank": "search.use_reranker",  # Support both names
            "reranker_model": "search.reranker_model",
            "top_k": "search.top_k",
            "reranker_top_k": "search.reranker_top_k",
        }
        
        for arg_name, config_key in cli_mappings.items():
            if arg_name in kwargs and kwargs[arg_name] is not None:
                self.builder.set_from_cli(config_key, kwargs[arg_name])
    
    def resolve_search_config(self, query: str, mode: SearchMode) -> ResolvedSearchConfig:
        """Resolve search configuration with all values."""
        logger.info("🔧 Resolving search configuration...")
        
        # Check for configuration conflicts
        self.builder.check_configuration_conflicts()
        
        # Collect source information for debugging
        sources = {}
        
        def get_value_and_source(key: str) -> tuple[Any, Optional[ConfigSource]]:
            config_value = self.builder.get_with_source(key)
            if config_value:
                sources[key] = config_value.source
                return config_value.value, config_value.source
            return None, None
        
        use_reranker, use_reranker_source = get_value_and_source("search.use_reranker")
        reranker_model, reranker_model_source = get_value_and_source("search.reranker_model")
        top_k, top_k_source = get_value_and_source("search.top_k")
        reranker_top_k, reranker_top_k_source = get_value_and_source("search.reranker_top_k")
        
        config = ResolvedSearchConfig(
            query=query,
            mode=mode,
            top_k=top_k,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            reranker_top_k=reranker_top_k,
            sources=sources
        )
        
        # Log resolved configuration with detailed sources
        logger.info("✅ Search configuration resolved:")
        logger.info(f"  📝 Query: {query}")
        logger.info(f"  🔍 Mode: {mode}")
        logger.info(f"  🔢 Top-k: {config.top_k} (from {top_k_source.name if top_k_source else 'DEFAULT'})")
        
        if use_reranker_source and use_reranker_source != ConfigSource.DEFAULT:
            logger.info(f"  🎯 Reranker: {config.use_reranker} (EXPLICITLY set from {use_reranker_source.name})")
        else:
            logger.info(f"  🎯 Reranker: {config.use_reranker} (from DEFAULT)")
            
        logger.info(f"  🤖 Reranker model: {config.reranker_model} (from {reranker_model_source.name if reranker_model_source else 'DEFAULT'})")
        logger.info(f"  🔟 Reranker top-k: {config.reranker_top_k} (from {reranker_top_k_source.name if reranker_top_k_source else 'DEFAULT'})")
        
        # Highlight if reranker was explicitly disabled/enabled
        if use_reranker_source and use_reranker_source == ConfigSource.CLI:
            if config.use_reranker:
                logger.info("🔥 Reranker EXPLICITLY ENABLED via CLI - will be used regardless of config defaults")
            else:
                logger.info("❄️ Reranker EXPLICITLY DISABLED via CLI - will NOT be used regardless of config defaults")
        
        return config
    
    def resolve_indexer_config(self) -> ResolvedIndexerConfig:
        """Resolve indexer configuration with all values."""
        sources = {}
        
        def get_value_and_source(key: str) -> tuple[Any, Optional[ConfigSource]]:
            config_value = self.builder.get_with_source(key)
            if config_value:
                sources[key] = config_value.source
                return config_value.value, config_value.source
            return None, None
        
        use_reranker, _ = get_value_and_source("indexer.use_reranker")
        reranker_model, _ = get_value_and_source("indexer.reranker_model")
        reranker_device, _ = get_value_and_source("indexer.reranker_device")
        reranker_use_onnx, _ = get_value_and_source("indexer.reranker_use_onnx")
        chunk_size, _ = get_value_and_source("indexer.chunk_size")
        chunk_overlap, _ = get_value_and_source("indexer.chunk_overlap")
        
        config = ResolvedIndexerConfig(
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            reranker_device=reranker_device,
            reranker_use_onnx=reranker_use_onnx,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sources=sources
        )
        
        logger.info("Resolved indexer configuration:")
        logger.info(f"  use_reranker: {config.use_reranker} (from {sources.get('indexer.use_reranker', 'unknown')})")
        
        return config

