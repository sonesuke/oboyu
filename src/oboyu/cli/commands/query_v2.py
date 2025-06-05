"""Updated query command using immutable configuration pattern."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from oboyu.common.config import ConfigManager
from oboyu.common.paths import DEFAULT_DB_PATH
from oboyu.common.types import SearchMode, SearchResult
from oboyu.config import (
    ConfigSource,
    ConfigurationResolver,
)
from oboyu.retriever.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a query operation."""
    
    results: List[SearchResult]
    elapsed_time: float
    mode: str
    total_results: int
    reranker_used: bool = False
    configuration_sources: Optional[Dict[str, str]] = None


class QueryCommandV2:
    """Query command service using immutable configuration pattern."""
    
    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize the query command service.
        
        Args:
            config_manager: Configuration manager instance

        """
        self.config_manager = config_manager
    
    def execute_query(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        db_path: Optional[Path] = None,
        rerank: Optional[bool] = None,
    ) -> QueryResult:
        """Execute a search query using immutable configuration."""
        # Create configuration resolver
        resolver = ConfigurationResolver()
        
        # Load configuration from file
        config_dict = self.config_manager.load_config()
        if config_dict:
            resolver.load_from_dict(config_dict, ConfigSource.FILE)
        
        # Apply CLI arguments - these take highest precedence
        cli_args = {}
        if top_k is not None:
            cli_args['top_k'] = top_k
        if rerank is not None:
            cli_args['use_reranker'] = rerank
        # Note: rrf_k is not yet supported in the new configuration system
        
        resolver.set_from_cli_args(**cli_args)
        
        # Log configuration for debugging
        resolver.builder.log_configuration()
        
        # Convert mode string to SearchMode enum
        search_mode = SearchMode.HYBRID
        if mode == "vector":
            search_mode = SearchMode.VECTOR
        elif mode == "bm25":
            search_mode = SearchMode.BM25
        
        # Resolve configuration
        search_config = resolver.resolve_search_config(query=query, mode=search_mode)
        
        # Determine database path
        query_config = self.config_manager.get_section("query")
        database_path = Path(db_path or query_config.get("database_path") or DEFAULT_DB_PATH)
        
        # Initialize retriever with proper configuration
        from oboyu.indexer.config.indexer_config import IndexerConfig
        
        config = IndexerConfig()
        config.db_path = database_path
        
        # Apply reranker configuration if enabled
        if search_config.use_reranker:
            assert config.search is not None, "SearchConfig should be initialized"
            assert config.model is not None, "ModelConfig should be initialized"
            config.search.use_reranker = True
            config.model.use_reranker = True
            config.model.reranker_model = search_config.reranker_model
        
        retriever = Retriever(config)
        
        try:
            start_time = time.time()
            
            try:
                # Execute search based on mode
                if search_mode == SearchMode.VECTOR:
                    results = retriever.vector_search(query, top_k=search_config.top_k)
                elif search_mode == SearchMode.BM25:
                    results = retriever.bm25_search(query, top_k=search_config.top_k)
                else:  # hybrid
                    results = retriever.hybrid_search(
                        query,
                        top_k=search_config.top_k,
                    )
                
                # Apply reranking if enabled
                if search_config.use_reranker and results:
                    try:
                        results = retriever.rerank_results(query, results)
                        # Limit results to reranker_top_k if specified
                        if search_config.reranker_top_k and len(results) > search_config.reranker_top_k:
                            results = results[:search_config.reranker_top_k]
                    except Exception as e:
                        # Check if this is a model loading error
                        if isinstance(e, RuntimeError) and "Failed to load" in str(e) and "model" in str(e):
                            logger.error(f"❌ Reranking failed due to model loading error: {e}")
                            logger.warning("Continuing with search results without reranking.")
                        else:
                            logger.warning(f"Reranking failed: {e}")
            
            except RuntimeError as e:
                # Check if this is a model loading error from our services
                if "Failed to load" in str(e) and "model" in str(e):
                    raise RuntimeError(f"❌ Search failed due to model loading error:\n{str(e)}") from e
                else:
                    raise
            
            elapsed_time = time.time() - start_time
            
            # Prepare configuration sources for debugging
            config_sources = {}
            if search_config.sources:
                for key, source in search_config.sources.items():
                    config_sources[key] = source.name
            
            return QueryResult(
                results=results,
                elapsed_time=elapsed_time,
                mode=mode,
                total_results=len(results),
                reranker_used=search_config.use_reranker and len(results) > 0,
                configuration_sources=config_sources,
            )
        except Exception:
            # Ensure clean shutdown
            try:
                retriever.close()
            except Exception as cleanup_error:
                # Ignore errors during cleanup but log for debugging
                logger.debug(f"Error during retriever cleanup: {cleanup_error}")
            raise
        finally:
            retriever.close()
    
    def get_database_path(self, db_path: Optional[Path] = None) -> str:
        """Get the resolved database path."""
        query_config = self.config_manager.get_section("query")
        database_path = Path(db_path or query_config.get("database_path") or DEFAULT_DB_PATH)
        return str(database_path)
    
    def get_query_config(
        self,
        top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        rerank: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Get query configuration with overrides."""
        # Create configuration resolver
        resolver = ConfigurationResolver()
        
        # Load configuration from file
        config_dict = self.config_manager.load_config()
        if config_dict:
            resolver.load_from_dict(config_dict, ConfigSource.FILE)
        
        # Apply CLI arguments
        cli_args = {}
        if top_k is not None:
            cli_args['top_k'] = top_k
        if rerank is not None:
            cli_args['use_reranker'] = rerank
        
        resolver.set_from_cli_args(**cli_args)
        
        # Return as dictionary
        return {
            "top_k": resolver.builder.get("search.top_k"),
            "use_reranker": resolver.builder.get("search.use_reranker"),
            "reranker_model": resolver.builder.get("search.reranker_model"),
            "rrf_k": rrf_k if rrf_k is not None else 60,  # Default RRF value
        }

