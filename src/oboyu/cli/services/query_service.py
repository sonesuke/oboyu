"""Query service for handling all search-related business logic."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from oboyu.common.config import ConfigManager
from oboyu.common.paths import DEFAULT_DB_PATH
from oboyu.common.types import SearchResult
from oboyu.retriever.retriever import Retriever
from oboyu.retriever.search.search_context import ContextBuilder, SettingSource


@dataclass
class QueryResult:
    """Result of a query operation."""
    
    results: List[SearchResult]
    elapsed_time: float
    mode: str
    total_results: int


class QueryService:
    """Service for handling query operations."""
    
    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize the query service.
        
        Args:
            config_manager: Configuration manager instance

        """
        self.config_manager = config_manager
    
    def execute_query(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: Optional[int] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        db_path: Optional[Path] = None,
        rerank: Optional[bool] = None,
    ) -> QueryResult:
        """Execute a search query.
        
        Args:
            query: Search query text
            mode: Search mode (vector, bm25, hybrid)
            top_k: Number of results to return
            vector_weight: Weight for vector search in hybrid mode
            bm25_weight: Weight for BM25 search in hybrid mode
            db_path: Optional database path override
            rerank: Whether to use reranking
            
        Returns:
            QueryResult with search results and metadata

        """
        # Get query engine configuration
        query_config = self.config_manager.get_section("query")
        
        # Override with provided options
        cli_overrides: Dict[str, Any] = {}
        if top_k is not None:
            cli_overrides["top_k"] = top_k
        if vector_weight is not None:
            cli_overrides["vector_weight"] = vector_weight
        if bm25_weight is not None:
            cli_overrides["bm25_weight"] = bm25_weight
        if rerank is not None:
            cli_overrides["use_reranker"] = rerank
        
        query_config = self.config_manager.merge_cli_overrides("query", cli_overrides)
        
        # Determine database path
        database_path = Path(db_path or query_config.get("database_path") or DEFAULT_DB_PATH)
        
        # Initialize retriever with proper configuration including reranker settings
        from oboyu.indexer.config.indexer_config import IndexerConfig
        
        config = IndexerConfig()
        config.db_path = database_path
        
        # Apply reranker configuration if enabled
        if query_config.get("use_reranker", False):
            assert config.search is not None, "SearchConfig should be initialized"
            assert config.model is not None, "ModelConfig should be initialized"
            config.search.use_reranker = True
            config.model.use_reranker = True
        
        retriever = Retriever(config)
        
        try:
            start_time = time.time()
            
            # Execute search based on mode
            if mode == "vector":
                results = retriever.vector_search(query, top_k=query_config.get("top_k", 10))
            elif mode == "bm25":
                results = retriever.bm25_search(query, top_k=query_config.get("top_k", 10))
            else:  # hybrid
                results = retriever.hybrid_search(
                    query,
                    top_k=query_config.get("top_k", 10),
                )
            
            # Apply reranking if enabled
            if query_config.get("use_reranker", False) and results:
                try:
                    results = retriever.rerank_results(query, results)
                except Exception as e:
                    # Log reranking failure but continue with original results
                    import logging
                    logging.warning(f"Reranking failed: {e}")
            
            elapsed_time = time.time() - start_time
            
            return QueryResult(
                results=results,
                elapsed_time=elapsed_time,
                mode=mode,
                total_results=len(results),
            )
        finally:
            retriever.close()
    
    def execute_query_with_context(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: Optional[int] = None,
        db_path: Optional[Path] = None,
        rerank: Optional[bool] = None,
    ) -> QueryResult:
        """Execute a search query using SearchContext pattern for explicit setting tracking.
        
        This method implements the Context Pattern to ensure user-specified
        settings are never overridden by default values.
        
        Args:
            query: Search query text
            mode: Search mode (vector, bm25, hybrid)
            top_k: Number of results to return
            db_path: Optional database path override
            rerank: Whether to use reranking
            
        Returns:
            QueryResult with search results and metadata

        """
        # Build search context from CLI arguments - only explicit values are set
        context_builder = ContextBuilder()
        
        if top_k is not None:
            context_builder.with_top_k(top_k, SettingSource.CLI_ARGUMENT)
        if rerank is not None:
            context_builder.with_reranker(rerank, SettingSource.CLI_ARGUMENT)
            
        context = context_builder.build()
        
        # Get query engine configuration for database path
        query_config = self.config_manager.get_section("query")
        
        # Determine database path
        database_path = Path(db_path or query_config.get("database_path") or DEFAULT_DB_PATH)
        
        # Initialize retriever with proper configuration
        from oboyu.common.types import SearchMode
        from oboyu.indexer.config.indexer_config import IndexerConfig
        
        config = IndexerConfig()
        config.db_path = database_path
        
        # Important: Enable reranker service if ANY explicit reranker setting exists
        # This ensures the reranker service is available when user explicitly wants it
        if context.is_explicitly_set('reranker_enabled'):
            assert config.search is not None, "SearchConfig should be initialized"
            assert config.model is not None, "ModelConfig should be initialized"
            config.search.use_reranker = True
            config.model.use_reranker = True
        
        retriever = Retriever(config)
        
        try:
            start_time = time.time()
            
            # Convert mode string to SearchMode enum
            search_mode = SearchMode.HYBRID
            if mode == "vector":
                search_mode = SearchMode.VECTOR
            elif mode == "bm25":
                search_mode = SearchMode.BM25
            
            # Execute search using context pattern
            results = retriever.search_orchestrator.search_with_context(
                query=query,
                context=context,
                mode=search_mode,
            )
            
            elapsed_time = time.time() - start_time
            
            return QueryResult(
                results=results,
                elapsed_time=elapsed_time,
                mode=mode,
                total_results=len(results),
            )
        finally:
            retriever.close()
    
    def get_database_path(self, db_path: Optional[Path] = None) -> str:
        """Get the resolved database path.
        
        Args:
            db_path: Optional database path override
            
        Returns:
            Resolved database path as string

        """
        query_config = self.config_manager.get_section("query")
        database_path = Path(db_path or query_config.get("database_path") or DEFAULT_DB_PATH)
        return str(database_path)
    
    def get_query_config(
        self,
        top_k: Optional[int] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        rerank: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Get query configuration with overrides.
        
        Args:
            top_k: Optional top_k override
            vector_weight: Optional vector_weight override
            bm25_weight: Optional bm25_weight override
            rerank: Optional rerank override
            
        Returns:
            Query configuration dictionary

        """
        # Override with provided options
        cli_overrides: Dict[str, Any] = {}
        if top_k is not None:
            cli_overrides["top_k"] = top_k
        if vector_weight is not None:
            cli_overrides["vector_weight"] = vector_weight
        if bm25_weight is not None:
            cli_overrides["bm25_weight"] = bm25_weight
        if rerank is not None:
            cli_overrides["use_reranker"] = rerank
        
        return self.config_manager.merge_cli_overrides("query", cli_overrides)
