"""Query service for handling all search-related business logic."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from oboyu.common.config import ConfigManager
from oboyu.common.paths import DEFAULT_DB_PATH
from oboyu.indexer import Indexer
from oboyu.indexer.search.search_result import SearchResult


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
        
        # Initialize indexer
        indexer = Indexer.from_path(database_path)
        
        try:
            start_time = time.time()
            
            # Execute search based on mode
            if mode == "vector":
                results = indexer.vector_search(query, top_k=query_config.get("top_k", 10))
            elif mode == "bm25":
                results = indexer.bm25_search(query, top_k=query_config.get("top_k", 10))
            else:  # hybrid
                results = indexer.hybrid_search(
                    query,
                    top_k=query_config.get("top_k", 10),
                    vector_weight=query_config.get("vector_weight", 0.7),
                    bm25_weight=query_config.get("bm25_weight", 0.3),
                )
            
            # Apply reranking if enabled
            if query_config.get("use_reranker", False) and results:
                try:
                    results = indexer.rerank_results(query, results)
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
            indexer.close()
    
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
