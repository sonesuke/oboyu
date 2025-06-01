"""Refactored indexer facade using orchestrator pattern."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from oboyu.crawler.crawler import CrawlerResult
from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.core.search_mode import SearchMode
from oboyu.indexer.orchestrators.indexing_pipeline import IndexingPipeline
from oboyu.indexer.orchestrators.search_orchestrator import SearchOrchestrator
from oboyu.indexer.orchestrators.service_registry import ServiceRegistry
from oboyu.indexer.search.search_filters import SearchFilters
from oboyu.indexer.search.search_result import SearchResult

logger = logging.getLogger(__name__)


class Indexer:
    """Main facade that coordinates indexing and search operations through orchestrators."""

    def __init__(self, config: Optional[IndexerConfig] = None) -> None:
        """Initialize the indexer orchestrator with configuration.

        Args:
            config: Indexer configuration

        """
        # Initialize configuration
        self.config = config or IndexerConfig()
        
        # Initialize service registry with all dependencies
        self.services = ServiceRegistry(self.config)
        
        # Initialize orchestrators
        self.indexing_pipeline = IndexingPipeline(self.services)
        self.search_orchestrator = SearchOrchestrator(self.services)
        
        # Backward compatibility: expose services directly
        self.database_service = self.services.get_database_service()
        self.embedding_service = self.services.get_embedding_service()
        self.document_processor = self.services.get_document_processor()
        self.bm25_indexer = self.services.get_bm25_indexer()
        self.tokenizer_service = self.services.get_tokenizer_service()
        self.reranker_service = self.services.get_reranker_service()
        self.search_engine = self.services.get_search_engine()
        self.change_detector = self.services.get_change_detector()

    def index_documents(self, crawler_results: List[CrawlerResult], progress_callback: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
        """Index documents using the indexing pipeline.

        Args:
            crawler_results: Results from document crawling
            progress_callback: Optional callback for progress updates

        Returns:
            Indexing result summary

        """
        return self.indexing_pipeline.index_documents(crawler_results, progress_callback)

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
        language_filter: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Search using the search orchestrator.

        Args:
            query: Search query
            limit: Maximum number of results
            mode: Search mode ("vector", "bm25", "hybrid")
            language_filter: Optional language filter
            filters: Optional search filters for date range and path filtering

        Returns:
            List of search results

        """
        # Convert mode string to enum
        if mode == "vector":
            search_mode = SearchMode.VECTOR
        elif mode == "bm25":
            search_mode = SearchMode.BM25
        elif mode == "hybrid":
            search_mode = SearchMode.HYBRID
        else:
            logger.warning(f"Unknown search mode: {mode}, using hybrid")
            search_mode = SearchMode.HYBRID

        return self.search_orchestrator.search(
            query=query,
            mode=search_mode,
            limit=limit,
            language_filter=language_filter,
            filters=filters,
        )

    def vector_search(
        self,
        query: Union[str, NDArray[np.float32]],
        top_k: int = 10,
        limit: Optional[int] = None,
        language_filter: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Vector search using query string or embedding.

        Args:
            query: Search query string or pre-computed embedding
            top_k: Maximum number of results (new parameter name)
            limit: Deprecated parameter name for compatibility
            language_filter: Optional language filter
            filters: Optional search filters for date range and path filtering

        Returns:
            List of search results

        """
        # Handle legacy parameter name
        result_limit = top_k if limit is None else limit
            
        return self.search_orchestrator.vector_search(
            query=query,
            limit=result_limit,
            language_filter=language_filter,
            filters=filters,
        )

    def delete_document(self, path: Path) -> int:
        """Delete a document and all its chunks.

        Args:
            path: Path to the document to delete

        Returns:
            Number of chunks deleted

        """
        return self.database_service.delete_chunks_by_path(path)

    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics.

        Returns:
            Dictionary with indexer statistics

        """
        # Ensure config is properly initialized
        assert self.config.model is not None, "ModelConfig should be initialized"
        
        return {
            "total_chunks": self.database_service.get_chunk_count(),
            "indexed_paths": len(self.database_service.get_paths_with_chunks()),
            "embedding_model": self.config.model.embedding_model,
            "reranker_enabled": self.config.use_reranker,
        }

    def clear_index(self) -> None:
        """Clear all indexed data."""
        self.database_service.clear_database()
        if hasattr(self.bm25_indexer, "clear"):
            self.bm25_indexer.clear()

    def close(self) -> None:
        """Close all services and connections."""
        self.services.close()

    @classmethod
    def from_path(cls, db_path: Union[str, Path]) -> "Indexer":
        """Create an indexer from a database path.

        Args:
            db_path: Path to the database file

        Returns:
            Initialized Indexer instance

        """
        from oboyu.indexer.config.indexer_config import IndexerConfig
        
        config = IndexerConfig()
        config.db_path = db_path  # Use the property which handles the assertion
        return cls(config)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics

        """
        return self.database_service.get_database_stats()

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        language_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """BM25 search using query string.

        Args:
            query: Search query string
            top_k: Maximum number of results
            language_filter: Optional language filter

        Returns:
            List of search results

        """
        return self.search_orchestrator.bm25_search(
            query=query,
            limit=top_k,
            language_filter=language_filter,
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        language_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Hybrid search using query string.

        Args:
            query: Search query string
            top_k: Maximum number of results
            vector_weight: Weight for vector search component
            bm25_weight: Weight for BM25 search component
            language_filter: Optional language filter

        Returns:
            List of search results

        """
        return self.search_orchestrator.hybrid_search(
            query=query,
            limit=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            language_filter=language_filter,
        )

    def rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results using the reranker service.

        Args:
            query: Original search query
            results: Search results to rerank

        Returns:
            Reranked search results

        """
        return self.search_orchestrator.rerank_results(query, results)
