"""Consolidated search engine combining SearchEngine and SearchOrchestrator functionality."""

import logging
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from oboyu.retriever.search.bm25_search import BM25Search
from oboyu.retriever.search.hybrid_search import HybridSearch
from oboyu.retriever.search.hybrid_search_combiner import HybridSearchCombiner
from oboyu.retriever.search.mode_router import SearchModeRouter
from oboyu.retriever.search.result_merger import ResultMerger
from oboyu.retriever.search.score_normalizer import ScoreNormalizer
from oboyu.retriever.search.search_context import SearchContext, SystemDefaults
from oboyu.retriever.search.search_filters import SearchFilters
from oboyu.retriever.search.search_mode import SearchMode
from oboyu.retriever.search.search_result import SearchResult
from oboyu.retriever.search.vector_search import VectorSearch

logger = logging.getLogger(__name__)


class SearchEngine:
    """Consolidated search engine that orchestrates all search operations.
    
    This class combines the functionality from the original SearchEngine
    and SearchOrchestrator classes to provide a unified interface for
    all search operations including vector search, BM25 search, and
    hybrid search with optional reranking.
    """

    def __init__(
        self,
        vector_search: VectorSearch,
        bm25_search: BM25Search,
        hybrid_search: HybridSearch,
        embedding_service: Optional[object] = None,
        tokenizer_service: Optional[object] = None,
        reranker_service: Optional[object] = None,
        config: Optional[object] = None,
    ) -> None:
        """Initialize search engine with required and optional services.

        Args:
            vector_search: Vector similarity search service
            bm25_search: BM25 keyword search service
            hybrid_search: Hybrid search combination service
            embedding_service: Optional embedding service for query processing
            tokenizer_service: Optional tokenizer service for query processing
            reranker_service: Optional reranker service for result refinement
            config: Optional configuration object

        """
        # Core search services
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.hybrid_search = hybrid_search
        
        # Optional high-level services
        self.embedding_service = embedding_service
        self.tokenizer_service = tokenizer_service
        self.reranker_service = reranker_service
        self.config = config
        
        # Initialize composed components
        self.router = SearchModeRouter(vector_search, bm25_search)
        self.merger = ResultMerger()
        self.normalizer = ScoreNormalizer()
        self.combiner = HybridSearchCombiner(
            vector_weight=hybrid_search.vector_weight,
            bm25_weight=hybrid_search.bm25_weight,
            score_normalizer=self.normalizer,
        )

    def search(
        self,
        query_vector: Optional[NDArray[np.float32]] = None,
        query_terms: Optional[List[str]] = None,
        mode: SearchMode = SearchMode.HYBRID,
        limit: int = 10,
        language_filter: Optional[str] = None,
        top_k_multiplier: int = 2,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Execute search using appropriate search mode with pre-processed inputs.

        Args:
            query_vector: Query embedding vector (required for vector and hybrid search)
            query_terms: Query terms (required for BM25 and hybrid search)
            mode: Search mode to use
            limit: Maximum number of results to return
            language_filter: Optional language filter
            top_k_multiplier: Multiplier for initial retrieval in hybrid search
            filters: Optional search filters for date range and path filtering

        Returns:
            List of search results

        """
        try:
            # Route non-hybrid searches to appropriate implementation
            if mode in (SearchMode.VECTOR, SearchMode.BM25):
                return self.router.route(
                    mode=mode,
                    query_vector=query_vector,
                    query_terms=query_terms,
                    limit=limit,
                    language_filter=language_filter,
                    filters=filters,
                )

            elif mode == SearchMode.HYBRID:
                if query_vector is None or query_terms is None:
                    raise ValueError("Both query vector and terms are required for hybrid search")

                # Get more results for hybrid combination
                initial_limit = limit * top_k_multiplier

                # Execute both searches through router
                vector_results = self.router.route(
                    mode=SearchMode.VECTOR,
                    query_vector=query_vector,
                    limit=initial_limit,
                    language_filter=language_filter,
                    filters=filters,
                )

                bm25_results = self.router.route(
                    mode=SearchMode.BM25,
                    query_terms=query_terms,
                    limit=initial_limit,
                    language_filter=language_filter,
                    filters=filters,
                )

                # Combine results using the new combiner
                combined_results = self.combiner.combine(
                    vector_results=vector_results,
                    bm25_results=bm25_results,
                    limit=limit,
                )
                
                # Also use the original hybrid search for backward compatibility
                legacy_results = self.hybrid_search.search(
                    vector_results=vector_results,
                    bm25_results=bm25_results,
                    limit=limit,
                )
                
                # Merge both results to ensure backward compatibility
                return self.merger.merge(combined_results, legacy_results, limit=limit)

            else:
                raise ValueError(f"Unknown search mode: {mode}")

        except Exception as e:
            logger.error(f"Search failed with mode {mode}: {e}")
            return []

    def search_with_query(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        limit: int = 10,
        use_reranker: Optional[bool] = None,
        language_filter: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[SearchResult]:
        """Execute search with query string processing and optional reranking.
        
        This method provides the high-level orchestration functionality
        from the original SearchOrchestrator class.
        
        Args:
            query: Search query string
            mode: Search mode (VECTOR, BM25, or HYBRID)
            limit: Maximum number of results
            use_reranker: Whether to use reranker (None uses config default)
            language_filter: Optional language filter
            filters: Optional search filters for date range and path filtering
            vector_weight: Optional weight for vector search in hybrid mode
            bm25_weight: Optional weight for BM25 search in hybrid mode
            
        Returns:
            List of search results
            
        """
        if not self.embedding_service or not self.tokenizer_service:
            raise RuntimeError("Embedding and tokenizer services required for query processing")
            
        try:
            # Prepare search inputs based on mode
            query_vector = None
            query_terms = None
            
            if mode in [SearchMode.VECTOR, SearchMode.HYBRID]:
                query_vector = self.embedding_service.generate_query_embedding(query)
                
            if mode in [SearchMode.BM25, SearchMode.HYBRID]:
                query_terms = self.tokenizer_service.tokenize_query(query)
                
            # Determine result limit considering reranking
            search_limit = limit
            if use_reranker is None and self.config:
                use_reranker = self.config.use_reranker
                
            if use_reranker and self.reranker_service and self.config and self.config.search:
                search_limit = limit * self.config.search.top_k_multiplier
                
            # Execute search
            results = self.search(
                query_vector=query_vector,
                query_terms=query_terms,
                mode=mode,
                limit=search_limit,
                language_filter=language_filter,
                filters=filters,
            )
            
            # Apply reranking if enabled
            if use_reranker and self.reranker_service and self.reranker_service.is_available() and results:
                results = self.reranker_service.rerank(query, results)
                
            # Return final results with limit
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Search with query failed: {e}")
            return []

    def search_with_context(
        self,
        query: str,
        context: SearchContext,
        mode: SearchMode = SearchMode.HYBRID,
        language_filter: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Execute search using SearchContext pattern that preserves explicit settings.
        
        This method implements the Context Pattern to ensure user-specified
        settings are never overridden by default values.
        
        Args:
            query: Search query
            context: SearchContext with explicit settings
            mode: Search mode (VECTOR, BM25, or HYBRID)
            language_filter: Optional language filter
            filters: Optional search filters for date range and path filtering
            
        Returns:
            List of search results
            
        """
        if not self.embedding_service or not self.tokenizer_service:
            raise RuntimeError("Embedding and tokenizer services required for query processing")
            
        try:
            # Prepare search inputs based on mode
            query_vector = None
            query_terms = None
            
            if mode in [SearchMode.VECTOR, SearchMode.HYBRID]:
                query_vector = self.embedding_service.generate_query_embedding(query)
                
            if mode in [SearchMode.BM25, SearchMode.HYBRID]:
                query_terms = self.tokenizer_service.tokenize_query(query)
            
            # Determine settings using context pattern - explicit settings are NEVER overridden
            if context.is_explicitly_set('reranker_enabled'):
                use_reranker = context.get_reranker_setting()
                logger.info(f"🔧 Using EXPLICIT reranker: {use_reranker}")
            else:
                use_reranker = SystemDefaults.get_reranker_default()
                logger.info(f"📝 Using DEFAULT reranker: {use_reranker}")
            
            if context.is_explicitly_set('top_k'):
                explicit_limit = context.get_top_k_setting()
                assert explicit_limit is not None, "Explicit top_k should not be None"
                limit = explicit_limit
                logger.debug(f"🔧 Using EXPLICIT top_k: {limit}")
            else:
                limit = SystemDefaults.get_top_k_default()
                logger.debug(f"📝 Using DEFAULT top_k: {limit}")
            
            # Log final settings for debugging
            final_settings = {
                'reranker_enabled': use_reranker,
                'top_k': limit,
            }
            context.log_final_settings(final_settings)
            
            # Determine result limit considering reranking
            search_limit = limit
            if use_reranker and self.reranker_service and self.config and self.config.search:
                search_limit = limit * self.config.search.top_k_multiplier
                logger.debug(f"🔧 Reranker enabled, expanding search limit to {search_limit}")
            
            # Execute search
            results = self.search(
                query_vector=query_vector,
                query_terms=query_terms,
                mode=mode,
                limit=search_limit,
                language_filter=language_filter,
                filters=filters,
            )
            
            # Apply reranking if enabled
            if use_reranker and self.reranker_service and self.reranker_service.is_available() and results:
                logger.info(f"🔧 Applying reranking to {len(results)} results")
                results = self.reranker_service.rerank(query, results)
                logger.info(f"🔧 Reranking completed, {len(results)} results returned")
            elif use_reranker and not self.reranker_service:
                logger.warning("🔧 Reranker explicitly requested but reranker service not available")
            elif use_reranker and self.reranker_service and not self.reranker_service.is_available():
                logger.warning("🔧 Reranker explicitly requested but reranker service not initialized")
                
            # Return final results with limit
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Search with context failed: {e}")
            return []
            
    def vector_search(
        self,
        query: Union[str, NDArray[np.float32]],
        limit: int = 10,
        language_filter: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Execute vector search using query string or embedding.
        
        Args:
            query: Search query string or pre-computed embedding
            limit: Maximum number of results
            language_filter: Optional language filter
            filters: Optional search filters
            
        Returns:
            List of search results
            
        """
        # Handle both string queries and embedding vectors
        if isinstance(query, str):
            if not self.embedding_service:
                raise RuntimeError("Embedding service required for string query processing")
            query_embedding = self.embedding_service.generate_query_embedding(query)
        else:
            query_embedding = query
            
        return self.search(
            query_vector=query_embedding,
            mode=SearchMode.VECTOR,
            limit=limit,
            language_filter=language_filter,
            filters=filters,
        )
        
    def bm25_search(
        self,
        query: str,
        limit: int = 10,
        language_filter: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Execute BM25 search using query string.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            language_filter: Optional language filter
            filters: Optional search filters
            
        Returns:
            List of search results
            
        """
        if not self.tokenizer_service:
            raise RuntimeError("Tokenizer service required for BM25 search")
            
        # Tokenize the query
        query_terms = self.tokenizer_service.tokenize_query(query)
        
        # Execute BM25 search
        return self.search(
            query_terms=query_terms,
            mode=SearchMode.BM25,
            limit=limit,
            language_filter=language_filter,
            filters=filters,
        )
        
    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        language_filter: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Execute hybrid search combining vector and BM25 search.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            vector_weight: Weight for vector search component
            bm25_weight: Weight for BM25 search component
            language_filter: Optional language filter
            filters: Optional search filters
            
        Returns:
            List of search results
            
        """
        if not self.embedding_service or not self.tokenizer_service:
            raise RuntimeError("Embedding and tokenizer services required for hybrid search")
            
        # Generate query embedding and tokenize query
        query_vector = self.embedding_service.generate_query_embedding(query)
        query_terms = self.tokenizer_service.tokenize_query(query)
        
        # Execute hybrid search
        return self.search(
            query_vector=query_vector,
            query_terms=query_terms,
            mode=SearchMode.HYBRID,
            limit=limit,
            language_filter=language_filter,
            filters=filters,
        )
        
    def rerank_results(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Rerank search results using the reranker service.
        
        Args:
            query: Original search query
            results: Search results to rerank
            
        Returns:
            Reranked search results
            
        """
        if not self.reranker_service or not self.reranker_service.is_available():
            logger.warning("Reranker service not available, returning original results")
            return results
            
        return self.reranker_service.rerank(query, results)


# Legacy aliases for backward compatibility
SearchOrchestrator = SearchEngine  # Allow code to import SearchOrchestrator as alias
