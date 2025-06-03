"""Search orchestrator for coordinating search operations across different modes."""

import logging
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from oboyu.common.types import SearchFilters, SearchMode, SearchResult
from oboyu.retriever.orchestrators.service_registry import ServiceRegistry
from oboyu.retriever.search.search_context import SearchContext, SystemDefaults

logger = logging.getLogger(__name__)


class SearchOrchestrator:
    """Coordinates search operations across different modes."""
    
    def __init__(self, services: ServiceRegistry) -> None:
        """Initialize the search orchestrator with services.
        
        Args:
            services: Service registry providing dependencies
            
        """
        self.services = services
        self.search_engine = services.get_search_engine()
        self.embedding_service = services.get_embedding_service()
        self.tokenizer_service = services.get_tokenizer_service()
        self.reranker_service = services.get_reranker_service()
        self.config = services.config
        
    def search(
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
        """Execute search with specified mode and parameters.
        
        Args:
            query: Search query
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
            if use_reranker is None:
                use_reranker = self.config.use_reranker
                
            if use_reranker and self.reranker_service:
                assert self.config.search is not None
                search_limit = limit * self.config.search.top_k_multiplier
                
            # Execute search
            results = self.search_engine.search(
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
            logger.error(f"Search failed: {e}")
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
            if use_reranker and self.reranker_service:
                assert self.config.search is not None
                search_limit = limit * self.config.search.top_k_multiplier
                logger.debug(f"🔧 Reranker enabled, expanding search limit to {search_limit}")
            
            # Execute search
            results = self.search_engine.search(
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
            query_embedding = self.embedding_service.generate_query_embedding(query)
        else:
            query_embedding = query
            
        return self.search_engine.search(
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
        # Tokenize the query
        query_terms = self.tokenizer_service.tokenize_query(query)
        
        # Execute BM25 search
        return self.search_engine.search(
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
        rrf_k: Optional[int] = None,
        language_filter: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Execute hybrid search combining vector and BM25 search with RRF.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            rrf_k: RRF parameter for rank fusion (default: from config)
            language_filter: Optional language filter
            filters: Optional search filters
            
        Returns:
            List of search results
            
        """
        # Generate query embedding and tokenize query
        query_vector = self.embedding_service.generate_query_embedding(query)
        query_terms = self.tokenizer_service.tokenize_query(query)
        
        # Execute hybrid search
        return self.search_engine.search(
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
