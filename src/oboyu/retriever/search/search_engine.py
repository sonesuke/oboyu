"""Search engine coordinator for different search modes."""

import logging
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from oboyu.common.types import SearchFilters, SearchMode, SearchResult
from oboyu.retriever.search.bm25_search import BM25Search
from oboyu.retriever.search.hybrid_search_combiner import HybridSearchCombiner
from oboyu.retriever.search.mode_router import SearchModeRouter
from oboyu.retriever.search.result_merger import ResultMerger
from oboyu.retriever.search.score_normalizer import ScoreNormalizer
from oboyu.retriever.search.vector_search import VectorSearch

logger = logging.getLogger(__name__)


class SearchEngine:
    """Simplified search engine that orchestrates focused components."""

    def __init__(
        self,
        vector_search: VectorSearch,
        bm25_search: BM25Search,
        rrf_k: int = 60,
    ) -> None:
        """Initialize search engine.

        Args:
            vector_search: Vector similarity search service
            bm25_search: BM25 keyword search service
            rrf_k: RRF parameter for hybrid search combination

        """
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.rrf_k = rrf_k
        
        # Initialize composed components
        self.router = SearchModeRouter(vector_search, bm25_search)
        self.merger = ResultMerger()
        self.normalizer = ScoreNormalizer()
        self.combiner = HybridSearchCombiner(
            rrf_k=rrf_k,
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
        """Execute search based on mode.

        Args:
            query_vector: Query embedding vector
            query_terms: Query terms for keyword search
            mode: Search mode (vector, bm25, or hybrid)
            limit: Maximum number of results
            language_filter: Optional language filter
            top_k_multiplier: Multiplier for initial retrieval before reranking
            filters: Optional search filters

        Returns:
            List of search results

        """
        try:
            if mode in [SearchMode.VECTOR, SearchMode.BM25]:
                # Route to single search mode
                return self.router.route(
                    mode=mode,
                    query_vector=query_vector,
                    query_terms=query_terms,
                    limit=limit,
                    language_filter=language_filter,
                    filters=filters,
                )
            
            elif mode == SearchMode.HYBRID:
                # Execute both searches
                initial_limit = limit * top_k_multiplier
                
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
                
                # Combine results using RRF (Reciprocal Rank Fusion)
                return self.combiner.combine(
                    vector_results=vector_results,
                    bm25_results=bm25_results,
                    limit=limit,
                )
            
            else:
                raise ValueError(f"Unknown search mode: {mode}")
        
        except Exception as e:
            logger.error(f"Search failed with mode {mode}: {e}")
            return []
