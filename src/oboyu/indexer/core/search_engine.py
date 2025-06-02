"""Search engine coordinator for different search modes."""

import logging
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from oboyu.retriever.search.bm25_search import BM25Search
from oboyu.retriever.search.hybrid_search import HybridSearch
from oboyu.retriever.search.hybrid_search_combiner import HybridSearchCombiner
from oboyu.retriever.search.mode_router import SearchModeRouter
from oboyu.retriever.search.result_merger import ResultMerger
from oboyu.retriever.search.score_normalizer import ScoreNormalizer
from oboyu.retriever.search.search_filters import SearchFilters
from oboyu.retriever.search.search_mode import SearchMode
from oboyu.retriever.search.search_result import SearchResult
from oboyu.retriever.search.vector_search import VectorSearch

logger = logging.getLogger(__name__)


class SearchEngine:
    """Simplified search engine that orchestrates focused components."""

    def __init__(
        self,
        vector_search: VectorSearch,
        bm25_search: BM25Search,
        hybrid_search: HybridSearch,
    ) -> None:
        """Initialize search engine.

        Args:
            vector_search: Vector similarity search service
            bm25_search: BM25 keyword search service
            hybrid_search: Hybrid search combination service

        """
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.hybrid_search = hybrid_search
        
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
        """Execute search using appropriate search mode.

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
                # For backward compatibility, we still use the old hybrid_search as well
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
