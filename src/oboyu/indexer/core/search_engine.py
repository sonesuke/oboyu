"""Search engine coordinator for different search modes."""

import logging
from enum import Enum
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from oboyu.indexer.search.bm25_search import BM25Search
from oboyu.indexer.search.hybrid_search import HybridSearch
from oboyu.indexer.search.search_filters import SearchFilters
from oboyu.indexer.search.search_result import SearchResult
from oboyu.indexer.search.vector_search import VectorSearch

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Available search modes."""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class SearchEngine:
    """Lightweight coordinator for different search modes."""

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
            if mode == SearchMode.VECTOR:
                if query_vector is None:
                    raise ValueError("Query vector is required for vector search")
                return self.vector_search.search(query_vector=query_vector, limit=limit, language_filter=language_filter, filters=filters)

            elif mode == SearchMode.BM25:
                if query_terms is None:
                    raise ValueError("Query terms are required for BM25 search")
                return self.bm25_search.search(terms=query_terms, limit=limit, language_filter=language_filter, filters=filters)

            elif mode == SearchMode.HYBRID:
                if query_vector is None or query_terms is None:
                    raise ValueError("Both query vector and terms are required for hybrid search")

                # Get more results for hybrid combination
                initial_limit = limit * top_k_multiplier

                # Execute both searches
                vector_results = self.vector_search.search(query_vector=query_vector, limit=initial_limit, language_filter=language_filter, filters=filters)

                bm25_results = self.bm25_search.search(terms=query_terms, limit=initial_limit, language_filter=language_filter, filters=filters)

                # Combine results
                return self.hybrid_search.search(vector_results=vector_results, bm25_results=bm25_results, limit=limit)

            else:
                raise ValueError(f"Unknown search mode: {mode}")

        except Exception as e:
            logger.error(f"Search failed with mode {mode}: {e}")
            return []
