"""Routes queries to appropriate search implementations."""

import logging
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from oboyu.common.types import SearchFilters, SearchMode, SearchResult
from oboyu.retriever.search.bm25_search import BM25Search
from oboyu.retriever.search.vector_search import VectorSearch

logger = logging.getLogger(__name__)


class SearchModeRouter:
    """Routes queries to appropriate search implementations based on search mode."""

    def __init__(
        self,
        vector_search: VectorSearch,
        bm25_search: BM25Search,
    ) -> None:
        """Initialize search mode router.

        Args:
            vector_search: Vector similarity search service
            bm25_search: BM25 keyword search service

        """
        self.vector_search = vector_search
        self.bm25_search = bm25_search

    def route(
        self,
        mode: SearchMode,
        query_vector: Optional[NDArray[np.float32]] = None,
        query_terms: Optional[List[str]] = None,
        limit: int = 10,
        language_filter: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Route search query to appropriate search implementation.

        Args:
            mode: Search mode to use
            query_vector: Query embedding vector (required for vector search)
            query_terms: Query terms (required for BM25 search)
            limit: Maximum number of results to return
            language_filter: Optional language filter
            filters: Optional search filters for date range and path filtering

        Returns:
            List of search results from the appropriate search method

        Raises:
            ValueError: If required parameters are missing for the selected mode

        """
        try:
            if mode == SearchMode.VECTOR:
                if query_vector is None:
                    raise ValueError("Query vector is required for vector search")
                return self.vector_search.search(
                    query_vector=query_vector,
                    limit=limit,
                    language_filter=language_filter,
                    filters=filters,
                )

            elif mode == SearchMode.BM25:
                if query_terms is None:
                    raise ValueError("Query terms are required for BM25 search")
                return self.bm25_search.search(
                    terms=query_terms,
                    limit=limit,
                    language_filter=language_filter,
                    filters=filters,
                )

            else:
                raise ValueError(f"Unsupported search mode for routing: {mode}")

        except Exception as e:
            logger.error(f"Search routing failed for mode {mode}: {e}")
            raise
