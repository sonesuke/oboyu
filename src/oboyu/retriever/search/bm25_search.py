"""BM25 keyword search implementation."""

import json
import logging
from typing import TYPE_CHECKING, List, Optional

from oboyu.common.types import SearchFilters

if TYPE_CHECKING:
    from oboyu.indexer.storage.database_service import DatabaseService
from oboyu.common.types import SearchResult

logger = logging.getLogger(__name__)


class BM25Search:
    """BM25 keyword search service."""

    def __init__(self, database_service: "DatabaseService") -> None:
        """Initialize BM25 search.

        Args:
            database_service: Database service for search operations

        """
        self.database_service = database_service

    def search(self, terms: List[str], limit: int, language_filter: Optional[str] = None, filters: Optional[SearchFilters] = None) -> List[SearchResult]:
        """Execute BM25 keyword search.

        Args:
            terms: List of search terms
            limit: Maximum number of results to return
            language_filter: Optional language filter
            filters: Optional search filters for date range and path filtering

        Returns:
            List of search results ordered by BM25 score

        """
        try:
            # Execute BM25 search through database service
            raw_results = self.database_service.bm25_search(terms=terms, limit=limit, language_filter=language_filter, filters=filters)

            # Convert to SearchResult objects
            search_results = []
            for result in raw_results:
                try:
                    # Parse metadata if it's a string, handle None case
                    metadata = result.get("metadata", {})
                    if metadata is None:
                        metadata = {}
                    elif isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    search_result = SearchResult(
                        chunk_id=result["id"],
                        path=result["path"],
                        title=result["title"],
                        content=result["content"],
                        chunk_index=result["chunk_index"],
                        language=result["language"],
                        metadata=metadata,
                        score=result.get("score", 0.0),
                    )
                    search_results.append(search_result)

                except (KeyError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to parse search result: {e}")
                    continue

            return search_results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
