"""Merges search results from multiple search methods."""

import logging
from typing import Dict, List

from oboyu.common.types import SearchResult

logger = logging.getLogger(__name__)


class ResultMerger:
    """Merges and deduplicates results from multiple search methods."""

    def merge(
        self,
        *result_lists: List[SearchResult],
        limit: int = 10,
    ) -> List[SearchResult]:
        """Merge multiple lists of search results.

        Deduplicates results by chunk_id, keeping the result with the highest score.

        Args:
            *result_lists: Variable number of search result lists to merge
            limit: Maximum number of results to return

        Returns:
            Merged and deduplicated list of search results, sorted by score

        """
        try:
            # Create a map to track the best result for each chunk_id
            best_results: Dict[str, SearchResult] = {}

            # Process each result list
            for results in result_lists:
                for result in results:
                    chunk_id = result.chunk_id
                    
                    # Keep the result with the highest score
                    if chunk_id not in best_results or result.score > best_results[chunk_id].score:
                        best_results[chunk_id] = result

            # Sort by score and return top results
            sorted_results = sorted(
                best_results.values(),
                key=lambda r: r.score,
                reverse=True
            )
            
            return sorted_results[:limit]

        except Exception as e:
            logger.error(f"Result merging failed: {e}")
            # Fallback: return first non-empty result list
            for results in result_lists:
                if results:
                    return results[:limit]
            return []
