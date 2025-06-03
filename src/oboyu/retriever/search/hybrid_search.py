"""Hybrid search combining vector and BM25 search results."""

import logging
from typing import Dict, List

from oboyu.common.types import SearchResult

logger = logging.getLogger(__name__)


class HybridSearch:
    """Hybrid search combining vector and BM25 results."""

    def __init__(self, vector_weight: float = 0.7, bm25_weight: float = 0.3) -> None:
        """Initialize hybrid search.

        Args:
            vector_weight: Weight for vector search results (0-1)
            bm25_weight: Weight for BM25 search results (0-1)

        """
        # Normalize weights to ensure they sum to 1
        total_weight = vector_weight + bm25_weight
        if total_weight > 0:
            self.vector_weight = vector_weight / total_weight
            self.bm25_weight = bm25_weight / total_weight
        else:
            self.vector_weight = 0.7
            self.bm25_weight = 0.3

    def search(self, vector_results: List[SearchResult], bm25_results: List[SearchResult], limit: int) -> List[SearchResult]:
        """Combine and weight search results.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            limit: Maximum number of results to return

        Returns:
            Combined and reweighted search results

        """
        try:
            # Create a map of chunk_id to combined scores
            combined_scores: Dict[str, float] = {}
            results_map: Dict[str, SearchResult] = {}

            # Add vector search results
            for result in vector_results:
                chunk_id = result.chunk_id
                weighted_score = float(result.score) * self.vector_weight
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + weighted_score
                results_map[chunk_id] = result

            # Add BM25 search results
            for result in bm25_results:
                chunk_id = result.chunk_id
                weighted_score = float(result.score) * self.bm25_weight
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + weighted_score

                # Use the result from BM25 if not already in map (or update with BM25 version)
                if chunk_id not in results_map:
                    results_map[chunk_id] = result

            # Sort by combined score
            sorted_chunk_ids = sorted(combined_scores.keys(), key=lambda chunk_id: combined_scores[chunk_id], reverse=True)

            # Create final results with updated scores
            final_results = []
            for chunk_id in sorted_chunk_ids[:limit]:
                result = results_map[chunk_id]
                # Update the score to the combined score
                final_result = SearchResult(
                    chunk_id=result.chunk_id,
                    path=result.path,
                    title=result.title,
                    content=result.content,
                    chunk_index=result.chunk_index,
                    language=result.language,
                    metadata=result.metadata,
                    score=combined_scores[chunk_id],
                )
                final_results.append(final_result)

            return final_results

        except Exception as e:
            logger.error(f"Hybrid search combination failed: {e}")
            # Fallback to vector results if combination fails
            return vector_results[:limit] if vector_results else bm25_results[:limit]
