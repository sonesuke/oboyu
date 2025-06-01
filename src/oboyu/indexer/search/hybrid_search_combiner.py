"""Combines vector and BM25 results with configurable weights."""

import logging
from typing import Dict, List, Optional

from oboyu.indexer.search.score_normalizer import ScoreNormalizer
from oboyu.indexer.search.search_result import SearchResult

logger = logging.getLogger(__name__)


class HybridSearchCombiner:
    """Combines vector and BM25 search results with configurable weights."""

    def __init__(
        self,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        score_normalizer: Optional[ScoreNormalizer] = None,
    ) -> None:
        """Initialize hybrid search combiner.

        Args:
            vector_weight: Weight for vector search results (0-1)
            bm25_weight: Weight for BM25 search results (0-1)
            score_normalizer: Optional score normalizer for preprocessing scores

        """
        # Normalize weights to ensure they sum to 1
        total_weight = vector_weight + bm25_weight
        if total_weight > 0:
            self.vector_weight = vector_weight / total_weight
            self.bm25_weight = bm25_weight / total_weight
        else:
            self.vector_weight = 0.7
            self.bm25_weight = 0.3
        
        self.score_normalizer = score_normalizer

    def combine(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        limit: int = 10,
    ) -> List[SearchResult]:
        """Combine vector and BM25 results with weighted scores.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            limit: Maximum number of results to return

        Returns:
            Combined search results with weighted scores

        """
        try:
            # Optionally normalize scores before combining
            if self.score_normalizer:
                vector_results = self.score_normalizer.normalize_scores(
                    vector_results, "vector"
                )
                bm25_results = self.score_normalizer.normalize_scores(
                    bm25_results, "bm25"
                )

            # Create a map of chunk_id to combined scores
            combined_scores: Dict[str, float] = {}
            results_map: Dict[str, SearchResult] = {}

            # Add vector search results with weights
            for result in vector_results:
                chunk_id = result.chunk_id
                weighted_score = float(result.score) * self.vector_weight
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + weighted_score
                results_map[chunk_id] = result

            # Add BM25 search results with weights
            for result in bm25_results:
                chunk_id = result.chunk_id
                weighted_score = float(result.score) * self.bm25_weight
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + weighted_score

                # Use the result from BM25 if not already in map
                if chunk_id not in results_map:
                    results_map[chunk_id] = result

            # Sort by combined score
            sorted_chunk_ids = sorted(
                combined_scores.keys(),
                key=lambda chunk_id: combined_scores[chunk_id],
                reverse=True
            )

            # Create final results with updated scores
            final_results = []
            for chunk_id in sorted_chunk_ids[:limit]:
                result = results_map[chunk_id]
                # Create new result with combined score
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
