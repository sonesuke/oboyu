"""Combines vector and BM25 results using RRF (Reciprocal Rank Fusion)."""

import logging
from typing import Dict, List, Optional

from oboyu.common.types import SearchResult
from oboyu.retriever.search.score_normalizer import ScoreNormalizer

logger = logging.getLogger(__name__)


class HybridSearchCombiner:
    """Combines vector and BM25 search results using RRF (Reciprocal Rank Fusion)."""

    def __init__(
        self,
        rrf_k: int = 60,
        score_normalizer: Optional[ScoreNormalizer] = None,
        # Deprecated parameters for backward compatibility
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> None:
        """Initialize hybrid search combiner with RRF.

        Args:
            rrf_k: RRF parameter (default: 60)
            score_normalizer: Optional score normalizer for preprocessing scores
            vector_weight: DEPRECATED - RRF doesn't use weights
            bm25_weight: DEPRECATED - RRF doesn't use weights

        """
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        
        self.rrf_k = rrf_k
        self.score_normalizer = score_normalizer
        
        # Issue deprecation warning for weight parameters
        if vector_weight is not None or bm25_weight is not None:
            logger.warning(
                "vector_weight and bm25_weight parameters are deprecated. "
                "RRF (Reciprocal Rank Fusion) does not use weights. "
                "Use rrf_k parameter instead."
            )

    def combine(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        limit: int = 10,
    ) -> List[SearchResult]:
        """Combine vector and BM25 results using RRF (Reciprocal Rank Fusion).

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            limit: Maximum number of results to return

        Returns:
            Combined search results with RRF scores

        """
        try:
            # Optionally normalize scores before combining (preserving for compatibility)
            if self.score_normalizer:
                vector_results = self.score_normalizer.normalize_scores(
                    vector_results, "vector"
                )
                bm25_results = self.score_normalizer.normalize_scores(
                    bm25_results, "bm25"
                )

            # Create rank maps for RRF calculation
            vector_ranks: Dict[str, int] = {}
            bm25_ranks: Dict[str, int] = {}
            results_map: Dict[str, SearchResult] = {}

            # Build vector search rank map
            for rank, result in enumerate(vector_results, start=1):
                vector_ranks[result.chunk_id] = rank
                results_map[result.chunk_id] = result

            # Build BM25 search rank map
            for rank, result in enumerate(bm25_results, start=1):
                bm25_ranks[result.chunk_id] = rank
                # Use BM25 result if not already in map (prefer vector result metadata)
                if result.chunk_id not in results_map:
                    results_map[result.chunk_id] = result

            # Calculate RRF scores: S_hybrid(d) = 1/(k + R_vector(d)) + 1/(k + R_fulltext(d))
            rrf_scores: Dict[str, float] = {}
            for chunk_id in results_map:
                vector_rank = vector_ranks.get(chunk_id, float('inf'))
                bm25_rank = bm25_ranks.get(chunk_id, float('inf'))
                
                rrf_score = 0.0
                if vector_rank != float('inf'):
                    rrf_score += 1.0 / (self.rrf_k + vector_rank)
                if bm25_rank != float('inf'):
                    rrf_score += 1.0 / (self.rrf_k + bm25_rank)
                
                rrf_scores[chunk_id] = rrf_score

            # Sort by RRF score (higher is better)
            sorted_chunk_ids = sorted(
                rrf_scores.keys(),
                key=lambda chunk_id: rrf_scores[chunk_id],
                reverse=True
            )

            # Create final results with RRF scores
            final_results = []
            for chunk_id in sorted_chunk_ids[:limit]:
                result = results_map[chunk_id]
                # Create new result with RRF score
                final_result = SearchResult(
                    chunk_id=result.chunk_id,
                    path=result.path,
                    title=result.title,
                    content=result.content,
                    chunk_index=result.chunk_index,
                    language=result.language,
                    metadata=result.metadata,
                    score=rrf_scores[chunk_id],
                )
                final_results.append(final_result)

            return final_results

        except Exception as e:
            logger.error(f"RRF hybrid search combination failed: {e}")
            # Fallback to vector results if combination fails
            return vector_results[:limit] if vector_results else bm25_results[:limit]
