"""Normalizes scores from different search methods for fair comparison."""

import logging
from enum import Enum
from typing import List

from oboyu.common.types import SearchResult

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Available score normalization methods."""

    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    RANK_BASED = "rank_based"


class ScoreNormalizer:
    """Normalizes scores from different search methods to enable fair comparison."""

    def __init__(self, method: NormalizationMethod = NormalizationMethod.MIN_MAX) -> None:
        """Initialize score normalizer.

        Args:
            method: Normalization method to use

        """
        self.method = method

    def normalize_scores(
        self,
        results: List[SearchResult],
        search_method: str,
    ) -> List[SearchResult]:
        """Normalize scores for a list of search results.

        Args:
            results: List of search results to normalize
            search_method: Name of the search method (for logging)

        Returns:
            List of search results with normalized scores

        """
        if not results:
            return results

        try:
            if self.method == NormalizationMethod.MIN_MAX:
                return self._min_max_normalize(results)
            elif self.method == NormalizationMethod.Z_SCORE:
                return self._z_score_normalize(results)
            elif self.method == NormalizationMethod.RANK_BASED:
                return self._rank_based_normalize(results)
            else:
                logger.warning(f"Unknown normalization method: {self.method}. Using min-max.")
                return self._min_max_normalize(results)

        except Exception as e:
            logger.error(f"Score normalization failed for {search_method}: {e}")
            return results

    def _min_max_normalize(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply min-max normalization to scores."""
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            return results
        
        normalized_results = []
        for result in results:
            normalized_score = (result.score - min_score) / (max_score - min_score)
            normalized_results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    path=result.path,
                    title=result.title,
                    content=result.content,
                    chunk_index=result.chunk_index,
                    language=result.language,
                    metadata=result.metadata,
                    score=max(0.0, min(1.0, normalized_score)),  # Ensure in [0, 1]
                )
            )
        
        return normalized_results

    def _z_score_normalize(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply z-score normalization to scores."""
        import numpy as np
        
        scores = np.array([r.score for r in results])
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Avoid division by zero
        if std_score == 0:
            return results
        
        normalized_results = []
        for result in results:
            # Z-score normalization
            z_score = (result.score - mean_score) / std_score
            # Convert to 0-1 range using sigmoid function
            normalized_score = 1 / (1 + np.exp(-z_score))
            
            normalized_results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    path=result.path,
                    title=result.title,
                    content=result.content,
                    chunk_index=result.chunk_index,
                    language=result.language,
                    metadata=result.metadata,
                    score=normalized_score,
                )
            )
        
        return normalized_results

    def _rank_based_normalize(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply rank-based normalization to scores."""
        n = len(results)
        if n == 0:
            return results
        
        # Sort by score to get ranks
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        
        # Create normalized results with rank-based scores
        normalized_results = []
        for i, result in enumerate(sorted_results):
            # Linear rank-based score: best result gets 1.0, worst gets 0.0
            normalized_score = (n - i) / n
            
            normalized_results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    path=result.path,
                    title=result.title,
                    content=result.content,
                    chunk_index=result.chunk_index,
                    language=result.language,
                    metadata=result.metadata,
                    score=normalized_score,
                )
            )
        
        return normalized_results
