"""Reranker service for improving search result relevance."""

import logging
from typing import List, Optional

from oboyu.indexer.reranker import BaseReranker, create_reranker
from oboyu.indexer.search.search_result import SearchResult

logger = logging.getLogger(__name__)


class RerankerService:
    """Service for reranking search results."""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-reranker-small",
        use_onnx: bool = False,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
        quantization_config: Optional[dict] = None,
        optimization_level: str = "none",
    ) -> None:
        """Initialize reranker service.

        Args:
            model_name: Name of the reranker model
            use_onnx: Whether to use ONNX optimization
            device: Device to run model on
            batch_size: Batch size for reranking
            max_length: Maximum sequence length
            quantization_config: ONNX quantization configuration
            optimization_level: ONNX optimization level

        """
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        # Initialize reranker
        self.reranker: Optional[BaseReranker] = None
        try:
            self.reranker = create_reranker(
                model_name=model_name,
                use_onnx=use_onnx,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                quantization_config=quantization_config,
                optimization_level=optimization_level,
            )
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results based on query relevance.

        Args:
            query: Original search query
            results: List of search results to rerank

        Returns:
            Reranked search results

        """
        if not self.reranker or not results:
            return results

        try:
            # Prepare query-document pairs
            pairs = [(query, result.content) for result in results]
            
            # Get reranking scores
            scores = self.reranker.rerank(pairs)
            
            # Create new results with updated scores
            reranked_results = []
            for result, score in zip(results, scores):
                reranked_result = SearchResult(
                    chunk_id=result.chunk_id,
                    path=result.path,
                    title=result.title,
                    content=result.content,
                    chunk_index=result.chunk_index,
                    language=result.language,
                    metadata=result.metadata,
                    score=score
                )
                reranked_results.append(reranked_result)
            
            # Sort by new scores
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            return reranked_results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    def is_available(self) -> bool:
        """Check if reranker is available.

        Returns:
            True if reranker is available, False otherwise

        """
        return self.reranker is not None
