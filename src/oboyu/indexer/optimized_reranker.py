"""Optimized reranker using sentence-transformers ONNX backend."""

import logging
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from oboyu.indexer.indexer import SearchResult

from oboyu.indexer.reranker import BaseReranker, RerankedResult

logger = logging.getLogger(__name__)


class OptimizedONNXReranker(BaseReranker):
    """Optimized reranker using sentence-transformers built-in ONNX backend."""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-reranker-small",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        """Initialize the optimized ONNX reranker.

        Args:
            model_name: Name of the reranker model
            device: Device to run the model on (cpu/cuda)
            batch_size: Batch size for reranking
            max_length: Maximum sequence length

        """
        super().__init__(model_name, device, batch_size, max_length)
        logger.debug(f"Initializing optimized ONNX reranker with model: {model_name}")

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Lazy load the CrossEncoder with ONNX backend."""
        if self._model is None:
            logger.debug(f"Loading CrossEncoder with ONNX backend: {self.model_name}")
            import os

            # Check if we already have an ONNX model cached
            from huggingface_hub import hf_hub_download
            from sentence_transformers import CrossEncoder
            try:
                # Try to download pre-exported ONNX model
                onnx_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename="model.onnx",
                    cache_dir=None,  # Use default cache
                    local_files_only=False,
                )
                logger.debug(f"Found pre-exported ONNX model at: {onnx_path}")
            except Exception:
                logger.debug("No pre-exported ONNX model found, will export during first use")

            # Use ONNX backend for better performance
            try:
                self._model = CrossEncoder(
                    self.model_name,
                    backend="onnx",
                    device=self.device,
                    max_length=self.max_length,
                    trust_remote_code=True,
                    model_kwargs={
                        "provider": "CPUExecutionProvider" if self.device == "cpu" else "CUDAExecutionProvider",
                        "session_options": {
                            "intra_op_num_threads": os.cpu_count() or 4,
                            "inter_op_num_threads": 1,
                        }
                    },
                )
                logger.debug("CrossEncoder with ONNX backend loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load ONNX backend, falling back to PyTorch: {e}")
                # Fallback to PyTorch if ONNX fails
                self._model = CrossEncoder(
                    self.model_name,
                    backend="torch",
                    device=self.device,
                    max_length=self.max_length,
                    trust_remote_code=True,
                )
                logger.debug("Fallback to PyTorch backend loaded successfully")
        return self._model

    def rerank(
        self,
        query: str,
        results: List["SearchResult"],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List["SearchResult"]:
        """Rerank search results using optimized ONNX CrossEncoder.

        Args:
            query: Search query
            results: Initial search results to rerank
            top_k: Number of top results to return
            threshold: Minimum score threshold

        Returns:
            Reranked search results

        """
        if not results:
            return []

        logger.debug(f"Reranking {len(results)} results for query: {query[:50]}...")

        # Prepare query-document pairs
        pairs = [(query, result.content) for result in results]

        # Use batch prediction with ONNX backend
        import time
        
        predict_start = time.time()
        logger.debug(f"ONNX (optimized) predict starting with {len(pairs)} documents")
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        predict_time = time.time() - predict_start
        logger.debug(f"ONNX (optimized) predict completed in {predict_time:.2f}s")

        # Create reranked results
        reranked_results: List[RerankedResult] = []
        for result, score in zip(results, scores):
            reranked_results.append(RerankedResult(result, float(score)))

        # Sort by rerank score (descending)
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Apply threshold if specified
        if threshold is not None:
            reranked_results = [r for r in reranked_results if r.rerank_score >= threshold]

        # Apply top_k if specified
        if top_k is not None:
            reranked_results = reranked_results[:top_k]

        # Convert back to SearchResult objects
        final_results = [r.to_search_result() for r in reranked_results]

        logger.debug(f"Reranking complete. Returning {len(final_results)} results")
        return final_results
