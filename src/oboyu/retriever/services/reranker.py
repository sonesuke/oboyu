"""Reranker service for improving search result relevance."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from oboyu.common.huggingface_utils import (
    get_fallback_models,
)
from oboyu.common.model_manager import RerankerModelManager

if TYPE_CHECKING:
    from oboyu.common.types import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """Result after reranking with relevance score."""

    original_result: "SearchResult"
    rerank_score: float

    def to_search_result(self) -> "SearchResult":
        """Convert back to SearchResult with updated score."""
        # Use Pydantic's model_copy to create a new instance with updated score
        return self.original_result.model_copy(update={"score": self.rerank_score})


class BaseReranker:
    """Base class for reranker implementations."""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-reranker-small",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        """Initialize the base reranker."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Optional[Any] = None

    def rerank(
        self,
        query: str,
        results: List["SearchResult"],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List["SearchResult"]:
        """Rerank search results based on query-document relevance."""
        raise NotImplementedError("Subclasses must implement rerank method")


class CrossEncoderReranker(BaseReranker):
    """Reranker using sentence-transformers CrossEncoder models."""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-reranker-small",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        """Initialize the CrossEncoder reranker with lazy loading."""
        super().__init__(model_name, batch_size, max_length)
        logger.debug(f"Initializing CrossEncoder reranker with model: {model_name}")

        # Create model manager for unified model loading
        try:
            self.model_manager = RerankerModelManager(
                model_name=model_name,
                use_onnx=False,  # This is the PyTorch version
                max_length=max_length,
            )
        except Exception as e:
            logger.error(f"Failed to initialize reranker model manager for {model_name}: {e}")
            # Suggest fallback models
            fallbacks = get_fallback_models("reranker")
            if fallbacks:
                fallback_msg = "Suggested fallback reranker models:\n"
                for model_id, description in fallbacks[:3]:
                    fallback_msg += f"• {model_id}: {description}\n"
                logger.error(fallback_msg)
            raise RuntimeError(f"Failed to initialize CrossEncoder reranker with model '{model_name}': {e}") from e

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Lazy load the CrossEncoder model."""
        return self.model_manager.model

    def rerank(
        self,
        query: str,
        results: List["SearchResult"],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List["SearchResult"]:
        """Rerank search results using CrossEncoder."""
        if not results:
            return []

        logger.debug(f"Reranking {len(results)} results for query: {query[:50]}...")

        # Prepare query-document pairs with content truncation
        pairs = []
        for result in results:
            # Truncate content to ensure it fits within max_length
            content = result.content
            if len(content) > self.max_length * 3:  # Rough character to token ratio
                content = content[:self.max_length * 3]
            pairs.append([query, content])

        # Score in batches
        all_scores: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            # Suppress tokenizer warnings completely
            import logging as base_logging
            import warnings
            
            # Save current levels
            transformers_logger = base_logging.getLogger("transformers.tokenization_utils_base")
            original_level = transformers_logger.level
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                transformers_logger.setLevel(base_logging.ERROR)
                scores = self.model.predict(batch, show_progress_bar=False)
                transformers_logger.setLevel(original_level)
            all_scores.extend(scores)

        # Convert scores to numpy array for easier manipulation
        scores_array = np.array(all_scores)

        # Create reranked results
        reranked_results: List[RerankedResult] = []
        for result, score in zip(results, scores_array):
            # Normalize score to [0, 1] range using sigmoid
            normalized_score = float(1 / (1 + np.exp(-score)))
            reranked_results.append(RerankedResult(result, normalized_score))

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


class ONNXCrossEncoderReranker(BaseReranker):
    """ONNX-optimized CrossEncoder reranker for faster CPU inference."""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-reranker-small",
        batch_size: int = 8,
        max_length: int = 512,
        cache_dir: Optional[Path] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
        optimization_level: str = "none",
    ) -> None:
        """Initialize the ONNX CrossEncoder reranker with lazy loading."""
        super().__init__(model_name, batch_size, max_length)
        logger.debug(f"Initializing ONNX CrossEncoder reranker with model: {model_name}")

        # Create model manager for unified model loading
        try:
            self.model_manager = RerankerModelManager(
                model_name=model_name,
                use_onnx=True,
                max_length=max_length,
                cache_dir=cache_dir,
                quantization_config=quantization_config or {"enabled": True, "weight_type": "uint8"},
                optimization_level=optimization_level,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ONNX reranker model manager for {model_name}: {e}")
            # Suggest fallback models
            fallbacks = get_fallback_models("reranker")
            if fallbacks:
                fallback_msg = "Suggested fallback reranker models:\n"
                for model_id, description in fallbacks[:3]:
                    fallback_msg += f"• {model_id}: {description}\n"
                logger.error(fallback_msg)
            raise RuntimeError(f"Failed to initialize ONNX CrossEncoder reranker with model '{model_name}': {e}") from e

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Lazy load the ONNX model and tokenizer."""
        return self.model_manager.model

    def rerank(
        self,
        query: str,
        results: List["SearchResult"],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List["SearchResult"]:
        """Rerank search results using ONNX CrossEncoder."""
        if not results:
            return []

        logger.debug(f"ONNX reranking {len(results)} results for query: {query[:50]}...")

        # Prepare query-document pairs
        queries = [query] * len(results)
        documents = [result.content for result in results]

        # Score using ONNX model
        import time

        predict_start = time.time()
        logger.debug(f"ONNX predict starting with {len(documents)} documents, batch_size={self.batch_size}")
        scores = self.model.predict(queries, documents, batch_size=self.batch_size)
        predict_time = time.time() - predict_start
        logger.debug(f"ONNX predict completed in {predict_time:.2f}s")

        # Create reranked results
        reranked_results: List[RerankedResult] = []
        for result, score in zip(results, scores):
            # Scores from ONNX model are already normalized
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

        logger.debug(f"ONNX reranking complete. Returning {len(final_results)} results")
        return final_results


class RerankerService:
    """Service for reranking search results."""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-reranker-small",
        use_onnx: bool = False,
        batch_size: int = 16,
        max_length: int = 512,
        quantization_config: Optional[Dict[str, Any]] = None,
        optimization_level: str = "none",
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize reranker service."""
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.batch_size = batch_size
        self.max_length = max_length

        # Initialize reranker
        self.reranker: Optional[BaseReranker] = None
        try:
            if use_onnx:
                self.reranker = ONNXCrossEncoderReranker(
                    model_name=model_name,
                    batch_size=batch_size,
                    max_length=max_length,
                    cache_dir=cache_dir,
                    quantization_config=quantization_config,
                    optimization_level=optimization_level,
                )
            else:
                self.reranker = CrossEncoderReranker(
                    model_name=model_name,
                    batch_size=batch_size,
                    max_length=max_length,
                )
        except Exception as e:
            logger.error(f"Failed to initialize reranker service: {e}")
            # Log fallback suggestions but don't fail initialization
            fallbacks = get_fallback_models("reranker")
            if fallbacks:
                fallback_msg = "Suggested fallback reranker models:\n"
                for model_id, description in fallbacks[:3]:
                    fallback_msg += f"• {model_id}: {description}\n"
                logger.error(fallback_msg)
            logger.warning("Reranker service will be disabled. Search results will not be reranked.")

    def rerank(
        self,
        query: str,
        results: List["SearchResult"],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List["SearchResult"]:
        """Rerank search results based on query relevance."""
        if not self.reranker or not results:
            return results

        try:
            return self.reranker.rerank(query, results, top_k, threshold)
        except Exception as e:
            # Check if this is a model loading error
            if isinstance(e, RuntimeError) and "Failed to load reranker model" in str(e):
                logger.error(f"Reranking failed due to model loading error: {e}")
                # Suggest user to try different model or check connectivity
                logger.error("Consider checking your internet connection or trying a different reranker model.")
            else:
                logger.error(f"Reranking failed: {e}")
            logger.warning("Returning original search results without reranking.")
            return results

    def is_available(self) -> bool:
        """Check if reranker is available."""
        return self.reranker is not None


# Factory function for backward compatibility
def create_reranker(
    model_name: str = "cl-nagoya/ruri-reranker-small",
    use_onnx: bool = True,
    batch_size: int = 8,
    max_length: int = 512,
    cache_dir: Optional[Path] = None,
    quantization_config: Optional[Dict[str, Any]] = None,
    optimization_level: str = "none",
) -> BaseReranker:
    """Create appropriate reranker instance."""
    if use_onnx:
        return ONNXCrossEncoderReranker(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            optimization_level=optimization_level,
        )
    else:
        return CrossEncoderReranker(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
        )
