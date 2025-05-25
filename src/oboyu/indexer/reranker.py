"""Reranker module for Oboyu indexer.

This module provides reranking capabilities using Cross-Encoder models
to improve search result quality, especially for Japanese queries in RAG applications.
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, cast

import numpy as np
from sentence_transformers import CrossEncoder

from oboyu.common.paths import EMBEDDING_CACHE_DIR

if TYPE_CHECKING:
    from oboyu.indexer.indexer import SearchResult
from oboyu.indexer.onnx_converter import get_or_convert_cross_encoder_onnx_model

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """Result after reranking with relevance score."""
    
    original_result: "SearchResult"
    rerank_score: float
    
    def to_search_result(self) -> "SearchResult":
        """Convert back to SearchResult with updated score."""
        # Use dataclasses.replace to create a new instance with updated score
        return replace(self.original_result, score=self.rerank_score)


class BaseReranker:
    """Base class for reranker implementations."""
    
    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-reranker-310m",
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        """Initialize the base reranker.
        
        Args:
            model_name: Name of the reranker model
            device: Device to run the model on
            batch_size: Batch size for reranking
            max_length: Maximum sequence length
        
        """
        self.model_name = model_name
        self.device = device
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
        """Rerank search results based on query-document relevance.
        
        Args:
            query: Search query
            results: Initial search results to rerank
            top_k: Number of top results to return (None = all)
            threshold: Minimum score threshold (None = no threshold)
            
        Returns:
            Reranked search results
        
        """
        raise NotImplementedError("Subclasses must implement rerank method")


class CrossEncoderReranker(BaseReranker):
    """Reranker using sentence-transformers CrossEncoder models."""
    
    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-reranker-310m",
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        """Initialize the CrossEncoder reranker with lazy loading."""
        super().__init__(model_name, device, batch_size, max_length)
        logger.info(f"Initializing CrossEncoder reranker with model: {model_name}")
    
    @property
    def model(self) -> CrossEncoder:
        """Lazy load the CrossEncoder model."""
        if self._model is None:
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            logger.info("CrossEncoder model loaded successfully")
        return cast(CrossEncoder, self._model)
    
    def rerank(
        self,
        query: str,
        results: List["SearchResult"],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List["SearchResult"]:
        """Rerank search results using CrossEncoder.
        
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
        pairs = [[query, result.content] for result in results]
        
        # Score in batches
        all_scores: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            scores = self.model.predict(batch)
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
        model_name: str = "cl-nagoya/ruri-v3-reranker-310m",
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the ONNX CrossEncoder reranker with lazy loading."""
        super().__init__(model_name, device, batch_size, max_length)
        self.cache_dir = cache_dir or EMBEDDING_CACHE_DIR / "models"
        self._tokenizer: Optional[Any] = None
        logger.info(f"Initializing ONNX CrossEncoder reranker with model: {model_name}")
    
    @property
    def model(self) -> Any:  # noqa: ANN401
        """Lazy load the ONNX model and tokenizer."""
        if self._model is None:
            logger.info(f"Loading ONNX CrossEncoder model: {self.model_name}")
            from oboyu.indexer.onnx_converter import ONNXCrossEncoderModel
            
            # Get or convert ONNX model
            onnx_path = get_or_convert_cross_encoder_onnx_model(
                self.model_name,
                self.cache_dir,
            )
            
            # Load ONNX model
            self._model = ONNXCrossEncoderModel(
                model_path=onnx_path,
                max_seq_length=self.max_length,
            )
            logger.info("ONNX CrossEncoder model loaded successfully")
        return self._model
    
    def rerank(
        self,
        query: str,
        results: List["SearchResult"],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List["SearchResult"]:
        """Rerank search results using ONNX CrossEncoder.
        
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
        
        logger.debug(f"ONNX reranking {len(results)} results for query: {query[:50]}...")
        
        # Prepare query-document pairs
        queries = [query] * len(results)
        documents = [result.content for result in results]
        
        # Score using ONNX model
        scores = self.model.predict(queries, documents, batch_size=self.batch_size)
        
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


def create_reranker(
    model_name: str = "cl-nagoya/ruri-v3-reranker-310m",
    use_onnx: bool = True,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 512,
    cache_dir: Optional[Path] = None,
) -> BaseReranker:
    """Create appropriate reranker instance.
    
    Args:
        model_name: Name of the reranker model
        use_onnx: Whether to use ONNX optimization
        device: Device to run the model on
        batch_size: Batch size for reranking
        max_length: Maximum sequence length
        cache_dir: Cache directory for ONNX models
        
    Returns:
        Reranker instance
    
    """
    if use_onnx and device == "cpu":
        return ONNXCrossEncoderReranker(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_dir,
        )
    else:
        return CrossEncoderReranker(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
