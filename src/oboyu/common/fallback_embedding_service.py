"""Fallback embedding service with circuit breaker and caching support."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from oboyu.common.circuit_breaker import (
    CircuitBreakerError,
    get_circuit_breaker_registry,
)
from oboyu.common.huggingface_utils import (
    HuggingFaceError,
    get_fallback_models,
    get_user_friendly_error_message,
)
from oboyu.indexer.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    """Fallback service using simple local embeddings when external services fail."""

    def __init__(self, dimensions: int = 256) -> None:
        """Initialize local embedding service.
        
        Args:
            dimensions: Dimension size for generated embeddings.
            
        """
        self.dimensions = dimensions
        self._word_vectors: dict[str, NDArray[np.float32]] = {}

    def generate_embeddings(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Generate simple hash-based embeddings as fallback.
        
        Args:
            texts: List of texts to generate embeddings for.
            
        Returns:
            List of embedding arrays.
            
        """
        embeddings = []
        for text in texts:
            if text in self._word_vectors:
                embeddings.append(self._word_vectors[text])
            else:
                # Generate deterministic embedding based on text hash
                text_hash = hash(text)
                # Create pseudo-random but deterministic vector
                np.random.seed(abs(text_hash) % (2**32))
                embedding = np.random.normal(0, 1, self.dimensions).astype(np.float32)
                # Normalize to unit vector
                embedding = embedding / np.linalg.norm(embedding)
                self._word_vectors[text] = embedding
                embeddings.append(embedding)
        
        return embeddings

    def generate_query_embedding(self, query: str) -> NDArray[np.float32]:
        """Generate embedding for a search query.
        
        Args:
            query: Search query text.
            
        Returns:
            Query embedding array.
            
        """
        return self.generate_embeddings([query])[0]

    def get_dimensions(self) -> int:
        """Get the dimension size of embeddings."""
        return self.dimensions

    def get_model_name(self) -> str:
        """Get the model name."""
        return "local_fallback"


class FallbackEmbeddingService:
    """Enhanced embedding service with circuit breaker and comprehensive fallback support."""

    def __init__(
        self,
        primary_service: EmbeddingService | None = None,
        model_name: str = "cl-nagoya/ruri-v3-30m",
        batch_size: int = 64,
        max_seq_length: int = 8192,
        query_prefix: str = "検索クエリ: ",
        use_onnx: bool = True,
        use_cache: bool = True,
        use_circuit_breaker: bool = True,
        enable_fallback_services: bool = True,
        enable_local_fallback: bool = True,
        fallback_model_names: list[str] | None = None,
        onnx_quantization_config: dict[str, Any] | None = None,
        onnx_optimization_level: str = "extended",
    ) -> None:
        """Initialize fallback embedding service.
        
        Args:
            primary_service: Primary embedding service to use.
            model_name: Name of the primary embedding model.
            batch_size: Batch size for embedding generation.
            max_seq_length: Maximum sequence length.
            query_prefix: Prefix for search queries.
            use_onnx: Whether to use ONNX optimization.
            use_cache: Whether to use embedding cache.
            use_circuit_breaker: Whether to use circuit breaker protection.
            enable_fallback_services: Whether to enable fallback services.
            enable_local_fallback: Whether to enable local fallback.
            fallback_model_names: List of fallback model names to try.
            onnx_quantization_config: ONNX quantization configuration.
            onnx_optimization_level: ONNX optimization level.
            
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.query_prefix = query_prefix
        self.use_onnx = use_onnx
        self.use_cache = use_cache
        self.use_circuit_breaker = use_circuit_breaker

        # Initialize primary service
        if primary_service is not None:
            self.primary_service = primary_service
        else:
            try:
                self.primary_service = EmbeddingService(
                    model_name=model_name,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    query_prefix=query_prefix,
                    use_onnx=use_onnx,
                    use_cache=use_cache,
                    onnx_quantization_config=onnx_quantization_config,
                    onnx_optimization_level=onnx_optimization_level,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize primary embedding service: {e}")
                self.primary_service = None

        # Initialize fallback services
        self.fallback_services: list[EmbeddingService] = []
        self.local_fallback: LocalEmbeddingService | None = None
        
        # Setup fallback model names
        if enable_fallback_services:
            if fallback_model_names is None:
                fallback_models = get_fallback_models("embedding")
                fallback_model_names = [model[0] for model in fallback_models[:3]]
            
            # Initialize fallback services for each fallback model
            for fallback_model in fallback_model_names:
                if fallback_model != model_name:  # Don't duplicate primary model
                    try:
                        fallback_service = EmbeddingService(
                            model_name=fallback_model,
                            batch_size=batch_size,
                            max_seq_length=max_seq_length,
                            query_prefix=query_prefix,
                            use_onnx=use_onnx,
                            use_cache=use_cache,
                            onnx_quantization_config=onnx_quantization_config,
                            onnx_optimization_level=onnx_optimization_level,
                        )
                        self.fallback_services.append(fallback_service)
                        logger.info(f"Initialized fallback embedding service: {fallback_model}")
                    except Exception as e:
                        logger.warning(f"Failed to initialize fallback embedding service {fallback_model}: {e}")

        # Initialize local fallback service
        if enable_local_fallback:
            try:
                primary_dims = self.primary_service.dimensions if self.primary_service else 256
                self.local_fallback = LocalEmbeddingService(dimensions=primary_dims or 256)
                logger.info("Initialized local fallback embedding service")
            except Exception as e:
                logger.warning(f"Failed to initialize local fallback service: {e}")

        # Get circuit breaker if enabled
        self.circuit_breaker = None
        if use_circuit_breaker:
            registry = get_circuit_breaker_registry()
            self.circuit_breaker = registry.get_or_create(f"embedding_{model_name}")

    def generate_embeddings(
        self,
        texts: list[str],
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[NDArray[np.float32]]:
        """Generate embeddings with fallback support.
        
        Args:
            texts: List of texts to generate embeddings for.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            List of embedding arrays.
            
        """
        if not texts:
            return []

        # Try primary service first
        if self.primary_service:
            try:
                if self.use_circuit_breaker and self.circuit_breaker:
                    return self.circuit_breaker.call(
                        lambda: self.primary_service.generate_embeddings(texts, progress_callback)
                    )
                else:
                    return self.primary_service.generate_embeddings(texts, progress_callback)
            except CircuitBreakerError as e:
                logger.warning(f"Circuit breaker open for primary embedding service: {e}")
                self._log_fallback_attempt("primary service circuit breaker open")
            except HuggingFaceError as e:
                logger.warning(f"HuggingFace error in primary embedding service: {e.message}")
                self._log_fallback_attempt(f"HuggingFace error: {e.message}")
                self._log_user_friendly_error(e)
            except Exception as e:
                logger.warning(f"Unexpected error in primary embedding service: {e}")
                self._log_fallback_attempt(f"unexpected error: {e}")

        # Try fallback services
        for i, fallback_service in enumerate(self.fallback_services):
            try:
                logger.info(f"Trying fallback embedding service {i+1}/{len(self.fallback_services)}: {fallback_service.model_name}")
                if self.use_circuit_breaker:
                    registry = get_circuit_breaker_registry()
                    fallback_circuit_breaker: Any = registry.get_or_create(f"embedding_{fallback_service.model_name}")
                    return fallback_circuit_breaker.call(
                        lambda: fallback_service.generate_embeddings(texts, progress_callback)
                    )
                else:
                    return fallback_service.generate_embeddings(texts, progress_callback)
            except CircuitBreakerError as e:
                logger.warning(f"Circuit breaker open for fallback service {fallback_service.model_name}: {e}")
                continue
            except HuggingFaceError as e:
                logger.warning(f"HuggingFace error in fallback service {fallback_service.model_name}: {e.message}")
                continue
            except Exception as e:
                logger.warning(f"Error in fallback service {fallback_service.model_name}: {e}")
                continue

        # Use local fallback as last resort
        if self.local_fallback:
            logger.warning("Using local fallback embedding service")
            try:
                return self.local_fallback.generate_embeddings(texts)
            except Exception as e:
                logger.error(f"Even local fallback failed: {e}")

        # If everything fails, return zero embeddings
        logger.error("All embedding services failed, returning zero embeddings")
        dimensions = self._get_dimensions()
        return [np.zeros(dimensions, dtype=np.float32) for _ in texts]

    def generate_query_embedding(self, query: str) -> NDArray[np.float32]:
        """Generate embedding for a search query with fallback support.
        
        Args:
            query: Search query text.
            
        Returns:
            Query embedding array.
            
        """
        embeddings = self.generate_embeddings([f"{self.query_prefix}{query}"])
        return embeddings[0] if embeddings else np.zeros(self._get_dimensions(), dtype=np.float32)

    def _get_dimensions(self) -> int:
        """Get embedding dimensions from available services."""
        if self.primary_service and hasattr(self.primary_service, 'dimensions') and self.primary_service.dimensions:
            return self.primary_service.dimensions
        
        for fallback_service in self.fallback_services:
            if hasattr(fallback_service, 'dimensions') and fallback_service.dimensions:
                return fallback_service.dimensions
        
        if self.local_fallback:
            return self.local_fallback.get_dimensions()
        
        return 256  # Default fallback

    def _log_fallback_attempt(self, reason: str) -> None:
        """Log fallback attempt with reason."""
        logger.warning(f"Falling back from primary embedding service due to: {reason}")

    def _log_user_friendly_error(self, error: HuggingFaceError) -> None:
        """Log user-friendly error message."""
        friendly_message = get_user_friendly_error_message(error)
        logger.error(f"User-friendly error message:\n{friendly_message}")

    def get_model_name(self) -> str:
        """Get the name of the current embedding model."""
        if self.primary_service and hasattr(self.primary_service, 'model_name'):
            return self.primary_service.model_name
        elif self.fallback_services:
            return self.fallback_services[0].model_name
        elif self.local_fallback:
            return self.local_fallback.get_model_name()
        else:
            return self.model_name  # Return the stored model name

    def get_dimensions(self) -> int:
        """Get the dimension size of embeddings."""
        return self._get_dimensions()

    def is_available(self) -> bool:
        """Check if any embedding service is available."""
        # Primary service availability
        if self.primary_service:
            if self.circuit_breaker:
                from oboyu.common.circuit_breaker import CircuitState
                if self.circuit_breaker.get_state() == CircuitState.CLOSED:
                    return True
            else:
                return True

        # Fallback services availability
        if self.fallback_services:
            if self.use_circuit_breaker:
                registry = get_circuit_breaker_registry()
                for fallback_service in self.fallback_services:
                    circuit_breaker = registry.get_circuit_breaker(f"embedding_{fallback_service.model_name}")
                    if circuit_breaker is None or circuit_breaker.get_state() == CircuitState.CLOSED:
                        return True
            else:
                return True

        # Local fallback is always available
        return self.local_fallback is not None

    def clear_cache(self) -> None:
        """Clear embedding cache for all services."""
        if self.primary_service:
            self.primary_service.clear_cache()
        
        for fallback_service in self.fallback_services:
            fallback_service.clear_cache()

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        status: dict[str, Any] = {}
        
        if self.circuit_breaker:
            status["primary"] = {
                "model": self.model_name,
                "state": self.circuit_breaker.get_state().value,
                "metrics": {
                    "total_requests": self.circuit_breaker.get_metrics().total_requests,
                    "successful_requests": self.circuit_breaker.get_metrics().successful_requests,
                    "failed_requests": self.circuit_breaker.get_metrics().failed_requests,
                    "rejected_requests": self.circuit_breaker.get_metrics().rejected_requests,
                    "failure_rate": self.circuit_breaker.get_metrics().get_failure_rate(),
                }
            }

        if self.use_circuit_breaker:
            registry = get_circuit_breaker_registry()
            status["fallbacks"] = []
            for fallback_service in self.fallback_services:
                circuit_breaker = registry.get_circuit_breaker(f"embedding_{fallback_service.model_name}")
                if circuit_breaker:
                    status["fallbacks"].append({
                        "model": fallback_service.model_name,
                        "state": circuit_breaker.get_state().value,
                        "metrics": {
                            "total_requests": circuit_breaker.get_metrics().total_requests,
                            "successful_requests": circuit_breaker.get_metrics().successful_requests,
                            "failed_requests": circuit_breaker.get_metrics().failed_requests,
                            "rejected_requests": circuit_breaker.get_metrics().rejected_requests,
                            "failure_rate": circuit_breaker.get_metrics().get_failure_rate(),
                        }
                    })

        return status
