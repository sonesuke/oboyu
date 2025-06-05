"""HuggingFace implementation of embedding service with circuit breaker support."""

import logging
from typing import Any, List, Optional

from ...common.fallback_embedding_service import FallbackEmbeddingService
from ...domain.value_objects.embedding_vector import EmbeddingVector
from ...indexer.services.embedding import EmbeddingService as IndexerEmbeddingService
from ...ports.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingService(EmbeddingService):
    """HuggingFace implementation of embedding service with circuit breaker and fallback support."""
    
    def __init__(
        self,
        embedding_service: Optional[IndexerEmbeddingService] = None,
        model_name: str = "cl-nagoya/ruri-v3-30m",
        use_circuit_breaker: bool = True,
        use_fallback: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize with embedding service and circuit breaker support.
        
        Args:
            embedding_service: Existing embedding service (for backward compatibility).
            model_name: Name of the embedding model.
            use_circuit_breaker: Whether to use circuit breaker protection.
            use_fallback: Whether to use fallback services.
            **kwargs: Additional arguments for FallbackEmbeddingService.
            
        """
        if embedding_service is not None:
            # Backward compatibility - wrap existing service
            self._embedding_service = embedding_service
            self._fallback_service = None
        else:
            # Use new fallback service with circuit breaker support
            self._embedding_service = None
            self._fallback_service = FallbackEmbeddingService(
                model_name=model_name,
                use_circuit_breaker=use_circuit_breaker,
                enable_fallback_services=use_fallback,
                enable_local_fallback=use_fallback,
                **kwargs,
            )
    
    async def generate_embedding(self, text: str) -> EmbeddingVector:
        """Generate embedding for a single text."""
        if self._fallback_service:
            vector_arrays = self._fallback_service.generate_embeddings([text])
            vector_array = vector_arrays[0] if vector_arrays else None
        else:
            vector_array = await self._embedding_service.generate_embedding(text)
        
        if vector_array is not None:
            return EmbeddingVector.create(vector_array.tolist())
        else:
            # Fallback to zero vector
            dimensions = self.get_dimensions()
            return EmbeddingVector.create([0.0] * dimensions)
    
    async def generate_embeddings(self, texts: List[str]) -> List[EmbeddingVector]:
        """Generate embeddings for multiple texts."""
        if self._fallback_service:
            vector_arrays = self._fallback_service.generate_embeddings(texts)
        else:
            vector_arrays = await self._embedding_service.generate_embeddings(texts)
        
        return [EmbeddingVector.create(vector.tolist()) for vector in vector_arrays]
    
    async def generate_query_embedding(self, query_text: str) -> EmbeddingVector:
        """Generate embedding for a search query."""
        if self._fallback_service:
            vector_array = self._fallback_service.generate_query_embedding(query_text)
        else:
            vector_array = await self._embedding_service.generate_query_embedding(query_text)
        
        return EmbeddingVector.create(vector_array.tolist())
    
    def get_dimensions(self) -> int:
        """Get the dimension size of embeddings produced by this service."""
        if self._fallback_service:
            return self._fallback_service.get_dimensions()
        else:
            return self._embedding_service.get_dimensions()
    
    def get_model_name(self) -> str:
        """Get the name of the embedding model."""
        if self._fallback_service:
            return self._fallback_service.get_model_name()
        else:
            return self._embedding_service.get_model_name()
    
    async def is_available(self) -> bool:
        """Check if the embedding service is available."""
        if self._fallback_service:
            return self._fallback_service.is_available()
        else:
            return await self._embedding_service.is_available()
            
    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Get circuit breaker status for monitoring.
        
        Returns:
            Dictionary containing circuit breaker status information.
            
        """
        if self._fallback_service:
            return self._fallback_service.get_circuit_breaker_status()
        else:
            return {"status": "circuit_breaker_not_enabled"}
            
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self._fallback_service:
            self._fallback_service.clear_cache()
        elif hasattr(self._embedding_service, 'clear_cache'):
            self._embedding_service.clear_cache()
