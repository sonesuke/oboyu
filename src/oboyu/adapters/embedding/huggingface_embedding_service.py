"""HuggingFace implementation of embedding service."""

import logging
from typing import List

from ...domain.value_objects.embedding_vector import EmbeddingVector
from ...ports.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingService(EmbeddingService):
    """HuggingFace implementation of embedding service."""
    
    def __init__(self, embedding_service):
        """Initialize with existing embedding service."""
        self._embedding_service = embedding_service
    
    async def generate_embedding(self, text: str) -> EmbeddingVector:
        """Generate embedding for a single text."""
        vector_array = await self._embedding_service.generate_embedding(text)
        return EmbeddingVector.create(vector_array.tolist())
    
    async def generate_embeddings(self, texts: List[str]) -> List[EmbeddingVector]:
        """Generate embeddings for multiple texts."""
        vector_arrays = await self._embedding_service.generate_embeddings(texts)
        return [EmbeddingVector.create(vector.tolist()) for vector in vector_arrays]
    
    async def generate_query_embedding(self, query_text: str) -> EmbeddingVector:
        """Generate embedding for a search query."""
        vector_array = await self._embedding_service.generate_query_embedding(query_text)
        return EmbeddingVector.create(vector_array.tolist())
    
    def get_dimensions(self) -> int:
        """Get the dimension size of embeddings produced by this service."""
        return self._embedding_service.get_dimensions()
    
    def get_model_name(self) -> str:
        """Get the name of the embedding model."""
        return self._embedding_service.get_model_name()
    
    async def is_available(self) -> bool:
        """Check if the embedding service is available."""
        return await self._embedding_service.is_available()
