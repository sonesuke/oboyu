"""Embedding service port interface."""

from abc import ABC, abstractmethod
from typing import List

from ...domain.value_objects.embedding_vector import EmbeddingVector


class EmbeddingService(ABC):
    """Abstract interface for embedding generation services."""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> EmbeddingVector:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[EmbeddingVector]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    async def generate_query_embedding(self, query_text: str) -> EmbeddingVector:
        """Generate embedding for a search query."""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get the dimension size of embeddings produced by this service."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the embedding model."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the embedding service is available."""
        pass
