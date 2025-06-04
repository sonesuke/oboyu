"""Reranker service port interface."""

from abc import ABC, abstractmethod
from typing import List

from ...domain.entities.query import Query
from ...domain.entities.search_result import SearchResult


class RerankerService(ABC):
    """Abstract interface for result reranking services."""
    
    @abstractmethod
    async def rerank(self, query: Query, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results based on query relevance."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the reranking model."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the reranking service is available."""
        pass
    
    @abstractmethod
    def supports_batch_reranking(self) -> bool:
        """Check if service supports batch reranking."""
        pass
