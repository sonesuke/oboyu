"""Consolidated query configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryConfig:
    """Query engine configuration."""

    # Basic search settings
    top_k: int = 10
    show_scores: bool = False
    interactive: bool = False
    
    # Reranking settings
    rerank: bool = True
    rerank_model: str = "cl-nagoya/ruri-reranker-small"
    
    # Search mode settings
    default_search_mode: str = "hybrid"
    
    # Hybrid search weights
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    
    # Performance settings
    timeout: int = 30
    max_retries: int = 3

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        # Normalize search weights
        total_weight = self.vector_weight + self.bm25_weight
        if total_weight > 0:
            self.vector_weight = self.vector_weight / total_weight
            self.bm25_weight = self.bm25_weight / total_weight
        else:
            self.vector_weight = 0.7
            self.bm25_weight = 0.3
            
        # Validate top_k
        if self.top_k <= 0:
            self.top_k = 10


# Default configuration
DEFAULT_CONFIG = {
    "query": QueryConfig()
}