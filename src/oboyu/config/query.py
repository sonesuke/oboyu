"""Consolidated query configuration."""

from dataclasses import dataclass


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
    
    # RRF (Reciprocal Rank Fusion) parameter
    rrf_k: int = 60
    
    # Performance settings
    timeout: int = 30
    max_retries: int = 3

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        # Validate top_k
        if self.top_k <= 0:
            self.top_k = 10
            
        # Validate RRF k parameter
        if self.rrf_k <= 0:
            self.rrf_k = 60


# Default configuration
DEFAULT_CONFIG = {
    "query": QueryConfig()
}
