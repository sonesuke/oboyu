"""Search-related configuration."""

from dataclasses import dataclass


@dataclass
class SearchConfig:
    """Search-related configuration."""
    
    # Hybrid search weights
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    
    # Search parameters
    use_reranker: bool = False
    top_k_multiplier: int = 2
    
    # BM25 parameters
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_min_token_length: int = 2
    
    def __post_init__(self) -> None:
        """Post-initialization validation."""
        # Normalize weights
        total_weight = self.vector_weight + self.bm25_weight
        if total_weight > 0:
            self.vector_weight = self.vector_weight / total_weight
            self.bm25_weight = self.bm25_weight / total_weight
        else:
            self.vector_weight = 0.7
            self.bm25_weight = 0.3
