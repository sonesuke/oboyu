"""Search-related configuration."""

from dataclasses import dataclass


@dataclass
class SearchConfig:
    """Search-related configuration."""

    # RRF (Reciprocal Rank Fusion) parameter
    rrf_k: int = 60

    # Search parameters
    use_reranker: bool = False
    top_k_multiplier: int = 2

    # Hybrid search weights
    vector_weight: float = 0.7
    bm25_weight: float = 0.3

    # BM25 parameters
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_min_token_length: int = 2

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        # Validate RRF k parameter
        if self.rrf_k <= 0:
            self.rrf_k = 60
