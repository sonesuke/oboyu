"""Immutable search context for query operations."""

from dataclasses import dataclass, replace
from typing import Optional

from ..common.types.search_mode import SearchMode


@dataclass(frozen=True)
class SearchContext:
    """Immutable search context that tracks user intent.
    
    This class represents what the user explicitly requested,
    with None values indicating no explicit preference.
    """

    query: str
    mode: SearchMode
    top_k: Optional[int] = None
    use_reranker: Optional[bool] = None  # None = not explicitly set
    reranker_model: Optional[str] = None
    reranker_top_k: Optional[int] = None
    
    def with_explicit_reranker(self, enabled: bool) -> 'SearchContext':
        """Create a new context with reranker explicitly set."""
        if self.use_reranker is not None:
            raise ValueError(
                f"Reranker already explicitly set to {self.use_reranker}. "
                "Cannot override explicit user preference."
            )
        return replace(self, use_reranker=enabled)
    
    def with_explicit_top_k(self, top_k: int) -> 'SearchContext':
        """Create a new context with top_k explicitly set."""
        if self.top_k is not None:
            raise ValueError(
                f"Top-k already explicitly set to {self.top_k}. "
                "Cannot override explicit user preference."
            )
        return replace(self, top_k=top_k)
    
    def with_explicit_reranker_model(self, model: str) -> 'SearchContext':
        """Create a new context with reranker model explicitly set."""
        if self.reranker_model is not None:
            raise ValueError(
                f"Reranker model already explicitly set to {self.reranker_model}. "
                "Cannot override explicit user preference."
            )
        return replace(self, reranker_model=model)
    
    def has_explicit_reranker_preference(self) -> bool:
        """Check if the user explicitly set reranker preference."""
        return self.use_reranker is not None
    
    def merge_with(self, other: 'SearchContext') -> 'SearchContext':
        """Merge with another context, preferring this context's explicit values.
        
        This is useful for combining CLI args with file config.
        """
        return SearchContext(
            query=self.query,  # Always use this query
            mode=self.mode,    # Always use this mode
            top_k=self.top_k if self.top_k is not None else other.top_k,
            use_reranker=self.use_reranker if self.use_reranker is not None else other.use_reranker,
            reranker_model=self.reranker_model if self.reranker_model is not None else other.reranker_model,
            reranker_top_k=self.reranker_top_k if self.reranker_top_k is not None else other.reranker_top_k,
        )

