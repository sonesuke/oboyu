"""Search mode value object."""

from enum import Enum


class SearchMode(Enum):
    """Available search modes as value object."""
    
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    
    @classmethod
    def from_string(cls, mode: str) -> "SearchMode":
        """Create search mode from string with validation."""
        normalized = mode.lower().strip()
        
        for search_mode in cls:
            if search_mode.value == normalized:
                return search_mode
        
        raise ValueError(f"Invalid search mode: {mode}")
    
    def supports_vector_search(self) -> bool:
        """Check if mode supports vector search."""
        return self.value in (self.VECTOR.value, self.HYBRID.value)
    
    def supports_bm25_search(self) -> bool:
        """Check if mode supports BM25 search."""
        return self.value in (self.BM25.value, self.HYBRID.value)
    
    def is_hybrid(self) -> bool:
        """Check if this is hybrid search mode."""
        return self.value == self.HYBRID.value
