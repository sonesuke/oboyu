"""Query entity - represents a search query."""

from dataclasses import dataclass
from typing import List, Optional

from ..value_objects.language_code import LanguageCode
from ..value_objects.search_mode import SearchMode


@dataclass
class Query:
    """Core query entity for search operations."""
    
    text: str
    mode: SearchMode
    top_k: int
    language: Optional[LanguageCode] = None
    similarity_threshold: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate query consistency."""
        if not self.text.strip():
            raise ValueError("Query text cannot be empty")
        
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        
        if self.top_k > 1000:
            raise ValueError("top_k cannot exceed 1000")
        
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
    
    def get_normalized_text(self) -> str:
        """Get normalized query text."""
        return ' '.join(self.text.split())
    
    def get_terms(self) -> List[str]:
        """Get query terms as a list."""
        return [term.strip().lower() for term in self.text.split() if term.strip()]
    
    def is_simple_query(self) -> bool:
        """Check if this is a simple single-term query."""
        terms = self.get_terms()
        return len(terms) == 1
    
    def is_phrase_query(self) -> bool:
        """Check if this appears to be a phrase query."""
        return '"' in self.text or len(self.get_terms()) > 3
    
    def should_use_vector_search(self) -> bool:
        """Determine if vector search would be beneficial."""
        return (
            self.mode in (SearchMode.VECTOR, SearchMode.HYBRID)
            and len(self.get_normalized_text()) > 10
        )
    
    def should_use_bm25_search(self) -> bool:
        """Determine if BM25 search would be beneficial."""
        return (
            self.mode in (SearchMode.BM25, SearchMode.HYBRID)
            and len(self.get_terms()) >= 1
        )
    
    def create_variant_with_mode(self, mode: SearchMode) -> "Query":
        """Create a variant of this query with a different search mode."""
        return Query(
            text=self.text,
            mode=mode,
            top_k=self.top_k,
            language=self.language,
            similarity_threshold=self.similarity_threshold
        )
