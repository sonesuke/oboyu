"""Search result entity - represents the result of a search operation."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..value_objects.chunk_id import ChunkId
from ..value_objects.language_code import LanguageCode
from ..value_objects.score import Score


@dataclass
class SearchResult:
    """Core search result entity."""
    
    chunk_id: ChunkId
    document_path: Path
    title: str
    content: str
    chunk_index: int
    language: LanguageCode
    score: Score
    metadata: Dict[str, object]
    highlighted_content: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    search_terms: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate search result consistency."""
        if not self.title.strip():
            raise ValueError("Search result title cannot be empty")
        
        if not self.content.strip():
            raise ValueError("Search result content cannot be empty")
        
        if self.chunk_index < 0:
            raise ValueError("Chunk index cannot be negative")
    
    def get_snippet(self, max_length: int = 200) -> str:
        """Get a snippet of the content with specified maximum length."""
        if self.highlighted_content and len(self.highlighted_content) <= max_length:
            return self.highlighted_content
        
        content = self.content
        if len(content) <= max_length:
            return content
        
        snippet = content[:max_length]
        
        last_period = snippet.rfind('.')
        last_exclamation = snippet.rfind('!')
        last_question = snippet.rfind('?')
        
        sentence_end = max(last_period, last_exclamation, last_question)
        
        if sentence_end > max_length * 0.7:
            return snippet[:sentence_end + 1]
        
        last_space = snippet.rfind(' ')
        if last_space > max_length * 0.8:
            return snippet[:last_space] + "..."
        
        return snippet + "..."
    
    def has_highlights(self) -> bool:
        """Check if result has highlighted content."""
        return bool(self.highlighted_content is not None and self.highlighted_content.strip())
    
    def has_context(self) -> bool:
        """Check if result has context information."""
        return bool(
            (self.context_before is not None and self.context_before.strip())
            or (self.context_after is not None and self.context_after.strip())
        )
    
    def is_high_quality(self) -> bool:
        """Determine if this is a high-quality search result."""
        return (
            self.score.value >= 0.5
            and len(self.content.strip()) >= 20
            and self.language != LanguageCode.UNKNOWN
        )
    
    def create_with_highlights(self, highlighted_content: str,
                             search_terms: List[str]) -> "SearchResult":
        """Create a copy with highlighted content."""
        return SearchResult(
            chunk_id=self.chunk_id,
            document_path=self.document_path,
            title=self.title,
            content=self.content,
            chunk_index=self.chunk_index,
            language=self.language,
            score=self.score,
            metadata=self.metadata.copy(),
            highlighted_content=highlighted_content,
            context_before=self.context_before,
            context_after=self.context_after,
            search_terms=search_terms.copy() if search_terms else None,
            created_at=self.created_at
        )
    
    def create_with_context(self, context_before: Optional[str],
                          context_after: Optional[str]) -> "SearchResult":
        """Create a copy with context information."""
        return SearchResult(
            chunk_id=self.chunk_id,
            document_path=self.document_path,
            title=self.title,
            content=self.content,
            chunk_index=self.chunk_index,
            language=self.language,
            score=self.score,
            metadata=self.metadata.copy(),
            highlighted_content=self.highlighted_content,
            context_before=context_before,
            context_after=context_after,
            search_terms=self.search_terms,
            created_at=self.created_at
        )
