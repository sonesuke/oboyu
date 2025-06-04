"""Chunk entity - represents a segment of a document for indexing."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..value_objects.chunk_id import ChunkId
from ..value_objects.language_code import LanguageCode


@dataclass
class Chunk:
    """Core chunk entity representing a document segment."""
    
    id: ChunkId
    document_path: Path
    title: str
    content: str
    chunk_index: int
    language: LanguageCode
    created_at: datetime
    modified_at: datetime
    metadata: Dict[str, object]
    start_char: int
    end_char: int
    prefix_content: Optional[str] = None
    prefix_type: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate chunk consistency."""
        if not self.title.strip():
            raise ValueError("Chunk title cannot be empty")
        
        if not self.content.strip():
            raise ValueError("Chunk content cannot be empty")
        
        if self.chunk_index < 0:
            raise ValueError("Chunk index cannot be negative")
        
        if self.start_char >= self.end_char:
            raise ValueError("Start character must be less than end character")
        
        if self.start_char < 0:
            raise ValueError("Start character cannot be negative")
        
        if self.created_at > self.modified_at:
            raise ValueError("Created date cannot be after modified date")
    
    def get_size(self) -> int:
        """Get chunk size in characters."""
        return len(self.content)
    
    def get_word_count(self) -> int:
        """Get approximate word count."""
        return len(self.content.split())
    
    def is_meaningful(self) -> bool:
        """Check if chunk contains meaningful content."""
        return (
            len(self.content.strip()) > 10
            and self.get_word_count() > 3
            and not self.content.isspace()
        )
    
    def get_snippet(self, max_length: int = 200) -> str:
        """Get a snippet of the content with specified maximum length."""
        if len(self.content) <= max_length:
            return self.content
        
        snippet = self.content[:max_length]
        
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
    
    def should_be_indexed(self) -> bool:
        """Determine if chunk should be included in search index."""
        return (
            self.is_meaningful()
            and self.language != LanguageCode.UNKNOWN
            and len(self.content.strip()) >= 5
        )
    
    def create_with_prefix(self, prefix_content: str, prefix_type: str) -> "Chunk":
        """Create a copy of the chunk with prefix content."""
        return Chunk(
            id=self.id,
            document_path=self.document_path,
            title=self.title,
            content=self.content,
            chunk_index=self.chunk_index,
            language=self.language,
            created_at=self.created_at,
            modified_at=self.modified_at,
            metadata=self.metadata.copy(),
            start_char=self.start_char,
            end_char=self.end_char,
            prefix_content=prefix_content,
            prefix_type=prefix_type
        )
