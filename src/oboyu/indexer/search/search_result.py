"""Unified search result format for all search types."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, field_validator


class SearchResultType(str, Enum):
    """Type of search result."""
    
    DOCUMENT_CHUNK = "document_chunk"
    FULL_DOCUMENT = "full_document"
    SNIPPET = "snippet"


class SearchResult(BaseModel):
    """Enhanced result of a search operation with comprehensive validation."""

    chunk_id: str = Field(description="ID of the matching chunk", pattern=r'^[a-zA-Z0-9\-_/:.]+$')
    path: str = Field(description="Path to the source document")
    title: str = Field(description="Title of the document or chunk", min_length=1, max_length=500)
    content: str = Field(description="Chunk text content", min_length=1)
    chunk_index: int = Field(ge=0, description="Position of this chunk in the original document")
    language: str = Field(description="Language code of the content", pattern=r'^[a-z]{2}$')
    metadata: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict,
        description="Additional metadata about the chunk"
    )
    score: float = Field(ge=0.0, le=1.0, description="Similarity score (0-1, where 1 is perfect match)")
    result_type: SearchResultType = Field(default=SearchResultType.DOCUMENT_CHUNK, description="Type of search result")
    highlighted_content: Optional[str] = Field(default=None, description="Content with highlighted matches")
    context_before: Optional[str] = Field(default=None, description="Context before the match")
    context_after: Optional[str] = Field(default=None, description="Context after the match")
    search_terms: Optional[list[str]] = Field(default=None, description="Terms that matched in this result")
    created_at: Optional[datetime] = Field(default=None, description="When the chunk was created")
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path format."""
        # Basic path validation
        if not v or v.isspace():
            raise ValueError('Path cannot be empty')
        # Convert to Path to validate format
        try:
            Path(v)
        except ValueError as e:
            raise ValueError(f'Invalid path format: {e}')
        return v
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty."""
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate and clean title."""
        cleaned = ' '.join(v.split())
        if not cleaned:
            raise ValueError('Title cannot be empty')
        return cleaned
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validate and round score."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Score must be between 0.0 and 1.0')
        return round(v, 6)  # Round to 6 decimal places for consistency
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language code."""
        v = v.lower()
        # Basic validation for common language codes
        valid_codes = {'ja', 'en', 'zh', 'ko', 'fr', 'de', 'es', 'it', 'pt', 'ru', 'ar', 'hi', 'unknown'}
        if len(v) == 2 and v.isalpha():
            return v
        if v in valid_codes:
            return v
        raise ValueError(f'Invalid language code: {v}')
    
    @field_validator('search_terms')
    @classmethod
    def validate_search_terms(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate search terms."""
        if v is not None:
            # Remove empty or whitespace-only terms
            cleaned = [term.strip() for term in v if term.strip()]
            return cleaned if cleaned else None
        return v
    
    def get_snippet(self, max_length: int = 200) -> str:
        """Get a snippet of the content with specified maximum length."""
        if self.highlighted_content and len(self.highlighted_content) <= max_length:
            return self.highlighted_content
        
        content = self.content
        if len(content) <= max_length:
            return content
        
        # Try to find a good breaking point
        snippet = content[:max_length]
        
        # Look for sentence boundary
        last_period = snippet.rfind('.')
        last_exclamation = snippet.rfind('!')
        last_question = snippet.rfind('?')
        
        sentence_end = max(last_period, last_exclamation, last_question)
        
        # If we found a sentence boundary within reasonable distance
        if sentence_end > max_length * 0.7:
            return snippet[:sentence_end + 1]
        
        # Look for word boundary
        last_space = snippet.rfind(' ')
        if last_space > max_length * 0.8:
            return snippet[:last_space] + "..."
        
        return snippet + "..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "title": self.title,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "language": self.language,
            "metadata": self.metadata,
            "score": self.score,
            "result_type": self.result_type.value,
            "highlighted_content": self.highlighted_content,
            "search_terms": self.search_terms
        }
