"""Shared types and configurations for snippet processing."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SnippetStrategy(Enum):
    """Strategy for snippet boundary detection."""
    
    FIXED_LENGTH = "fixed_length"
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"


class SnippetLevel(BaseModel):
    """Configuration for a snippet detail level."""
    
    type: str = Field(description="Type of snippet level (summary, detailed, full_context)")
    length: int = Field(gt=0, description="Maximum length in characters")


class SnippetConfig(BaseModel):
    """Configuration for snippet generation."""
    
    length: int = Field(default=300, gt=0, description="Maximum snippet length in characters")
    context_window: int = Field(default=50, ge=0, description="Characters before/after match for context")
    max_snippets_per_result: int = Field(default=1, gt=0, description="Maximum number of snippets per search result")
    highlight_matches: bool = Field(default=True, description="Whether to highlight search matches")
    strategy: SnippetStrategy = Field(default=SnippetStrategy.SENTENCE_BOUNDARY, description="Strategy for snippet boundary detection")
    prefer_complete_sentences: bool = Field(default=True, description="Try to end snippets at sentence boundaries")
    include_surrounding_context: bool = Field(default=True, description="Include context around matches")
    japanese_aware: bool = Field(default=True, description="Consider Japanese sentence boundaries")
    levels: Optional[List[SnippetLevel]] = Field(default=None, description="Multi-level snippet configurations")


@dataclass
class SnippetMatch:
    """Information about a text match for snippet generation."""
    
    start: int
    end: int
    text: str
    score: float = 0.0
