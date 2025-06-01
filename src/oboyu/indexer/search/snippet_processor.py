"""Snippet processing for search results with Japanese text support."""

import re
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


class SnippetProcessor:
    """Processes text content to generate smart snippets with Japanese support."""
    
    # Japanese sentence boundary patterns
    JAPANESE_SENTENCE_ENDINGS = re.compile(r'[。！？]')
    JAPANESE_PARAGRAPH_BREAKS = re.compile(r'\n\s*\n')
    
    # General sentence boundary patterns
    SENTENCE_ENDINGS = re.compile(r'[.!?]\s+')
    PARAGRAPH_BREAKS = re.compile(r'\n\s*\n')
    
    def __init__(self, config: SnippetConfig) -> None:
        """Initialize snippet processor with configuration.
        
        Args:
            config: Snippet processing configuration

        """
        self.config = config
    
    def generate_snippet(self, content: str, query: str, score: float = 0.0) -> str:
        """Generate a snippet from content based on query and configuration.
        
        Args:
            content: Full text content
            query: Search query used to find matches
            score: Relevance score of the match
            
        Returns:
            Generated snippet text

        """
        if not content or not content.strip():
            return ""
        
        # Use multi-level snippets if configured
        if self.config.levels:
            return self._generate_multi_level_snippet(content, query, score)
        
        # Find query matches in content
        matches = self._find_query_matches(content, query)
        
        if not matches and self.config.strategy == SnippetStrategy.FIXED_LENGTH:
            # No matches found, return beginning of content
            return self._truncate_to_length(content, self.config.length)
        
        if not matches:
            # No matches but using sentence/paragraph strategy
            return self._extract_by_strategy(content, 0, self.config.length)
        
        # Generate snippet around best match
        best_match = max(matches, key=lambda m: m.score) if matches else None
        if best_match:
            snippet = self._extract_snippet_around_match(content, best_match)
            if self.config.highlight_matches:
                snippet = self._highlight_matches(snippet, query)
            return snippet
        
        return self._truncate_to_length(content, self.config.length)
    
    def _generate_multi_level_snippet(self, content: str, query: str, score: float) -> str:
        """Generate multi-level snippet based on configuration.
        
        Args:
            content: Full text content
            query: Search query
            score: Relevance score
            
        Returns:
            Multi-level snippet text

        """
        if not self.config.levels:
            return self.generate_snippet(content, query, score)
        
        # For now, use the first level configuration
        # In a more advanced implementation, we could select level based on score
        first_level = self.config.levels[0]
        
        # Create temporary config for this level
        temp_config = SnippetConfig(
            length=first_level.length,
            context_window=self.config.context_window,
            max_snippets_per_result=self.config.max_snippets_per_result,
            highlight_matches=self.config.highlight_matches,
            strategy=self.config.strategy,
            prefer_complete_sentences=self.config.prefer_complete_sentences,
            include_surrounding_context=self.config.include_surrounding_context,
            japanese_aware=self.config.japanese_aware
        )
        
        processor = SnippetProcessor(temp_config)
        return processor.generate_snippet(content, query, score)
    
    def _find_query_matches(self, content: str, query: str) -> List[SnippetMatch]:
        """Find all occurrences of query terms in content.
        
        Args:
            content: Text content to search
            query: Search query
            
        Returns:
            List of matches found in content

        """
        matches = []
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Simple exact phrase matching
        start = 0
        while True:
            pos = content_lower.find(query_lower, start)
            if pos == -1:
                break
            
            match = SnippetMatch(
                start=pos,
                end=pos + len(query),
                text=content[pos:pos + len(query)],
                score=1.0  # Simple scoring for exact matches
            )
            matches.append(match)
            start = pos + 1
        
        # Also find individual word matches
        query_words = query.split()
        for word in query_words:
            if len(word) < 2:  # Skip very short words
                continue
            
            word_lower = word.lower()
            start = 0
            while True:
                pos = content_lower.find(word_lower, start)
                if pos == -1:
                    break
                
                # Check if it's a word boundary
                if (pos == 0 or not content[pos-1].isalnum()) and \
                   (pos + len(word) >= len(content) or not content[pos + len(word)].isalnum()):
                    
                    match = SnippetMatch(
                        start=pos,
                        end=pos + len(word),
                        text=content[pos:pos + len(word)],
                        score=0.5  # Lower score for individual words
                    )
                    matches.append(match)
                
                start = pos + 1
        
        return matches
    
    def _extract_snippet_around_match(self, content: str, match: SnippetMatch) -> str:
        """Extract snippet around a specific match.
        
        Args:
            content: Full text content
            match: Match to center snippet around
            
        Returns:
            Extracted snippet text

        """
        # Calculate context window around match
        match_center = (match.start + match.end) // 2
        context_start = max(0, match_center - self.config.context_window)
        context_end = min(len(content), match_center + self.config.context_window)
        
        # Expand to desired length, ensuring we include the match
        target_length = self.config.length
        current_length = context_end - context_start
        
        if current_length < target_length:
            expansion = (target_length - current_length) // 2
            new_start = max(0, context_start - expansion)
            new_end = min(len(content), context_end + expansion)
            
            # Make sure we still include the match
            context_start = min(context_start, new_start)
            context_end = max(context_end, new_end)
        
        # Ensure the match is definitely included
        context_start = min(context_start, match.start)
        context_end = max(context_end, match.end)
        
        # Apply strategy-specific boundary detection
        return self._extract_by_strategy(content, context_start, context_end - context_start)
    
    def _extract_by_strategy(self, content: str, start: int, max_length: int) -> str:
        """Extract text using the configured strategy.
        
        Args:
            content: Full text content
            start: Starting position
            max_length: Maximum length to extract
            
        Returns:
            Extracted text following strategy rules

        """
        end = min(len(content), start + max_length)
        text = content[start:end]
        
        if self.config.strategy == SnippetStrategy.FIXED_LENGTH:
            return self._truncate_to_length(text, self.config.length)
        
        # Apply sentence boundary strategy
        if self.config.strategy == SnippetStrategy.SENTENCE_BOUNDARY and self.config.prefer_complete_sentences:
            text = self._adjust_to_sentence_boundaries(text, content, start)
        
        # Apply paragraph boundary strategy
        elif self.config.strategy == SnippetStrategy.PARAGRAPH_BOUNDARY:
            text = self._adjust_to_paragraph_boundaries(text, content, start)
        
        # Always enforce maximum length as final step
        return self._truncate_to_length(text.strip(), self.config.length)
    
    def _adjust_to_sentence_boundaries(self, text: str, full_content: str, start_pos: int) -> str:
        """Adjust text boundaries to complete sentences.
        
        Args:
            text: Current text excerpt
            full_content: Full content for context
            start_pos: Starting position in full content
            
        Returns:
            Text adjusted to sentence boundaries

        """
        if self.config.japanese_aware:
            # Use Japanese sentence patterns
            sentence_pattern = self.JAPANESE_SENTENCE_ENDINGS
        else:
            # Use general sentence patterns
            sentence_pattern = self.SENTENCE_ENDINGS
        
        # Try to find a good ending point
        matches = list(sentence_pattern.finditer(text))
        if matches:
            # Use the last sentence ending that fits
            last_match = matches[-1]
            return text[:last_match.end()].strip()
        
        # If no sentence ending found, try to avoid cutting words
        return self._avoid_word_breaks(text)
    
    def _adjust_to_paragraph_boundaries(self, text: str, full_content: str, start_pos: int) -> str:
        """Adjust text boundaries to complete paragraphs.
        
        Args:
            text: Current text excerpt
            full_content: Full content for context
            start_pos: Starting position in full content
            
        Returns:
            Text adjusted to paragraph boundaries

        """
        if self.config.japanese_aware:
            paragraph_pattern = self.JAPANESE_PARAGRAPH_BREAKS
        else:
            paragraph_pattern = self.PARAGRAPH_BREAKS
        
        # Find paragraph breaks
        matches = list(paragraph_pattern.finditer(text))
        if matches:
            last_match = matches[-1]
            return text[:last_match.start()].strip()
        
        # Fallback to sentence boundaries
        return self._adjust_to_sentence_boundaries(text, full_content, start_pos)
    
    def _avoid_word_breaks(self, text: str) -> str:
        """Avoid breaking words at the end of text.
        
        Args:
            text: Text to adjust
            
        Returns:
            Text with word breaks avoided

        """
        if not text:
            return text
        
        # If last character is alphanumeric, try to find last word boundary
        if text[-1].isalnum():
            for i in range(len(text) - 1, -1, -1):
                if not text[i].isalnum():
                    # Return text up to end of the last complete word
                    return text[:i+1].strip()
            # If no word boundary found, return first word only
            for i in range(len(text)):
                if not text[i].isalnum():
                    return text[:i].strip()
        
        return text.strip()
    
    def _highlight_matches(self, text: str, query: str) -> str:
        """Add highlighting to query matches in text.
        
        Args:
            text: Text to highlight
            query: Query to highlight
            
        Returns:
            Text with highlighted matches

        """
        if not self.config.highlight_matches:
            return text
        
        # Simple highlighting with **bold** markers
        # In a real implementation, you might use HTML or other markup
        query_words = query.split()
        highlighted_text = text
        
        for word in query_words:
            if len(word) < 2:
                continue
            
            # Case-insensitive replacement with word boundaries
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            highlighted_text = pattern.sub(lambda m: f'**{m.group()}**', highlighted_text)
        
        return highlighted_text
    
    def _truncate_to_length(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length.
        
        Args:
            text: Text to truncate
            max_length: Maximum length in characters
            
        Returns:
            Truncated text

        """
        if len(text) <= max_length:
            return text
        
        truncated = text[:max_length]
        
        # Avoid word breaks if possible
        if self.config.prefer_complete_sentences:
            truncated = self._avoid_word_breaks(truncated)
        
        return truncated.strip()

