"""Snippet extraction strategies using Strategy pattern."""

from abc import ABC, abstractmethod
from typing import List

from .context_provider import ContextProvider
from .japanese_snippet_processor import JapaneseSnippetProcessor
from .snippet_types import SnippetConfig, SnippetMatch


class SnippetStrategy(ABC):
    """Base class for different snippet extraction strategies."""

    def __init__(
        self,
        context_provider: ContextProvider,
        japanese_processor: JapaneseSnippetProcessor
    ) -> None:
        """Initialize strategy with required components.
        
        Args:
            context_provider: Context provider for generating context around matches
            japanese_processor: Japanese text processor for language-specific handling

        """
        self.context_provider = context_provider
        self.japanese_processor = japanese_processor

    @abstractmethod
    def process(
        self,
        content: str,
        matches: List[SnippetMatch],
        config: SnippetConfig
    ) -> str:
        """Process content to extract snippet using this strategy.
        
        Args:
            content: Full text content
            matches: List of matches found in content
            config: Snippet processing configuration
            
        Returns:
            Extracted snippet text

        """
        pass

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
        
        return text[:max_length].strip()


class FixedLengthStrategy(SnippetStrategy):
    """Fixed character length snippet strategy."""

    def process(
        self,
        content: str,
        matches: List[SnippetMatch],
        config: SnippetConfig
    ) -> str:
        """Extract snippet with fixed character length.
        
        Args:
            content: Full text content
            matches: List of matches found in content
            config: Snippet processing configuration
            
        Returns:
            Fixed-length snippet

        """
        if not content:
            return ""
        
        if not matches:
            # No matches, return beginning of content
            return self._truncate_to_length(content, config.length)
        
        # Find the best match to center around
        best_match = max(matches, key=lambda m: m.score)
        match_center = (best_match.start + best_match.end) // 2
        
        # Calculate boundaries
        half_length = config.length // 2
        start = max(0, match_center - half_length)
        end = min(len(content), start + config.length)
        
        # Adjust start if we hit the end boundary
        if end == len(content):
            start = max(0, end - config.length)
        
        return content[start:end].strip()


class SentenceBoundaryStrategy(SnippetStrategy):
    """Sentence-aware snippet extraction."""

    def process(
        self,
        content: str,
        matches: List[SnippetMatch],
        config: SnippetConfig
    ) -> str:
        """Extract snippet respecting sentence boundaries.
        
        Args:
            content: Full text content
            matches: List of matches found in content
            config: Snippet processing configuration
            
        Returns:
            Sentence-boundary aware snippet

        """
        if not content:
            return ""
        
        # Start with fixed-length extraction
        fixed_strategy = FixedLengthStrategy(self.context_provider, self.japanese_processor)
        initial_snippet = fixed_strategy.process(content, matches, config)
        
        if not config.prefer_complete_sentences:
            return initial_snippet
        
        # Adjust to sentence boundaries
        if config.japanese_aware and self.japanese_processor.is_japanese_text(initial_snippet):
            adjusted_snippet = self.japanese_processor.adjust_to_sentence_boundaries(
                initial_snippet,
                prefer_complete=True
            )
        else:
            adjusted_snippet = self._adjust_to_english_sentence_boundaries(initial_snippet)
        
        # If adjustment made the snippet too short, fall back to fixed length
        if len(adjusted_snippet) < config.length * 0.5:  # Less than 50% of target length
            return initial_snippet
        
        return adjusted_snippet

    def _adjust_to_english_sentence_boundaries(self, text: str) -> str:
        """Adjust text to English sentence boundaries.
        
        Args:
            text: Text to adjust
            
        Returns:
            Text adjusted to sentence boundaries

        """
        import re
        
        # Find sentence endings
        sentence_pattern = re.compile(r'[.!?]\s+')
        matches = list(sentence_pattern.finditer(text))
        
        if matches:
            # Use the last sentence ending that fits
            last_match = matches[-1]
            return text[:last_match.end()].strip()
        
        # No sentence ending found, avoid word breaks
        return self._avoid_word_breaks(text)

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


class ParagraphBoundaryStrategy(SnippetStrategy):
    """Paragraph-aware snippet extraction."""

    def process(
        self,
        content: str,
        matches: List[SnippetMatch],
        config: SnippetConfig
    ) -> str:
        """Extract snippet respecting paragraph boundaries.
        
        Args:
            content: Full text content
            matches: List of matches found in content
            config: Snippet processing configuration
            
        Returns:
            Paragraph-boundary aware snippet

        """
        if not content:
            return ""
        
        # Start with sentence boundary extraction
        sentence_strategy = SentenceBoundaryStrategy(
            self.context_provider,
            self.japanese_processor
        )
        initial_snippet = sentence_strategy.process(content, matches, config)
        
        # Adjust to paragraph boundaries
        if config.japanese_aware and self.japanese_processor.is_japanese_text(initial_snippet):
            adjusted_snippet = self.japanese_processor.adjust_to_paragraph_boundaries(
                initial_snippet,
                prefer_complete=True
            )
        else:
            adjusted_snippet = self._adjust_to_english_paragraph_boundaries(initial_snippet)
        
        # If adjustment made the snippet too short, fall back to sentence strategy
        if len(adjusted_snippet) < config.length * 0.3:  # Less than 30% of target length
            return initial_snippet
        
        return adjusted_snippet

    def _adjust_to_english_paragraph_boundaries(self, text: str) -> str:
        """Adjust text to English paragraph boundaries.
        
        Args:
            text: Text to adjust
            
        Returns:
            Text adjusted to paragraph boundaries

        """
        import re
        
        # Find paragraph breaks
        paragraph_pattern = re.compile(r'\n\s*\n')
        matches = list(paragraph_pattern.finditer(text))
        
        if matches:
            last_match = matches[-1]
            return text[:last_match.start()].strip()
        
        # No paragraph break found, fallback to sentence boundaries
        sentence_strategy = SentenceBoundaryStrategy(
            self.context_provider,
            self.japanese_processor
        )
        return sentence_strategy._adjust_to_english_sentence_boundaries(text)
