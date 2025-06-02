"""Snippet processing for search results with Japanese text support."""

from typing import Dict, List

from .context_provider import ContextProvider
from .japanese_snippet_processor import JapaneseSnippetProcessor
from .snippet_strategies import (
    FixedLengthStrategy,
    ParagraphBoundaryStrategy,
    SentenceBoundaryStrategy,
    SnippetStrategy as SnippetStrategyBase,
)
from .snippet_types import SnippetConfig, SnippetMatch, SnippetStrategy
from .text_highlighter import TextHighlighter


class SnippetProcessor:
    """Orchestrates snippet processing using specialized components."""
    
    def __init__(self, config: SnippetConfig) -> None:
        """Initialize snippet processor with configuration.
        
        Args:
            config: Snippet processing configuration

        """
        self.config = config
        
        # Initialize components
        self.context_provider = ContextProvider()
        self.japanese_processor = JapaneseSnippetProcessor()
        self.text_highlighter = TextHighlighter()
        
        # Initialize strategies
        self.strategies: Dict[SnippetStrategy, SnippetStrategyBase] = {
            SnippetStrategy.FIXED_LENGTH: FixedLengthStrategy(
                self.context_provider, self.japanese_processor
            ),
            SnippetStrategy.SENTENCE_BOUNDARY: SentenceBoundaryStrategy(
                self.context_provider, self.japanese_processor
            ),
            SnippetStrategy.PARAGRAPH_BOUNDARY: ParagraphBoundaryStrategy(
                self.context_provider, self.japanese_processor
            ),
        }
    
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
        
        # Get the appropriate strategy
        strategy = self.strategies[self.config.strategy]
        
        # Generate snippet using the strategy
        snippet = strategy.process(content, matches, self.config)
        
        # Apply highlighting if configured
        if self.config.highlight_matches and query:
            snippet = self.text_highlighter.highlight_query(snippet, query)
        
        return snippet
    
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

