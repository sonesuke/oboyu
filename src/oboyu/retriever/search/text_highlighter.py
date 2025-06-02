"""Text highlighting and formatting functionality."""

import re
from typing import List


class TextHighlighter:
    """Manages text highlighting and formatting."""

    def __init__(self, highlight_format: str = "**{}**") -> None:
        """Initialize text highlighter.
        
        Args:
            highlight_format: Format string for highlighting matches

        """
        self.highlight_format = highlight_format

    def highlight_matches(
        self,
        text: str,
        matches: List[str],
        case_sensitive: bool = False
    ) -> str:
        """Add highlighting to query matches in text.
        
        Args:
            text: Text to highlight
            matches: List of terms to highlight
            case_sensitive: Whether matching should be case sensitive
            
        Returns:
            Text with highlighted matches

        """
        if not matches:
            return text
        
        highlighted_text = text
        
        for match in matches:
            if len(match) < 2:  # Skip very short matches
                continue
            
            highlighted_text = self._highlight_single_match(
                highlighted_text, match, case_sensitive
            )
        
        return highlighted_text

    def highlight_query(self, text: str, query: str, case_sensitive: bool = False) -> str:
        """Highlight all terms from a query in text.
        
        Args:
            text: Text to highlight
            query: Query string containing terms to highlight
            case_sensitive: Whether matching should be case sensitive
            
        Returns:
            Text with highlighted query terms

        """
        if not query:
            return text
        
        # Split query into individual words
        query_words = query.split()
        return self.highlight_matches(text, query_words, case_sensitive)

    def _highlight_single_match(
        self,
        text: str,
        match: str,
        case_sensitive: bool = False
    ) -> str:
        """Highlight a single match in text.
        
        Args:
            text: Text to highlight
            match: Term to highlight
            case_sensitive: Whether matching should be case sensitive
            
        Returns:
            Text with highlighted match

        """
        flags = 0 if case_sensitive else re.IGNORECASE
        
        # Use word boundaries to avoid partial matches
        pattern = re.compile(r'\b' + re.escape(match) + r'\b', flags)
        
        return pattern.sub(
            lambda m: self.highlight_format.format(m.group()),
            text
        )

    def remove_highlights(self, text: str) -> str:
        """Remove highlighting from text.
        
        Args:
            text: Text with highlights
            
        Returns:
            Text without highlights

        """
        # Remove **bold** markers (default format)
        return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    def set_highlight_format(self, format_string: str) -> None:
        """Set the highlight format string.
        
        Args:
            format_string: Format string with {} placeholder for match

        """
        self.highlight_format = format_string
