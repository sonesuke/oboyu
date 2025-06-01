"""Core snippet extraction logic with boundary detection."""

from typing import List, Tuple


class SnippetExtractor:
    """Handles core snippet extraction logic and boundary detection."""

    def extract_snippet(
        self,
        content: str,
        match_positions: List[Tuple[int, int]],
        length: int,
        context_window: int = 50
    ) -> str:
        """Extract snippet from content around match positions.
        
        Args:
            content: Full text content
            match_positions: List of (start, end) positions of matches
            length: Maximum snippet length in characters
            context_window: Characters before/after match for context
            
        Returns:
            Extracted snippet text

        """
        if not content or not content.strip():
            return ""
            
        if not match_positions:
            # No matches, return beginning of content
            return self._truncate_to_length(content, length)
        
        # Find the best position to center the snippet
        center_pos = self._find_optimal_center(match_positions)
        
        # Calculate initial boundaries
        start = max(0, center_pos - context_window)
        end = min(len(content), center_pos + context_window)
        
        # Expand to desired length while ensuring we include all matches
        start, end = self._expand_to_target_length(
            content, start, end, length, match_positions
        )
        
        return content[start:end]
    
    def _find_optimal_center(self, match_positions: List[Tuple[int, int]]) -> int:
        """Find optimal center position for snippet extraction.
        
        Args:
            match_positions: List of (start, end) positions of matches
            
        Returns:
            Optimal center position

        """
        if not match_positions:
            return 0
        
        # Use the center of the first match as the starting point
        first_match = match_positions[0]
        return (first_match[0] + first_match[1]) // 2
    
    def _expand_to_target_length(
        self,
        content: str,
        start: int,
        end: int,
        target_length: int,
        match_positions: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Expand snippet boundaries to target length while including matches.
        
        Args:
            content: Full text content
            start: Current start position
            end: Current end position
            target_length: Target snippet length
            match_positions: Match positions to ensure inclusion
            
        Returns:
            Tuple of (new_start, new_end)

        """
        current_length = end - start
        
        if current_length >= target_length:
            return start, end
        
        # Calculate expansion needed
        expansion = (target_length - current_length) // 2
        new_start = max(0, start - expansion)
        new_end = min(len(content), end + expansion)
        
        # Ensure all matches are included
        for match_start, match_end in match_positions:
            new_start = min(new_start, match_start)
            new_end = max(new_end, match_end)
        
        # Final adjustment to respect content boundaries
        if new_end - new_start > target_length:
            # If still too long, truncate from the end
            new_end = new_start + target_length
        
        return new_start, min(len(content), new_end)
    
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
