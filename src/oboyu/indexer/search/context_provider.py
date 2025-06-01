"""Context generation around matches functionality."""

from typing import Tuple


class ContextProvider:
    """Generates context around matches."""

    def get_surrounding_context(
        self,
        content: str,
        match_position: int,
        context_window: int
    ) -> Tuple[str, str]:
        """Get context before and after a match position.
        
        Args:
            content: Full text content
            match_position: Position of the match
            context_window: Number of characters for context window
            
        Returns:
            Tuple of (before_context, after_context)

        """
        if not content:
            return "", ""
        
        start = max(0, match_position - context_window)
        end = min(len(content), match_position + context_window)
        
        before_context = content[start:match_position]
        after_context = content[match_position:end]
        
        return before_context, after_context

    def get_context_around_range(
        self,
        content: str,
        start_pos: int,
        end_pos: int,
        context_window: int
    ) -> Tuple[str, str]:
        """Get context before and after a text range.
        
        Args:
            content: Full text content
            start_pos: Start position of the range
            end_pos: End position of the range
            context_window: Number of characters for context window
            
        Returns:
            Tuple of (before_context, after_context)

        """
        if not content or start_pos >= end_pos:
            return "", ""
        
        before_start = max(0, start_pos - context_window)
        after_end = min(len(content), end_pos + context_window)
        
        before_context = content[before_start:start_pos]
        after_context = content[end_pos:after_end]
        
        return before_context, after_context

    def expand_context_to_boundaries(
        self,
        content: str,
        center_pos: int,
        initial_window: int,
        boundary_chars: str = ".!?。！？"
    ) -> Tuple[int, int]:
        """Expand context to natural boundaries like sentence endings.
        
        Args:
            content: Full text content
            center_pos: Center position to expand from
            initial_window: Initial context window size
            boundary_chars: Characters that represent natural boundaries
            
        Returns:
            Tuple of (start_pos, end_pos) for expanded context

        """
        if not content:
            return 0, 0
        
        # Initial boundaries
        start = max(0, center_pos - initial_window)
        end = min(len(content), center_pos + initial_window)
        
        # Expand start to previous boundary
        expanded_start = start
        for i in range(start - 1, -1, -1):
            if content[i] in boundary_chars:
                expanded_start = i + 1
                break
            expanded_start = i
        
        # Expand end to next boundary
        expanded_end = end
        for i in range(end, len(content)):
            if content[i] in boundary_chars:
                expanded_end = i + 1
                break
            expanded_end = i + 1
        
        return max(0, expanded_start), min(len(content), expanded_end)

    def get_optimal_context_window(
        self,
        content_length: int,
        target_snippet_length: int,
        match_count: int = 1
    ) -> int:
        """Calculate optimal context window size based on content and target length.
        
        Args:
            content_length: Total length of content
            target_snippet_length: Desired snippet length
            match_count: Number of matches to accommodate
            
        Returns:
            Recommended context window size

        """
        if content_length <= target_snippet_length:
            return content_length // 2
        
        # Base context window accounting for multiple matches
        base_window = target_snippet_length // (2 * max(1, match_count))
        
        # Ensure minimum window size
        min_window = 20
        
        return max(min_window, base_window)
