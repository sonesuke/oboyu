"""Text utility functions for the CLI."""

import re

from rich.console import Console

# Create console for rich output
console = Console()


def detect_language(text: str) -> str:
    """Detect language of text with special handling for Japanese.

    Args:
        text: Text to detect language for

    Returns:
        Language code (e.g., 'ja', 'en')

    """
    # Import here to avoid circular imports
    from oboyu.crawler.extractor import _detect_language

    # Use the FastText-based detection from extractor
    return _detect_language(text)


def contains_japanese(text: str) -> bool:
    """Check if text contains Japanese characters.

    Args:
        text: Text to check

    Returns:
        True if text contains Japanese characters, False otherwise

    """
    # Check for Hiragana, Katakana, or Kanji
    japanese_ranges = [
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0x4E00, 0x9FFF),  # Kanji (CJK Unified Ideographs)
        (0x3400, 0x4DBF),  # Kanji (CJK Unified Ideographs Extension A)
    ]

    for char in text:
        code_point = ord(char)
        for start, end in japanese_ranges:
            if start <= code_point <= end:
                return True

    return False


def format_snippet(text: str, query: str, length: int = 160, highlight: bool = True) -> str:
    """Format a text snippet with optional highlighting.

    Args:
        text: Full text content
        query: Search query
        length: Maximum snippet length
        highlight: Whether to highlight query terms

    Returns:
        Formatted snippet

    """
    # Find match position (simple implementation for now)
    query_terms = re.split(r"\s+", query.lower())
    match_pos = -1

    for term in query_terms:
        if term and len(term) > 2:  # Skip short terms
            pos = text.lower().find(term)
            if pos >= 0:
                match_pos = pos
                break

    # Extract snippet around match
    if match_pos >= 0:
        start = max(0, match_pos - length // 2)
        end = min(len(text), start + length)

        # Adjust to avoid breaking words
        while start > 0 and text[start] != " ":
            start -= 1

        while end < len(text) and text[end] != " ":
            end += 1

        snippet = text[start:end]

        # Add ellipsis if necessary
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        snippet = f"{prefix}{snippet}{suffix}"
    else:
        # If no match found, use the beginning of the text
        snippet = text[:length] + ("..." if len(text) > length else "")

    # Highlighting disabled for cleaner output
    # (Previously highlighted query terms)

    return snippet


def format_path_for_display(path: str, max_width: int = 60, preserve_components: int = 2) -> str:
    """Format a file path for display with intelligent truncation.
    
    Args:
        path: File path to format
        max_width: Maximum width for the path display
        preserve_components: Number of final path components to always preserve
        
    Returns:
        Formatted path string with intelligent truncation
        
    """
    import os
    from pathlib import Path
    
    # Normalize the path
    path_obj = Path(path).resolve()
    
    # Convert to home directory format if applicable
    try:
        relative_to_home = path_obj.relative_to(Path.home())
        formatted_path = f"~/{relative_to_home}"
    except ValueError:
        # Path is not within home directory
        formatted_path = str(path_obj)
    
    # If path is already short enough, return as-is
    if len(formatted_path) <= max_width:
        return formatted_path
    
    # Split path into components
    parts = formatted_path.split(os.sep)
    
    # Always preserve the final components (e.g., final directory and filename)
    preserved_parts = parts[-preserve_components:] if len(parts) >= preserve_components else parts
    preserved_length = len(os.sep.join(preserved_parts))
    
    # Calculate available space for middle components
    prefix = parts[0] if parts[0] else os.sep  # Root or first component
    available_space = max_width - len(prefix) - preserved_length - len("/.../")
    
    if available_space <= 0:
        # Not enough space, show minimal format: prefix/.../final_components
        if len(parts) > preserve_components + 1:
            return f"{prefix}/.../{''.join(preserved_parts)}"
        else:
            # Very short path, truncate in middle
            return f"{formatted_path[:max_width//2]}...{formatted_path[-(max_width//2):]}"
    
    # Try to fit some middle components
    middle_parts = parts[1:-preserve_components] if len(parts) > preserve_components + 1 else []
    included_middle = []
    current_length = 0
    
    # Add middle components while they fit
    for part in middle_parts:
        part_length = len(part) + 1  # +1 for separator
        if current_length + part_length <= available_space:
            included_middle.append(part)
            current_length += part_length
        else:
            break
    
    # Construct the final path
    if included_middle:
        if len(included_middle) < len(middle_parts):
            # Some middle parts were omitted
            result_parts = [prefix] + included_middle + ["..."] + preserved_parts
        else:
            # All middle parts included
            result_parts = [prefix] + included_middle + preserved_parts
    else:
        # No middle parts could fit
        if len(middle_parts) > 0:
            result_parts = [prefix, "..."] + preserved_parts
        else:
            result_parts = [prefix] + preserved_parts
    
    # Join path components and clean up any double separators
    result = os.sep.join(filter(None, result_parts))
    
    # Clean up multiple consecutive separators
    while os.sep + os.sep in result:
        result = result.replace(os.sep + os.sep, os.sep)
    
    return result
