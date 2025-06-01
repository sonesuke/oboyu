"""Utility functions for the Oboyu CLI.

This module provides helper functions for the CLI.
"""

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
