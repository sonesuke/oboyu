"""Utility functions for the Oboyu CLI.

This module provides helper functions for the CLI.
"""

import locale
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
    from langdetect import detect

    # Check for Japanese characters
    if contains_japanese(text):
        return "ja"

    # Use langdetect for other languages
    try:
        result: str = detect(text)
        return result
    except Exception:
        # Default to English if detection fails
        return "en"


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

    # Highlight query terms if requested
    if highlight:
        for term in query_terms:
            if term and len(term) > 2:  # Skip short terms
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                snippet = pattern.sub(f"[bold][yellow]{term}[/yellow][/bold]", snippet)

    return snippet


def get_system_locale() -> str:
    """Get the system locale.

    Returns:
        System locale string

    """
    try:
        # Get system locale
        current_locale = locale.getlocale()[0]
        return current_locale or "en_US"
    except Exception:
        # Default to English if detection fails
        return "en_US"


def is_japanese_locale() -> bool:
    """Check if system locale is Japanese.

    Returns:
        True if system locale is Japanese, False otherwise

    """
    current_locale = get_system_locale().lower()
    return current_locale.startswith("ja")
