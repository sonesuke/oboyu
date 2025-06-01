"""Language-specific text processing component.

This module handles language-specific text preparation, including
special processing for Japanese text and other language-specific operations.
"""

from oboyu.crawler.japanese import process_japanese_text


class LanguageProcessor:
    """Handles language-specific text preparation."""

    def prepare_text(self, text: str, language: str) -> str:
        """Prepare text based on its language.

        Args:
            text: Original text content
            language: Language code (e.g., 'ja' for Japanese, 'en' for English)

        Returns:
            Text prepared for the specific language

        """
        # Apply special processing for Japanese text
        if language == "ja":
            # Use the Japanese text processing from the crawler component
            return process_japanese_text(text, "utf-8")
            
        # For other languages, return as-is (can be extended in the future)
        return text

