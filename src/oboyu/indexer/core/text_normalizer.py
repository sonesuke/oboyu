"""Text normalization component for document processing.

This module handles text cleaning and normalization operations including
whitespace normalization and other text cleanup tasks.
"""

import re


class TextNormalizer:
    """Handles text cleaning and normalization operations."""

    def normalize(self, text: str, language: str = "en") -> str:
        """Normalize text by cleaning whitespace and other artifacts.

        Args:
            text: The text to normalize
            language: Language code (currently unused but available for future extensions)

        Returns:
            Normalized text with cleaned whitespace

        """
        # Remove excessive whitespace while preserving structure
        normalized = re.sub(r"\s+", " ", text).strip()
        
        return normalized

