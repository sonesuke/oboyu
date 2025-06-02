"""Japanese-specific text processing and boundary detection."""

import re
from typing import List


class JapaneseSnippetProcessor:
    """Handles Japanese-specific text processing and boundaries."""

    # Japanese sentence boundary patterns
    JAPANESE_SENTENCE_ENDINGS = re.compile(r'[。！？]')
    JAPANESE_PARAGRAPH_BREAKS = re.compile(r'\n\s*\n')
    
    # Japanese character ranges
    HIRAGANA_RANGE = re.compile(r'[\u3040-\u309F]')
    KATAKANA_RANGE = re.compile(r'[\u30A0-\u30FF]')
    KANJI_RANGE = re.compile(r'[\u4E00-\u9FAF]')
    JAPANESE_PUNCTUATION = re.compile(r'[\u3000-\u303F]')

    def find_sentence_boundaries(self, text: str) -> List[int]:
        """Find Japanese sentence boundaries in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of positions where sentences end

        """
        boundaries = []
        
        for match in self.JAPANESE_SENTENCE_ENDINGS.finditer(text):
            boundaries.append(match.end())
        
        return boundaries

    def find_paragraph_boundaries(self, text: str) -> List[int]:
        """Find Japanese paragraph boundaries in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of positions where paragraphs end

        """
        boundaries = []
        
        for match in self.JAPANESE_PARAGRAPH_BREAKS.finditer(text):
            boundaries.append(match.start())
        
        return boundaries

    def is_japanese_text(self, text: str) -> bool:
        """Check if text contains Japanese characters.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains Japanese characters

        """
        if not text:
            return False
        
        # Check for any Japanese character ranges
        return bool(
            self.HIRAGANA_RANGE.search(text) or
            self.KATAKANA_RANGE.search(text) or
            self.KANJI_RANGE.search(text)
        )

    def normalize_japanese_text(self, text: str) -> str:
        """Normalize Japanese text for processing.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text

        """
        if not text:
            return text
        
        # Basic normalization - could be expanded with more rules
        normalized = text
        
        # Normalize whitespace around Japanese punctuation
        normalized = re.sub(r'\s+([。！？])', r'\1', normalized)
        normalized = re.sub(r'([。！？])\s+', r'\1 ', normalized)
        
        return normalized.strip()

    def adjust_to_sentence_boundaries(
        self,
        text: str,
        prefer_complete: bool = True
    ) -> str:
        """Adjust text to end at Japanese sentence boundaries.
        
        Args:
            text: Text to adjust
            prefer_complete: Whether to prefer complete sentences
            
        Returns:
            Text adjusted to sentence boundaries

        """
        if not text or not prefer_complete:
            return text
        
        boundaries = self.find_sentence_boundaries(text)
        
        if boundaries:
            # Use the last sentence boundary that fits
            last_boundary = boundaries[-1]
            return text[:last_boundary].strip()
        
        # No sentence boundary found, return as is
        return text

    def adjust_to_paragraph_boundaries(
        self,
        text: str,
        prefer_complete: bool = True
    ) -> str:
        """Adjust text to end at Japanese paragraph boundaries.
        
        Args:
            text: Text to adjust
            prefer_complete: Whether to prefer complete paragraphs
            
        Returns:
            Text adjusted to paragraph boundaries

        """
        if not text or not prefer_complete:
            return text
        
        boundaries = self.find_paragraph_boundaries(text)
        
        if boundaries:
            # Use the last paragraph boundary that fits
            last_boundary = boundaries[-1]
            return text[:last_boundary].strip()
        
        # Fallback to sentence boundaries
        return self.adjust_to_sentence_boundaries(text, prefer_complete)

    def get_character_density(self, text: str) -> dict[str, float]:
        """Get density of different Japanese character types.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with character type densities

        """
        if not text:
            return {"hiragana": 0.0, "katakana": 0.0, "kanji": 0.0, "other": 1.0}
        
        total_chars = len(text)
        hiragana_count = len(self.HIRAGANA_RANGE.findall(text))
        katakana_count = len(self.KATAKANA_RANGE.findall(text))
        kanji_count = len(self.KANJI_RANGE.findall(text))
        other_count = total_chars - hiragana_count - katakana_count - kanji_count
        
        return {
            "hiragana": hiragana_count / total_chars,
            "katakana": katakana_count / total_chars,
            "kanji": kanji_count / total_chars,
            "other": other_count / total_chars
        }

    def avoid_word_breaks_japanese(self, text: str) -> str:
        """Avoid breaking Japanese text at inappropriate positions.
        
        Args:
            text: Text to adjust
            
        Returns:
            Text with word breaks avoided

        """
        if not text:
            return text
        
        # For Japanese text, we generally don't need to worry about
        # word breaks in the same way as English, but we should avoid
        # breaking in the middle of punctuation sequences
        
        # Remove trailing punctuation if incomplete
        text = text.rstrip('、。！？‥…')
        
        return text.strip()
