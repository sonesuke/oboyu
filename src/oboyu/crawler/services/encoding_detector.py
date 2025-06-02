"""Encoding detection and Japanese text processing service."""

import re
from typing import List, Optional

import charset_normalizer
import ftfy
import mojimoji
import neologdn  # type: ignore[import-not-found]


class EncodingDetector:
    """Service responsible for handling encoding detection and conversion for Japanese files."""

    def detect_encoding(self, content: str, preferred_encodings: Optional[List[str]] = None) -> str:
        """Detect the encoding of Japanese text.
        
        Args:
            content: Text content
            preferred_encodings: List of preferred encodings to check (deprecated, kept for compatibility)
            
        Returns:
            Detected encoding
            
        Note:
            This is called after the text has already been decoded (in the extractor),
            so we're using heuristics to determine if we need to convert to another encoding.

        """
        # Default encodings for Japanese text
        default_encodings = ["utf-8", "shift-jis", "euc-jp", "cp932", "iso-2022-jp"]
        preferred_encodings = preferred_encodings or default_encodings
        
        # Check for Unicode replacement character, which indicates decoding issues
        has_replacement_char = "\ufffd" in content

        # Count Japanese characters to check if it's likely Japanese text
        kanji_count = len(re.findall(r"[\u4e00-\u9faf]", content))
        hiragana_count = len(re.findall(r"[\u3040-\u309f]", content))
        katakana_count = len(re.findall(r"[\u30a0-\u30ff]", content))
        has_japanese = (kanji_count + hiragana_count + katakana_count) > 0

        # If we seem to have valid Japanese text with no issues, assume it's UTF-8
        if has_japanese and not has_replacement_char:
            return "utf-8"

        # For strings with issues, use charset-normalizer for better detection
        if has_replacement_char:
            for encoding in preferred_encodings:
                try:
                    # Convert to bytes using the current encoding
                    encoded = content.encode(encoding, errors="replace")

                    # Use charset-normalizer for better detection
                    results = charset_normalizer.from_bytes(encoded).best()
                    if results and results.encoding:
                        return str(results.encoding)
                except (LookupError, ValueError, AttributeError):
                    # These are expected errors when trying different encodings
                    # Just try the next encoding
                    continue

        # Use default Japanese encodings
        japanese_encodings = default_encodings

        # If we found Japanese characters but couldn't determine encoding,
        # use the first Japanese-specific encoding if available
        if has_japanese and japanese_encodings:
            return japanese_encodings[0]

        # If we have preferred encodings, use the first one
        if preferred_encodings:
            return preferred_encodings[0]

        # Default to UTF-8
        return "utf-8"

    def process_japanese_text(self, content: str, encoding: str) -> str:
        """Process Japanese text using proven libraries.
        
        Args:
            content: Text content
            encoding: Detected encoding
            
        Returns:
            Normalized text

        """
        # Fix encoding issues with ftfy
        # ftfy automatically handles mojibake and other text encoding problems
        fixed_content = ftfy.fix_text(content)

        # Normalize Japanese text with neologdn
        # This handles Unicode normalization (NFKC), repeated characters, etc.
        normalized = neologdn.normalize(fixed_content)

        # Check if we need additional width conversion
        # neologdn already handles most normalization, but we may want to ensure
        # numbers and ASCII are half-width while keeping Japanese characters as-is
        if self._needs_width_conversion(normalized):
            # Convert only numbers and ASCII to half-width, keep kana as-is
            normalized = mojimoji.zen_to_han(normalized, kana=False, ascii=True, digit=True)

        # Standardize line endings
        standardized = self._standardize_line_endings(normalized)

        return standardized

    def _needs_width_conversion(self, text: str) -> bool:
        """Check if text needs width conversion.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains full-width ASCII or numbers

        """
        # Check for full-width numbers
        if re.search(r"[\uff10-\uff19]", text):
            return True

        # Check for full-width ASCII letters
        if re.search(r"[\uff21-\uff3a\uff41-\uff5a]", text):
            return True

        return False

    def _standardize_line_endings(self, text: str) -> str:
        """Standardize line endings to Unix style (LF).
        
        Args:
            text: Text content
            
        Returns:
            Text with standardized line endings

        """
        # Replace all Windows line endings with Unix line endings
        text = text.replace("\r\n", "\n")

        # Replace any remaining Mac line endings with Unix line endings
        text = text.replace("\r", "\n")

        # Normalize multiple consecutive newlines to at most two
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text
