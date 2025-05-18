"""Japanese text processing module for Oboyu.

This module provides utilities for handling Japanese text and encodings.
"""

import re
from typing import List

import chardet
import charset_normalizer


def detect_encoding(content: str, preferred_encodings: List[str]) -> str:
    """Detect the encoding of Japanese text.

    Args:
        content: Text content
        preferred_encodings: List of preferred encodings to check

    Returns:
        Detected encoding

    Note:
        This is called after the text has already been decoded (in the extractor),
        so we're using heuristics to determine if we need to convert to another encoding.

    """
    # If content is already in memory as a string, it's already been decoded
    # But we can still check for common signs of encoding issues

    # Check for Unicode replacement character, which indicates decoding issues
    has_replacement_char = '\ufffd' in content

    # Count Japanese characters to check if it's likely Japanese text
    kanji_count = len(re.findall(r'[\u4e00-\u9faf]', content))
    hiragana_count = len(re.findall(r'[\u3040-\u309f]', content))
    katakana_count = len(re.findall(r'[\u30a0-\u30ff]', content))
    has_japanese = (kanji_count + hiragana_count + katakana_count) > 0

    # If we seem to have valid Japanese text with no issues, assume it's UTF-8
    if has_japanese and not has_replacement_char:
        return 'utf-8'

    # For strings with issues, try multiple approaches
    if has_replacement_char:
        # First, try charset-normalizer for more accurate detection
        for encoding in preferred_encodings:
            try:
                # Convert to bytes using the current encoding
                encoded = content.encode(encoding, errors='replace')

                # Use charset-normalizer for better detection
                results = charset_normalizer.from_bytes(encoded).best()
                if results:
                    detected_encoding = results.encoding
                    if detected_encoding:
                        return str(detected_encoding)
            except (LookupError, ValueError, AttributeError):
                # These are expected errors when trying different encodings
                # Just try the next encoding
                continue

        # If charset-normalizer doesn't produce good results, fall back to chardet
        for encoding in preferred_encodings:
            try:
                # Try to encode with each preferred encoding
                encoded = content.encode(encoding, errors='replace')
                detected = chardet.detect(encoded)
                confidence = detected.get('confidence', 0)
                if confidence > 0.7 and detected['encoding']:  # Only use if confidence is high
                    return str(detected['encoding'])
            except (LookupError, ValueError, KeyError):
                # These are expected errors when trying different encodings
                # Just try the next encoding
                continue

    # Prioritize Japanese-specific encodings when Japanese characters are present
    japanese_encodings = [enc for enc in preferred_encodings if enc.lower() in
                         ('utf-8', 'shift-jis', 'sjis', 'euc-jp', 'cp932', 'iso-2022-jp')]

    # If we found Japanese characters but couldn't determine encoding,
    # use the first Japanese-specific encoding if available
    if has_japanese and japanese_encodings:
        return japanese_encodings[0]

    # If we have preferred encodings, use the first one
    if preferred_encodings:
        return preferred_encodings[0]

    # Default to UTF-8
    return 'utf-8'


def process_japanese_text(content: str, encoding: str) -> str:
    """Process Japanese text for better handling.

    Args:
        content: Text content
        encoding: Detected encoding

    Returns:
        Normalized text

    """
    # Normalize text
    normalized = _normalize_japanese(content)

    # Fix any encoding issues
    fixed = _fix_encoding_issues(normalized, encoding)

    # Standardize line endings
    standardized = _standardize_line_endings(fixed)

    return standardized


def _normalize_japanese(text: str) -> str:
    """Normalize Japanese text.

    Args:
        text: Japanese text

    Returns:
        Normalized text

    """
    # Convert full-width numbers to half-width
    text = re.sub(r'[\uff10-\uff19]', lambda x: chr(ord(x.group(0)) - 0xfee0), text)

    # Convert full-width alphabet to half-width
    text = re.sub(r'[\uff21-\uff3a\uff41-\uff5a]', lambda x: chr(ord(x.group(0)) - 0xfee0), text)

    # Normalize Japanese space characters
    text = text.replace('\u3000', ' ')

    # Convert various Japanese symbols to standardized forms
    # (This would be more extensive in a full implementation)
    text = text.replace('～', '〜')
    text = text.replace('－', '-')

    return text


def _fix_encoding_issues(text: str, encoding: str) -> str:
    """Fix common encoding issues in Japanese text.

    Args:
        text: Japanese text
        encoding: Detected encoding

    Returns:
        Fixed text

    """
    # Replace Unicode replacement character
    text = text.replace('\ufffd', '')

    # Fix common Shift-JIS conversion issues
    if encoding == 'shift-jis':
        # Fix common Shift-JIS mojibake patterns
        # This is a simplified version - a full implementation would be more comprehensive
        pass

    # Fix common EUC-JP conversion issues
    elif encoding == 'euc-jp':
        # Fix common EUC-JP mojibake patterns
        # This is a simplified version - a full implementation would be more comprehensive
        pass

    return text


def _standardize_line_endings(text: str) -> str:
    """Standardize line endings to Unix style (LF).

    Args:
        text: Text content

    Returns:
        Text with standardized line endings

    """
    # Replace all Windows line endings with Unix line endings
    text = text.replace('\r\n', '\n')

    # Replace any remaining Mac line endings with Unix line endings
    text = text.replace('\r', '\n')

    # Normalize multiple consecutive newlines to at most two
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text
