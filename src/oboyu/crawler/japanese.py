"""Japanese text processing utilities.

This module provides utilities for processing Japanese text.
Note: Most functionality has been moved to EncodingDetector service.
This file is kept for backward compatibility and will be removed in future versions.
"""

import re


def standardize_line_endings(text: str) -> str:
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
