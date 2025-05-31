"""Tokenizer service for text processing."""

import logging
from typing import Any, Dict, List, Optional

from oboyu.indexer.tokenizer import create_tokenizer

logger = logging.getLogger(__name__)


class TokenizerService:
    """Service for text tokenization."""

    def __init__(
        self,
        language: str = "ja",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize tokenizer service.

        Args:
            language: Primary language for tokenization
            tokenizer_kwargs: Additional tokenizer configuration

        """
        self.language = language
        self.tokenizer_kwargs = tokenizer_kwargs or {}

        # Initialize tokenizer
        try:
            self.tokenizer = create_tokenizer(
                language=language,
                **self.tokenizer_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens

        """
        if not self.tokenizer:
            # Fallback to simple whitespace tokenization
            return text.lower().split()

        try:
            return self.tokenizer.tokenize(text)
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            # Fallback to simple whitespace tokenization
            return text.lower().split()

    def tokenize_query(self, query: str) -> List[str]:
        """Tokenize search query.

        Args:
            query: Search query text

        Returns:
            List of query tokens

        """
        return self.tokenize(query)

    def is_available(self) -> bool:
        """Check if tokenizer is available.

        Returns:
            True if tokenizer is available, False otherwise

        """
        return self.tokenizer is not None
