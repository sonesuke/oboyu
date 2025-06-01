"""Legacy import compatibility for tokenizer module."""

# Re-export from the new location
from oboyu.indexer.models.tokenizer_service import (
    FallbackTokenizer,
    JapaneseTokenizer,
    TokenizerService,
    create_tokenizer,
)

__all__ = [
    "FallbackTokenizer",
    "JapaneseTokenizer",
    "TokenizerService",
    "create_tokenizer",
]
