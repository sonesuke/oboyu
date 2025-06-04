"""Tokenizer service port interface."""

from abc import ABC, abstractmethod
from typing import List

from ...domain.value_objects.language_code import LanguageCode


class TokenizerService(ABC):
    """Abstract interface for text tokenization services."""
    
    @abstractmethod
    def tokenize(self, text: str, language: LanguageCode) -> List[str]:
        """Tokenize text into terms."""
        pass
    
    @abstractmethod
    def tokenize_query(self, query: str, language: LanguageCode) -> List[str]:
        """Tokenize a search query into terms."""
        pass
    
    @abstractmethod
    def normalize_text(self, text: str, language: LanguageCode) -> str:
        """Normalize text for processing."""
        pass
    
    @abstractmethod
    def supports_language(self, language: LanguageCode) -> bool:
        """Check if tokenizer supports a specific language."""
        pass
