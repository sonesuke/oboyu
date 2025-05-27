"""Japanese tokenizer for BM25 search.

This module provides tokenization functionality optimized for Japanese text,
using fugashi (MeCab wrapper) for morphological analysis.
"""

import logging
import re
import unicodedata
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    import fugashi
    import jaconv
    import unidic_lite

try:
    import fugashi
    import jaconv
    import unidic_lite  # noqa: F401  # Required for MeCab to find dictionary
    HAS_JAPANESE_TOKENIZER = True
except ImportError:
    HAS_JAPANESE_TOKENIZER = False

logger = logging.getLogger(__name__)


# Default Japanese stop words
DEFAULT_JAPANESE_STOP_WORDS = {
    # Particles
    "は", "が", "を", "に", "で", "と", "も", "や", "の", "へ", "から", "まで", "より", "ね", "よ",
    # Auxiliary verbs
    "です", "ます", "だ", "である", "でした", "ました", "でしょう", "ましょう",
    # Pronouns
    "これ", "それ", "あれ", "どれ", "この", "その", "あの", "どの",
    # Common words
    "こと", "もの", "ため", "とき", "ところ", "ほう", "さん", "くん", "ちゃん",
    # English common words in Japanese text
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
}

# Part-of-speech tags to exclude (non-content words)
EXCLUDED_POS_TAGS = {
    "助詞",      # Particles
    "助動詞",    # Auxiliary verbs
    "記号",      # Symbols
    "補助記号",  # Auxiliary symbols
}

# Part-of-speech tags to include (content words)
INCLUDED_POS_TAGS = {
    "名詞",      # Nouns
    "動詞",      # Verbs
    "形容詞",    # Adjectives
    "形容動詞",  # Adjectival nouns
    "副詞",      # Adverbs
}


class JapaneseTokenizer:
    """Tokenizer for Japanese text using MeCab morphological analyzer.
    
    This tokenizer provides:
    - Morphological analysis-based tokenization
    - Part-of-speech filtering
    - Stop word removal
    - Text normalization
    - Support for mixed Japanese/English text
    """
    
    def __init__(
        self,
        stop_words: Optional[Set[str]] = None,
        min_token_length: int = 2,
        use_pos_filter: bool = True,
        normalize_text: bool = True,
    ) -> None:
        """Initialize the Japanese tokenizer.
        
        Args:
            stop_words: Set of stop words to exclude (uses default if None)
            min_token_length: Minimum token length to keep
            use_pos_filter: Whether to filter tokens by part-of-speech
            normalize_text: Whether to normalize text before tokenization
            
        Raises:
            ImportError: If fugashi is not installed

        """
        if not HAS_JAPANESE_TOKENIZER:
            raise ImportError(
                "Japanese tokenizer dependencies not installed. "
                "Install with: pip install fugashi unidic-lite jaconv"
            )
        
        self.tagger = fugashi.Tagger()
        self.stop_words = stop_words or DEFAULT_JAPANESE_STOP_WORDS
        self.min_token_length = min_token_length
        self.use_pos_filter = use_pos_filter
        self.normalize_text = normalize_text
        
        # Compile regex patterns
        self._english_pattern = re.compile(r'[a-zA-Z]+')
        self._number_pattern = re.compile(r'\d+')
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Japanese text into tokens.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens

        """
        if not text:
            return []
        
        # Normalize text if enabled
        if self.normalize_text:
            text = self._normalize_japanese_text(text)
        
        tokens = []
        
        # Use MeCab to parse the text
        for node in self.tagger(text):
            # Get surface form and features
            token = node.surface
            features = node.feature
            
            # Skip if token is too short
            if len(token) < self.min_token_length:
                continue
            
            # Skip stop words
            if token.lower() in self.stop_words:
                continue
            
            # Apply POS filtering if enabled
            if self.use_pos_filter:
                pos = features[0]  # First feature is the part-of-speech
                
                # Skip excluded POS tags
                if pos in EXCLUDED_POS_TAGS:
                    continue
                
                # For included POS tags, use the base form if available
                if pos in INCLUDED_POS_TAGS:
                    # Try to get the base form (lemma)
                    base_form = features[7] if len(features) > 7 else None
                    if base_form and base_form != "*":
                        token = base_form
            
            tokens.append(token)
        
        return tokens
    
    def tokenize_with_positions(self, text: str) -> List[Tuple[str, int, int]]:
        """Tokenize text and return tokens with their positions.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of (token, start_pos, end_pos) tuples

        """
        if not text:
            return []
        
        # Normalize text if enabled
        normalized_text = self._normalize_japanese_text(text) if self.normalize_text else text
        
        tokens_with_positions = []
        
        # Use MeCab to parse the text
        for node in self.tagger(normalized_text):
            token = node.surface
            features = node.feature
            
            # Get position information
            # Note: fugashi doesn't directly provide character positions,
            # so we need to track them ourselves
            start_pos = node.char_begin if hasattr(node, 'char_begin') else 0
            end_pos = node.char_end if hasattr(node, 'char_end') else start_pos + len(token)
            
            # Apply same filtering as tokenize()
            if len(token) < self.min_token_length:
                continue
            
            if token.lower() in self.stop_words:
                continue
            
            if self.use_pos_filter:
                pos = features[0]
                if pos in EXCLUDED_POS_TAGS:
                    continue
                
                if pos in INCLUDED_POS_TAGS:
                    base_form = features[7] if len(features) > 7 else None
                    if base_form and base_form != "*":
                        token = base_form
            
            tokens_with_positions.append((token, start_pos, end_pos))
        
        return tokens_with_positions
    
    def get_term_frequencies(self, text: str) -> Dict[str, int]:
        """Get term frequencies from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping terms to their frequencies

        """
        tokens = self.tokenize(text)
        term_freq: Dict[str, int] = {}
        
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        
        return term_freq
    
    def _normalize_japanese_text(self, text: str) -> str:
        """Normalize Japanese text for consistent tokenization.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text

        """
        # Unicode normalization (NFKC)
        text = unicodedata.normalize('NFKC', text)
        
        # Convert to lowercase for non-Japanese characters
        # Japanese doesn't have case, but mixed text might have English
        text = text.lower()
        
        # Use jaconv for additional Japanese-specific conversions
        if jaconv:
            # Convert half-width katakana to full-width
            text = jaconv.h2z(text, kana=True, ascii=False, digit=False)
            
            # Normalize tildes and dashes
            text = text.replace('〜', 'ー')
            text = text.replace('～', 'ー')
        
        return text
    
    def is_japanese_text(self, text: str) -> bool:
        """Check if text contains Japanese characters.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains Japanese characters

        """
        # Check for Hiragana, Katakana, or Kanji
        for char in text:
            if '\u3040' <= char <= '\u309f':  # Hiragana
                return True
            if '\u30a0' <= char <= '\u30ff':  # Katakana
                return True
            if '\u4e00' <= char <= '\u9fff':  # Kanji
                return True
        return False


class FallbackTokenizer:
    """Simple fallback tokenizer for when fugashi is not available.
    
    This tokenizer provides basic functionality using regex patterns,
    suitable for simple cases but not recommended for production use
    with Japanese text.
    """
    
    def __init__(
        self,
        stop_words: Optional[Set[str]] = None,
        min_token_length: int = 2,
    ) -> None:
        """Initialize the fallback tokenizer.
        
        Args:
            stop_words: Set of stop words to exclude
            min_token_length: Minimum token length to keep

        """
        self.stop_words = stop_words or set()
        self.min_token_length = min_token_length
        
        # Pattern for basic tokenization
        # Matches sequences of:
        # - Hiragana/Katakana/Kanji
        # - Alphanumeric characters
        self._token_pattern = re.compile(
            r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+|[a-zA-Z0-9]+'
        )
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using regex patterns.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens

        """
        if not text:
            return []
        
        # Normalize text
        text = unicodedata.normalize('NFKC', text)
        text = text.lower()
        
        # Extract tokens
        tokens = []
        for match in self._token_pattern.finditer(text):
            token = match.group()
            
            # Apply filters
            if len(token) < self.min_token_length:
                continue
            if token in self.stop_words:
                continue
            
            tokens.append(token)
        
        return tokens
    
    def get_term_frequencies(self, text: str) -> Dict[str, int]:
        """Get term frequencies from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping terms to their frequencies

        """
        tokens = self.tokenize(text)
        term_freq: Dict[str, int] = {}
        
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        
        return term_freq


def create_tokenizer(
    language: str = "ja",
    stop_words: Optional[Set[str]] = None,
    min_token_length: int = 2,
    use_fallback: bool = False,
) -> Union[JapaneseTokenizer, FallbackTokenizer]:
    """Create an appropriate tokenizer for the given language.
    
    Args:
        language: Language code (e.g., "ja" for Japanese)
        stop_words: Set of stop words to exclude
        min_token_length: Minimum token length to keep
        use_fallback: Force use of fallback tokenizer
        
    Returns:
        Tokenizer instance

    """
    if language == "ja" and not use_fallback:
        if HAS_JAPANESE_TOKENIZER:
            return JapaneseTokenizer(
                stop_words=stop_words,
                min_token_length=min_token_length,
            )
        else:
            logger.warning(
                "Japanese tokenizer not available, using fallback. "
                "For better results, install: pip install fugashi unidic-lite jaconv"
            )
            return FallbackTokenizer(
                stop_words=stop_words or DEFAULT_JAPANESE_STOP_WORDS,
                min_token_length=min_token_length,
            )
    else:
        # For non-Japanese languages, use the fallback tokenizer
        return FallbackTokenizer(
            stop_words=stop_words,
            min_token_length=min_token_length,
        )
