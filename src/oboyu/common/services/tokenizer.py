"""Tokenizer service for text processing."""

import logging
import re
import unicodedata
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

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

# Import Japanese stop words from separate module
from oboyu.common.stop_words import DEFAULT_JAPANESE_STOP_WORDS

logger = logging.getLogger(__name__)

# POS tags to exclude from indexing (too generic or not content-bearing)
EXCLUDED_POS_TAGS = {
    "助詞",  # Particles
    "助動詞",  # Auxiliary verbs
    "記号",  # Symbols
    "補助記号",  # Auxiliary symbols
    "空白",  # Whitespace
    "フィラー",  # Fillers
    "感動詞",  # Interjections
}

# POS tags to include (content-bearing words)
INCLUDED_POS_TAGS = {
    "名詞",  # Nouns
    "動詞",  # Verbs
    "形容詞",  # Adjectives
    "副詞",  # Adverbs
    "連体詞",  # Attributive words
    "接続詞",  # Conjunctions
}

# Subcategory exclusions: (POS, subcategory) pairs to exclude
EXCLUDED_POS_SUBCATEGORIES = [
    ("名詞", "代名詞"),  # Pronouns
    ("名詞", "非自立"),  # Dependent nouns
    ("動詞", "非自立"),  # Dependent verbs
    ("形容詞", "非自立"),  # Dependent adjectives
]


class JapaneseTokenizer:
    """Japanese tokenizer using MeCab/fugashi for morphological analysis."""

    def __init__(
        self,
        stop_words: Optional[Set[str]] = None,
        min_token_length: int = 2,
        use_pos_filter: bool = True,
        normalize_text: bool = True,
        use_lemmatization: bool = True,
        debug: bool = False,
    ) -> None:
        """Initialize the Japanese tokenizer."""
        if not HAS_JAPANESE_TOKENIZER:
            raise ImportError("Required packages not available. Install with: pip install fugashi[unidic-lite] jaconv")

        self.stop_words = stop_words or DEFAULT_JAPANESE_STOP_WORDS
        self.min_token_length = min_token_length
        self.use_pos_filter = use_pos_filter
        self.normalize_text = normalize_text
        self.use_lemmatization = use_lemmatization
        self.debug = debug

        # Initialize MeCab tagger
        try:
            self.tagger = fugashi.Tagger()
        except Exception as e:
            logger.error(f"Failed to initialize MeCab tagger: {e}")
            raise

        # Debug tracking
        if self.debug:
            self.pos_distribution: Dict[str, int] = defaultdict(int)
            self.filtered_by_pos: Dict[str, int] = defaultdict(int)
            self.filtered_by_stopwords: Dict[str, int] = defaultdict(int)
            self.filtered_by_length: Dict[str, int] = defaultdict(int)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Japanese text into meaningful tokens."""
        if not text:
            return []

        # Normalize text if enabled
        normalized_text = self._normalize_japanese_text(text) if self.normalize_text else text
        tokens = []

        # Use MeCab to parse the text
        for node in self.tagger(normalized_text):
            token = node.surface
            # Handle different feature types (string vs UnidicFeatures object)
            if hasattr(node, "feature"):
                if isinstance(node.feature, str):
                    features = node.feature.split(",")
                else:
                    # For UnidicFeatures objects, convert to string first
                    features = str(node.feature).split(",")
            else:
                features = []

            # Apply minimum length filter
            if len(token) < self.min_token_length:
                if self.debug:
                    self.filtered_by_length[token] += 1
                continue

            # Apply stop words filter
            if token.lower() in self.stop_words:
                if self.debug:
                    self.filtered_by_stopwords[token] += 1
                continue

            # Apply POS filtering if enabled
            if self.use_pos_filter:
                pos = features[0]  # First feature is the part-of-speech
                subpos1 = features[1] if len(features) > 1 else ""
                subpos2 = features[2] if len(features) > 2 else ""

                # Track POS distribution
                if self.debug:
                    full_pos = f"{pos}-{subpos1}-{subpos2}"
                    self.pos_distribution[full_pos] += 1

                # Skip excluded POS tags
                if pos in EXCLUDED_POS_TAGS:
                    if self.debug:
                        self.filtered_by_pos[f"{token}({pos})"] += 1
                    continue

                # Check subcategory exclusions
                excluded_by_subcategory = False
                for exc_pos, exc_subpos in EXCLUDED_POS_SUBCATEGORIES:
                    if pos == exc_pos and (exc_subpos == "*" or subpos1 == exc_subpos):
                        if self.debug:
                            self.filtered_by_pos[f"{token}({pos}-{subpos1})"] += 1
                        excluded_by_subcategory = True
                        break

                if excluded_by_subcategory:
                    continue

                # For included POS tags, use the base form if available
                if pos in INCLUDED_POS_TAGS and self.use_lemmatization:
                    # Try to get the base form (lemma)
                    base_form = features[7] if len(features) > 7 else None
                    if base_form and base_form != "*":
                        # Skip lemmatization if it adds non-Japanese suffixes
                        if "-" in base_form and not self.is_japanese_text(base_form.split("-", 1)[1]):
                            # Keep the original surface form
                            pass
                        else:
                            token = base_form

            tokens.append(token)

        return tokens

    def tokenize_with_positions(self, text: str) -> List[Tuple[str, int, int]]:
        """Tokenize text and return tokens with their positions."""
        if not text:
            return []

        # Normalize text if enabled
        normalized_text = self._normalize_japanese_text(text) if self.normalize_text else text
        tokens_with_positions = []

        # Use MeCab to parse the text
        for node in self.tagger(normalized_text):
            token = node.surface
            # Handle different feature types (string vs UnidicFeatures object)
            if hasattr(node, "feature"):
                if isinstance(node.feature, str):
                    features = node.feature.split(",")
                else:
                    # For UnidicFeatures objects, convert to string first
                    features = str(node.feature).split(",")
            else:
                features = []

            # Get position information
            start_pos = node.char_begin if hasattr(node, "char_begin") else 0
            end_pos = node.char_end if hasattr(node, "char_end") else start_pos + len(token)

            # Apply same filtering as tokenize()
            if len(token) < self.min_token_length:
                continue

            if token.lower() in self.stop_words:
                continue

            if self.use_pos_filter:
                pos = features[0]
                if pos in EXCLUDED_POS_TAGS:
                    continue

                if pos in INCLUDED_POS_TAGS and self.use_lemmatization:
                    base_form = features[7] if len(features) > 7 else None
                    if base_form and base_form != "*":
                        token = base_form

            tokens_with_positions.append((token, start_pos, end_pos))

        return tokens_with_positions

    def get_term_frequencies(self, text: str) -> Dict[str, int]:
        """Get term frequencies from text."""
        tokens = self.tokenize(text)
        term_freq: Dict[str, int] = {}

        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        return term_freq

    def _normalize_japanese_text(self, text: str) -> str:
        """Normalize Japanese text for consistent tokenization."""
        # Unicode normalization (NFKC)
        text = unicodedata.normalize("NFKC", text)

        # Convert to lowercase for non-Japanese characters
        text = text.lower()

        # Use jaconv for additional Japanese-specific conversions
        if jaconv:
            # Convert half-width katakana to full-width
            text = jaconv.h2z(text, kana=True, ascii=False, digit=False)

            # Normalize tildes and dashes
            text = text.replace("〜", "ー")
            text = text.replace("～", "ー")

        return text

    def is_japanese_text(self, text: str) -> bool:
        """Check if text contains Japanese characters."""
        # Check for Hiragana, Katakana, or Kanji
        for char in text:
            if "\u3040" <= char <= "\u309f":  # Hiragana
                return True
            if "\u30a0" <= char <= "\u30ff":  # Katakana
                return True
            if "\u4e00" <= char <= "\u9fff":  # Kanji
                return True
        return False


class FallbackTokenizer:
    """Simple fallback tokenizer for when fugashi is not available."""

    def __init__(
        self,
        stop_words: Optional[Set[str]] = None,
        min_token_length: int = 2,
    ) -> None:
        """Initialize the fallback tokenizer."""
        self.stop_words = stop_words or set()
        self.min_token_length = min_token_length

        # Pattern for basic tokenization
        self._token_pattern = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+|[a-zA-Z0-9]+")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using regex patterns."""
        if not text:
            return []

        # Find all token matches
        matches = self._token_pattern.findall(text.lower())

        # Filter tokens
        tokens = []
        for token in matches:
            if len(token) >= self.min_token_length and token not in self.stop_words:
                tokens.append(token)

        return tokens

    def get_term_frequencies(self, text: str) -> Dict[str, int]:
        """Get term frequencies from text."""
        tokens = self.tokenize(text)
        term_freq: Dict[str, int] = {}

        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        return term_freq


class TokenizerService:
    """Service for text tokenization."""

    def __init__(
        self,
        language: str = "ja",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize tokenizer service."""
        self.language = language
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        
        # Type annotation for tokenizer field
        self.tokenizer: Optional[Union[JapaneseTokenizer, FallbackTokenizer]] = None

        # Initialize tokenizer
        try:
            self.tokenizer = self._create_tokenizer(language, **self.tokenizer_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None

    def _create_tokenizer(
        self,
        language: str = "ja",
        stop_words: Optional[Set[str]] = None,
        min_token_length: int = 2,
        use_fallback: bool = False,
        use_stopwords: bool = True,
    ) -> Union[JapaneseTokenizer, FallbackTokenizer]:
        """Create an appropriate tokenizer for the given language."""
        # Use default stop words if enabled and none provided
        if use_stopwords and stop_words is None:
            stop_words = DEFAULT_JAPANESE_STOP_WORDS
        elif not use_stopwords:
            stop_words = set()

        if language == "ja" and not use_fallback:
            if HAS_JAPANESE_TOKENIZER:
                return JapaneseTokenizer(
                    stop_words=stop_words,
                    min_token_length=min_token_length,
                )
            else:
                logger.warning("Japanese tokenizer dependencies not available, using fallback")
                return FallbackTokenizer(
                    stop_words=stop_words,
                    min_token_length=min_token_length,
                )
        else:
            # Use fallback for non-Japanese or when explicitly requested
            return FallbackTokenizer(
                stop_words=stop_words,
                min_token_length=min_token_length,
            )

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
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
        """Tokenize search query."""
        return self.tokenize(query)

    def is_available(self) -> bool:
        """Check if tokenizer is available."""
        return self.tokenizer is not None

    def get_term_frequencies(self, text: str) -> Dict[str, int]:
        """Get term frequencies from text."""
        if not self.tokenizer:
            return {}

        try:
            return self.tokenizer.get_term_frequencies(text)
        except Exception as e:
            logger.error(f"Term frequency calculation failed: {e}")
            return {}


# Factory function for backward compatibility
def create_tokenizer(
    language: str = "ja",
    stop_words: Optional[Set[str]] = None,
    min_token_length: int = 2,
    use_fallback: bool = False,
    use_stopwords: bool = True,
) -> Union[JapaneseTokenizer, FallbackTokenizer]:
    """Create an appropriate tokenizer for the given language."""
    service = TokenizerService(language=language)
    return service._create_tokenizer(language, stop_words, min_token_length, use_fallback, use_stopwords)
