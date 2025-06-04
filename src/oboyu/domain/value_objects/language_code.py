"""Language code value object."""

from enum import Enum


class LanguageCode(str, Enum):
    """Supported language codes as value object."""
    
    JAPANESE = "ja"
    ENGLISH = "en"
    CHINESE = "zh"
    KOREAN = "ko"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, code: str) -> "LanguageCode":
        """Create language code from string with validation."""
        normalized = code.lower().strip()
        
        for lang_code in cls:
            if lang_code.value == normalized:
                return lang_code
        
        if len(normalized) == 2 and normalized.isalpha():
            return cls.UNKNOWN
        
        raise ValueError(f"Invalid language code: {code}")
    
    def is_cjk(self) -> bool:
        """Check if this is a CJK (Chinese, Japanese, Korean) language."""
        return self in (self.JAPANESE, self.CHINESE, self.KOREAN)
    
    def requires_special_tokenization(self) -> bool:
        """Check if language requires special tokenization."""
        return self.is_cjk()
