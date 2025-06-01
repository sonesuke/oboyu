"""Language detection service for identifying document languages."""

import urllib.request
from typing import Optional

import fasttext

# Global FastText model instance
_fasttext_model: Optional[fasttext.FastText._FastText] = None


class LanguageDetector:
    """Service responsible for detecting the language of document content."""

    def __init__(self) -> None:
        """Initialize the language detector."""
        pass

    def detect_language(self, content: str) -> str:
        """Detect the language of the text using FastText.
        
        Args:
            content: Text content
            
        Returns:
            ISO 639-1 language code

        """
        # Trim text to just a reasonable sample for faster processing
        sample = content[:5000].replace('\n', ' ').strip()
        
        # If sample is too short, use fallback logic
        if len(sample) < 10:
            return "en"

        # Count Japanese characters as a quick pre-check
        japanese_char_count = sum(1 for char in sample if 0x3000 <= ord(char) <= 0x9FFF)

        # If we have a significant number of Japanese characters, it's likely Japanese
        if japanese_char_count > len(sample) * 0.1:
            return "ja"

        # Use FastText for language detection
        try:
            model = self._get_fasttext_model()
            predictions = model.predict(sample, k=1)
            
            if predictions and len(predictions) == 2 and len(predictions[0]) > 0:
                # Extract language code from __label__xx format
                language_label = predictions[0][0]
                confidence = float(predictions[1][0])
                
                # Remove __label__ prefix
                if language_label.startswith('__label__'):
                    detected = language_label[9:]  # Remove '__label__' prefix
                else:
                    detected = language_label
                
                # Use detection if confidence is high enough
                if confidence >= 0.5:
                    # Handle Japanese detection
                    if detected == "ja":
                        return "ja"
                    
                    # Handle common languages
                    if detected in ["en", "zh", "ko", "fr", "de", "es", "it", "ru"]:
                        return str(detected)
                
                # For other languages or low confidence, check again for Japanese characters
                if japanese_char_count > 0:
                    return "ja"
                
                # Return detected language even with low confidence for common languages
                if detected in ["en", "zh", "ko", "fr", "de", "es", "it", "ru"]:
                    return str(detected)
                
                return str(detected)
            
        except Exception:  # noqa: S110
            # If FastText fails, fall back to simpler detection
            # This is expected behavior for graceful fallback
            pass
        
        # Fallback detection
        if japanese_char_count > 0:
            return "ja"

        # Default to English if we can't determine
        return "en"

    def _get_fasttext_model(self) -> fasttext.FastText._FastText:
        """Get or load the FastText language identification model.
        
        Returns:
            FastText model instance
            
        Raises:
            RuntimeError: If model cannot be loaded or downloaded

        """
        global _fasttext_model
        
        if _fasttext_model is not None:
            return _fasttext_model
        
        # Try to load from cache first
        from oboyu.common.paths import CACHE_BASE_DIR
        model_path = CACHE_BASE_DIR / "fasttext" / "lid.176.bin"
        
        if model_path.exists():
            try:
                _fasttext_model = fasttext.load_model(str(model_path))
                return _fasttext_model
            except Exception:
                # If cached model is corrupted, remove it and re-download
                model_path.unlink(missing_ok=True)
        
        # Download model if not cached
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        
        try:
            # Use HTTPS URL which is secure for downloading the official FastText model
            urllib.request.urlretrieve(model_url, str(model_path))  # noqa: S310
            _fasttext_model = fasttext.load_model(str(model_path))
            return _fasttext_model
        except Exception as e:
            raise RuntimeError(f"Failed to download or load FastText model: {e}") from e
