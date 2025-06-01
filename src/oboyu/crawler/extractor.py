"""Content extraction module for Oboyu.

This module provides utilities for extracting content from various file types.
"""

import mimetypes
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import chardet
import charset_normalizer
import fasttext
import frontmatter

# Initialize mimetypes
mimetypes.init()

# Global FastText model instance
_fasttext_model: Optional[fasttext.FastText._FastText] = None


def extract_content(file_path: Path) -> Tuple[str, str, Dict[str, Any]]:
    """Extract content from a file and detect its language.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (content, language_code, metadata)
        Where metadata contains optional fields: title, created_at, updated_at, uri

    """
    # Ensure the file exists
    if not file_path.exists() or not file_path.is_file():
        raise ValueError(f"File does not exist or is not a file: {file_path}")

    # Get file type
    file_type = _get_file_type(file_path)

    # Extract content and metadata based on file type
    content, metadata = _extract_by_type(file_path, file_type)

    # Detect language
    language = _detect_language(content)

    return content, language, metadata


def _get_file_type(file_path: Path) -> str:
    """Determine the file type based on extension and content.

    Args:
        file_path: Path to the file

    Returns:
        File type string

    """
    # First check by extension
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type:
        # Return the main type
        return mime_type.split("/")[0]

    # If we can't determine by extension, check file header (first few bytes)
    try:
        with open(file_path, "rb") as f:
            header = f.read(512)  # Read first 512 bytes

            # Check for common file signatures
            if header.startswith(b"%PDF-"):
                return "application/pdf"
            if header.startswith(b"\x25\x21"):
                return "text/plain"  # Likely a script
            if b"<!DOCTYPE html>" in header or b"<html" in header:
                return "text/html"
            if b"<?xml" in header:
                return "text/xml"
    except IOError:
        # If we can't read the file, just continue to the default return
        # This is expected for some files, so we don't need to log an error
        pass

    # Default to text/plain if we can't determine
    return "text/plain"


def _extract_by_type(file_path: Path, file_type: str) -> Tuple[str, Dict[str, Any]]:
    """Extract content based on file type.

    Args:
        file_path: Path to the file
        file_type: File type string

    Returns:
        Tuple of (content, metadata)

    """
    # Oboyu only supports text files
    return _extract_text_file(file_path)


def _extract_text_file(file_path: Path) -> Tuple[str, Dict[str, Any]]:
    """Extract content and metadata from a text file.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (content, metadata)

    """
    # Read file with size limit for efficiency
    max_size = 10 * 1024 * 1024  # 10MB limit per file
    file_size = file_path.stat().st_size

    if file_size > max_size:
        # For large files, read only the beginning
        with open(file_path, "rb") as f:
            raw_data = f.read(max_size)
    else:
        # First read the file as binary
        with open(file_path, "rb") as f:
            raw_data = f.read()  # Read the entire file

    # Decode the content
    content = _decode_content(raw_data)

    # Parse YAML front matter if present
    content, metadata = _parse_front_matter(content)

    return content, metadata


def _decode_content(raw_data: bytes) -> str:
    """Decode raw bytes to string using various encoding detection methods.

    Args:
        raw_data: Raw bytes to decode

    Returns:
        Decoded string

    """
    # Try common encodings first (much faster for Japanese content)
    common_encodings = ["utf-8", "shift_jis", "euc-jp", "iso-2022-jp"]
    for encoding in common_encodings:
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            continue

    # For small files, use only a sample for detection (faster)
    sample_size = min(len(raw_data), 32768)  # 32KB sample for faster detection
    sample_data = raw_data[:sample_size] if len(raw_data) > sample_size else raw_data

    # Try charset-normalizer as fallback (more accurate but slower)
    charset_results = charset_normalizer.from_bytes(sample_data).best()
    if charset_results:
        encoding_name = charset_results.encoding
        if encoding_name:
            try:
                return raw_data.decode(encoding_name)
            except UnicodeDecodeError:
                pass  # Fall through to other methods

    # If charset-normalizer didn't work well, try chardet
    chardet_result = chardet.detect(raw_data)
    chardet_encoding: str = chardet_result.get("encoding", "utf-8") or "utf-8"
    confidence = chardet_result.get("confidence", 0.0)

    # If encoding was detected with high confidence, use it
    if chardet_encoding and confidence > 0.7:
        try:
            return raw_data.decode(chardet_encoding)
        except UnicodeDecodeError:
            pass  # Fall through to other methods

    # Try specific encodings for Japanese content
    # These are common in Japanese text files
    for encoding in ["utf-8", "shift-jis", "euc-jp", "cp932", "iso-2022-jp"]:
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            continue

    # Try common western encodings
    for encoding in ["utf-8-sig", "latin-1", "windows-1252"]:
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            continue

    # Last resort: use UTF-8 with replacement for invalid characters
    return raw_data.decode("utf-8", errors="replace")


def _parse_front_matter(content: str) -> Tuple[str, Dict[str, Any]]:
    """Parse YAML front matter from content if present.

    Extracts YAML front matter and returns the content without it,
    along with the parsed metadata.

    Args:
        content: File content that may contain YAML front matter

    Returns:
        Tuple of (content without front matter, metadata dict)
        Metadata may contain: title, created_at, updated_at, uri

    """
    # Use python-frontmatter to parse
    post = frontmatter.loads(content)

    # Extract supported metadata fields
    metadata: Dict[str, Any] = {}

    # Extract title
    if "title" in post.metadata:
        metadata["title"] = str(post.metadata["title"])

    # Extract dates - convert to datetime if string
    for date_field in ["created_at", "updated_at"]:
        if date_field in post.metadata:
            value = post.metadata[date_field]
            if isinstance(value, str):
                # Try to parse ISO format datetime
                try:
                    metadata[date_field] = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    # If parsing fails, store as string
                    metadata[date_field] = value
            elif isinstance(value, datetime):
                metadata[date_field] = value
            else:
                # Store other types as-is
                metadata[date_field] = value

    # Extract URI
    if "uri" in post.metadata:
        metadata["uri"] = str(post.metadata["uri"])

    # Return content without front matter and the metadata
    return post.content, metadata


def _get_fasttext_model() -> fasttext.FastText._FastText:
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


def _detect_language(text: str) -> str:
    """Detect the language of the text using FastText.

    Args:
        text: Text content

    Returns:
        ISO 639-1 language code

    """
    # Trim text to just a reasonable sample for faster processing
    sample = text[:5000].replace('\n', ' ').strip()
    
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
        model = _get_fasttext_model()
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
