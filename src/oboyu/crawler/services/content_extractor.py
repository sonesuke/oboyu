"""Content extraction service for handling content extraction from different file formats."""

import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    pass

import chardet
import charset_normalizer
import frontmatter

from .optimized_pdf_processor import OptimizedPDFProcessor

# Initialize mimetypes
mimetypes.init()


class ContentExtractor:
    """Service responsible for extracting content from different file formats."""

    def __init__(self, max_file_size: int = 50 * 1024 * 1024) -> None:
        """Initialize the content extractor.
        
        Args:
            max_file_size: Maximum file size in bytes to process (default: 50MB)

        """
        self.max_file_size = max_file_size
        self._pdf_processor = OptimizedPDFProcessor(max_file_size=max_file_size)

    def extract_content(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract content from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (content, metadata)
            Where metadata contains optional fields: title, created_at, updated_at, uri

        """
        # Ensure the file exists
        if not file_path.exists() or not file_path.is_file():
            raise ValueError(f"File does not exist or is not a file: {file_path}")

        # Get file type
        file_type = self._get_file_type(file_path)

        # Extract content and metadata based on file type
        content, metadata = self._extract_by_type(file_path, file_type)

        return content, metadata

    def _get_file_type(self, file_path: Path) -> str:
        """Determine the file type based on extension and content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type string

        """
        # First check by extension
        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type:
            # Return full mime type for PDF files, main type for others
            if mime_type == "application/pdf":
                return "application/pdf"
            else:
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
            pass

        # Default to text/plain if we can't determine
        return "text/plain"

    def _extract_by_type(self, file_path: Path, file_type: str) -> Tuple[str, Dict[str, Any]]:
        """Extract content based on file type.
        
        Args:
            file_path: Path to the file
            file_type: File type string
            
        Returns:
            Tuple of (content, metadata)

        """
        # Check if it's a PDF file
        if file_type == "application/pdf" or file_type == "application" and file_path.suffix.lower() == ".pdf":
            return self._extract_pdf_file(file_path)
        
        # Default to text file extraction
        return self._extract_text_file(file_path)

    def _extract_text_file(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract content and metadata from a text file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (content, metadata)

        """
        # Read file with size limit for efficiency
        file_size = file_path.stat().st_size

        if file_size > self.max_file_size:
            # For large files, read only the beginning
            with open(file_path, "rb") as f:
                raw_data = f.read(self.max_file_size)
        else:
            # First read the file as binary
            with open(file_path, "rb") as f:
                raw_data = f.read()  # Read the entire file

        # Decode the content
        content = self._decode_content(raw_data)

        # Parse YAML front matter if present
        content, metadata = self._parse_front_matter(content)

        return content, metadata

    def _decode_content(self, raw_data: bytes) -> str:
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

    def _parse_front_matter(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """Parse YAML front matter from content if present.
        
        Extracts YAML front matter and returns the content without it,
        along with the parsed metadata.
        
        Args:
            content: File content that may contain YAML front matter
            
        Returns:
            Tuple of (content without front matter, metadata dict)
            Metadata may contain: title, created_at, updated_at, uri

        """
        from datetime import datetime

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

    def _extract_pdf_file(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract content and metadata from a PDF file using optimized processor.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (content, metadata)

        """
        return self._pdf_processor.extract_pdf(file_path)

