"""Content extraction service for handling content extraction from different file formats."""

import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    import pypdf

import chardet
import charset_normalizer
import frontmatter

# Initialize mimetypes
mimetypes.init()


class ContentExtractor:
    """Service responsible for extracting content from different file formats."""

    def __init__(self, max_file_size: int = 10 * 1024 * 1024) -> None:
        """Initialize the content extractor.
        
        Args:
            max_file_size: Maximum file size in bytes to process (default: 10MB)

        """
        self.max_file_size = max_file_size

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
        """Extract content and metadata from a PDF file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (content, metadata)

        """
        try:
            import pypdf
        except ImportError:
            raise RuntimeError("pypdf library is required for PDF processing. Install with: uv add pypdf")
        
        # Check file size before processing
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise RuntimeError(f"PDF file too large ({file_size / (1024*1024):.1f}MB). Maximum supported size: {self.max_file_size / (1024*1024):.1f}MB")
        
        try:
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                
                # Check if PDF is encrypted and decrypt if needed
                self._handle_pdf_encryption(pdf_reader, file_path)
                
                # Extract text content from all pages
                content = self._extract_pdf_text_content(pdf_reader, file_path)
                
                # Extract PDF metadata
                metadata = self._extract_pdf_metadata(pdf_reader, len(content.split("\n\n")))
                
                return content, metadata
                
        except pypdf.errors.PdfReadError as e:
            self._handle_pdf_read_error(e, file_path)
            raise  # This should never be reached due to exception handling
        except Exception as e:
            self._handle_pdf_general_error(e, file_path)
            raise  # This should never be reached due to exception handling

    def _handle_pdf_encryption(self, pdf_reader: "pypdf.PdfReader", file_path: Path) -> None:
        """Handle PDF encryption and decryption."""
        if pdf_reader.is_encrypted:
            # Try to decrypt with empty password (common case)
            if not pdf_reader.decrypt(""):
                raise RuntimeError(f"PDF file is password-protected and cannot be processed: {file_path.name}")

    def _extract_pdf_text_content(self, pdf_reader: "pypdf.PdfReader", file_path: Path) -> str:
        """Extract text content from PDF pages with progress indication."""
        text_content = []
        total_pages = len(pdf_reader.pages)
        
        # Show progress for large PDFs (more than 10 pages)
        if total_pages > 10:
            print(f"Processing large PDF: {file_path.name} ({total_pages} pages)")
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_content.append(page_text)
                
                # Show progress for large PDFs
                if total_pages > 10 and (page_num + 1) % max(1, total_pages // 10) == 0:
                    progress = ((page_num + 1) / total_pages) * 100
                    print(f"  Progress: {progress:.0f}% ({page_num + 1}/{total_pages} pages)")
                    
            except Exception as page_error:
                # Continue processing other pages if one fails
                print(f"Warning: Failed to extract text from page {page_num + 1} of {file_path.name}: {page_error}")
                continue
        
        content = "\n\n".join(text_content)  # Use double newline for better page separation
        
        # Warn if no text was extracted
        if not content.strip():
            print(f"Warning: No text content extracted from PDF: {file_path.name} (may be image-only or corrupted)")
        
        return content

    def _extract_pdf_metadata(self, pdf_reader: "pypdf.PdfReader", extracted_pages: int) -> Dict[str, Any]:
        """Extract metadata from PDF."""
        metadata: Dict[str, Any] = {}
        metadata["total_pages"] = len(pdf_reader.pages)
        metadata["extracted_pages"] = extracted_pages
        
        if pdf_reader.metadata:
            # Extract title
            if pdf_reader.metadata.title:
                metadata["title"] = str(pdf_reader.metadata.title)
            
            # Extract creator
            if hasattr(pdf_reader.metadata, "creator") and pdf_reader.metadata.creator:
                metadata["creator"] = str(pdf_reader.metadata.creator)
            
            # Extract dates
            self._extract_pdf_dates(pdf_reader.metadata, metadata)
        
        return metadata

    def _extract_pdf_dates(self, pdf_metadata: "pypdf.DocumentInformation", metadata: Dict[str, Any]) -> None:
        """Extract creation and modification dates from PDF metadata."""
        from datetime import datetime
        
        # Extract creation date
        if hasattr(pdf_metadata, "creation_date") and pdf_metadata.creation_date:
            creation_date = pdf_metadata.creation_date
            if isinstance(creation_date, datetime):
                metadata["created_at"] = creation_date
            else:
                # Try to parse string date
                try:
                    metadata["created_at"] = datetime.fromisoformat(str(creation_date).replace("Z", "+00:00"))
                except ValueError:
                    # Store as string if parsing fails
                    metadata["created_at"] = str(creation_date)
        
        # Extract modification date
        if hasattr(pdf_metadata, "modification_date") and pdf_metadata.modification_date:
            mod_date = pdf_metadata.modification_date
            if isinstance(mod_date, datetime):
                metadata["updated_at"] = mod_date
            else:
                # Try to parse string date
                try:
                    metadata["updated_at"] = datetime.fromisoformat(str(mod_date).replace("Z", "+00:00"))
                except ValueError:
                    # Store as string if parsing fails
                    metadata["updated_at"] = str(mod_date)

    def _handle_pdf_read_error(self, error: Exception, file_path: Path) -> None:
        """Handle PDF read errors with specific error messages."""
        error_msg = str(error).lower()
        if "crypt" in error_msg or "encrypt" in error_msg or "aes" in error_msg:
            try:
                import cryptography  # noqa: F401
            except ImportError:
                raise RuntimeError("PDF file requires cryptography library for decryption. Install with: uv add cryptography")
            raise RuntimeError(f"Failed to decrypt PDF file (may be password-protected): {file_path.name}")
        else:
            raise RuntimeError(f"Failed to read PDF file '{file_path.name}': {error}")

    def _handle_pdf_general_error(self, error: Exception, file_path: Path) -> None:
        """Handle general PDF processing errors."""
        error_msg = str(error).lower()
        if "crypt" in error_msg or "aes" in error_msg:
            try:
                import cryptography  # noqa: F401
            except ImportError:
                raise RuntimeError("PDF file requires cryptography library for decryption. Install with: uv add cryptography")
        raise RuntimeError(f"Failed to extract PDF content from '{file_path.name}': {error}")
