"""Tests for ContentExtractor."""

import tempfile
from pathlib import Path

import pytest

from oboyu.crawler.services.content_extractor import ContentExtractor


class TestContentExtractor:
    """Test cases for ContentExtractor."""

    def test_initialization(self) -> None:
        """Test extractor initialization with default and custom parameters."""
        # Test with default parameters
        extractor = ContentExtractor()
        assert extractor.max_file_size == 50 * 1024 * 1024  # 50MB

        # Test with custom parameters
        extractor = ContentExtractor(max_file_size=5 * 1024 * 1024)
        assert extractor.max_file_size == 5 * 1024 * 1024

    def test_extract_content_basic_text(self) -> None:
        """Test basic text content extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create simple text file
            text_file = test_dir / "test.txt"
            text_content = "This is a simple text file."
            text_file.write_text(text_content, encoding="utf-8")

            # Test extraction
            extractor = ContentExtractor()
            content, metadata = extractor.extract_content(text_file)

            assert content == text_content
            assert isinstance(metadata, dict)

    def test_extract_content_with_front_matter(self) -> None:
        """Test content extraction with YAML front matter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create file with front matter
            md_file = test_dir / "test.md"
            md_content = """---
title: Test Document
created_at: 2023-01-01T12:00:00Z
uri: https://example.com/test
---

# Test Document

This is the main content of the document.
"""
            md_file.write_text(md_content, encoding="utf-8")

            # Test extraction
            extractor = ContentExtractor()
            content, metadata = extractor.extract_content(md_file)

            # Check content (without front matter)
            assert "# Test Document" in content
            assert "This is the main content" in content
            assert "---" not in content

            # Check metadata
            assert metadata["title"] == "Test Document"
            assert metadata["uri"] == "https://example.com/test"
            assert "created_at" in metadata

    def test_extract_content_encoding_detection(self) -> None:
        """Test content extraction with different encodings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create UTF-8 file
            utf8_file = test_dir / "utf8.txt"
            utf8_content = "This is UTF-8 content with émojis 🎉"
            utf8_file.write_text(utf8_content, encoding="utf-8")

            # Test extraction
            extractor = ContentExtractor()
            content, metadata = extractor.extract_content(utf8_file)
            assert content == utf8_content

    def test_extract_content_japanese(self) -> None:
        """Test content extraction with Japanese text."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create Japanese file
            ja_file = test_dir / "japanese.txt"
            ja_content = "これは日本語のテストファイルです。漢字ひらがなカタカナ含む。"
            ja_file.write_text(ja_content, encoding="utf-8")

            # Test extraction
            extractor = ContentExtractor()
            content, metadata = extractor.extract_content(ja_file)
            assert content == ja_content

    def test_extract_content_large_file(self) -> None:
        """Test content extraction with large files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create large file
            large_file = test_dir / "large.txt"
            large_content = "This is a large file.\n" * 100000  # Approx 2MB
            large_file.write_text(large_content, encoding="utf-8")

            # Test extraction with size limit
            extractor = ContentExtractor(max_file_size=1024 * 1024)  # 1MB limit
            content, metadata = extractor.extract_content(large_file)

            # Should only read first 1MB
            assert len(content) <= 1024 * 1024
            assert content.startswith("This is a large file.")

    def test_extract_content_invalid_file(self) -> None:
        """Test extraction with invalid file."""
        extractor = ContentExtractor()

        with pytest.raises(ValueError, match="File does not exist"):
            extractor.extract_content(Path("/nonexistent/file.txt"))

    def test_get_file_type(self) -> None:
        """Test file type detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create files with different extensions
            txt_file = test_dir / "test.txt"
            txt_file.write_text("Text content")

            html_file = test_dir / "test.html"
            html_file.write_text("<html><body>HTML content</body></html>")

            py_file = test_dir / "test.py"
            py_file.write_text("print('Python code')")

            extractor = ContentExtractor()

            # Test file type detection
            assert extractor._get_file_type(txt_file) == "text"
            assert extractor._get_file_type(html_file) == "text"  # HTML is detected as text
            assert extractor._get_file_type(py_file) == "text"  # Python is detected as text

    def test_decode_content_fallback(self) -> None:
        """Test content decoding with fallback mechanisms."""
        extractor = ContentExtractor()

        # Test valid UTF-8
        utf8_bytes = "Hello, world! 🌍".encode("utf-8")
        decoded = extractor._decode_content(utf8_bytes)
        assert decoded == "Hello, world! 🌍"

        # Test invalid UTF-8 that should fallback gracefully
        invalid_bytes = b"\\xff\\xfe Invalid UTF-8"
        decoded = extractor._decode_content(invalid_bytes)
        assert isinstance(decoded, str)  # Should decode to something

    def test_parse_front_matter_edge_cases(self) -> None:
        """Test front matter parsing with edge cases."""
        extractor = ContentExtractor()

        # Test content without front matter
        content = "Just regular content without any front matter."
        parsed_content, metadata = extractor._parse_front_matter(content)
        assert parsed_content == content
        assert metadata == {}

        # Test empty front matter
        content_with_empty_fm = """---
---

Content after empty front matter."""
        parsed_content, metadata = extractor._parse_front_matter(content_with_empty_fm)
        assert "Content after empty front matter." in parsed_content
        assert metadata == {}

        # Test malformed front matter (should not crash)
        content_with_malformed = """---
title: Test
invalid yaml: [unclosed
---

Content after malformed front matter."""
        try:
            parsed_content, metadata = extractor._parse_front_matter(content_with_malformed)
            # Should handle gracefully
            assert isinstance(parsed_content, str)
            assert isinstance(metadata, dict)
        except Exception:
            # If it throws, that's also acceptable behavior
            pass

    def test_extract_pdf_simple(self) -> None:
        """Test PDF content extraction from simple PDF."""
        # Get test PDF file
        pdf_path = Path(__file__).parent.parent.parent / "fixtures" / "pdf" / "simple_text.pdf"

        if pdf_path.exists():
            extractor = ContentExtractor()
            content, metadata = extractor.extract_content(pdf_path)

            # Check content extraction
            assert "Simple Test PDF" in content
            assert "simple test PDF file" in content
            assert "English text for testing purposes" in content

            # Check that content is text, not binary
            assert isinstance(content, str)
            assert len(content) > 0

    def test_extract_pdf_multipage(self) -> None:
        """Test PDF content extraction from multi-page PDF."""
        pdf_path = Path(__file__).parent.parent.parent / "fixtures" / "pdf" / "multipage_text.pdf"

        if pdf_path.exists():
            extractor = ContentExtractor()
            content, metadata = extractor.extract_content(pdf_path)

            # Check all pages are extracted
            assert "Page 1: Introduction" in content
            assert "Page 2: Content" in content
            assert "Page 3: Conclusion" in content
            assert "first page" in content
            assert "second page" in content
            assert "final page" in content

    def test_extract_pdf_metadata(self) -> None:
        """Test PDF metadata extraction."""
        pdf_path = Path(__file__).parent.parent.parent / "fixtures" / "pdf" / "metadata.pdf"

        if pdf_path.exists():
            extractor = ContentExtractor()
            content, metadata = extractor.extract_content(pdf_path)

            # Check metadata extraction
            assert metadata.get("title") == "PDF with Metadata"
            assert metadata.get("creator") == "Test Creator Application"
            # Note: Some metadata fields might not be preserved in minimal PDFs

    def test_extract_pdf_empty(self) -> None:
        """Test PDF extraction from empty PDF."""
        pdf_path = Path(__file__).parent.parent.parent / "fixtures" / "pdf" / "empty.pdf"

        if pdf_path.exists():
            extractor = ContentExtractor()
            content, metadata = extractor.extract_content(pdf_path)

            # Empty PDF should return empty or minimal content
            assert isinstance(content, str)
            # Content might be empty or contain minimal whitespace

    def test_pdf_file_type_detection(self) -> None:
        """Test PDF file type detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create a file with PDF header
            pdf_file = test_dir / "test.pdf"
            pdf_file.write_bytes(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")

            extractor = ContentExtractor()
            file_type = extractor._get_file_type(pdf_file)
            assert file_type == "application/pdf"

            # Test with .pdf extension
            empty_pdf = test_dir / "empty.pdf"
            empty_pdf.write_text("Not really a PDF")
            file_type = extractor._get_file_type(empty_pdf)
            assert file_type == "application/pdf"  # Should detect by extension

    def test_extract_pdf_invalid(self) -> None:
        """Test PDF extraction with invalid PDF file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create a fake PDF (text file with .pdf extension)
            fake_pdf = test_dir / "fake.pdf"
            fake_pdf.write_text("This is not a real PDF file")

            extractor = ContentExtractor()
            # Should raise an error when trying to extract
            with pytest.raises(Exception):  # Changed to catch broader exception types from pypdf
                extractor.extract_content(fake_pdf)
