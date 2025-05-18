"""Tests for the content extraction functionality."""

import tempfile
from pathlib import Path

import pytest

from oboyu.crawler.extractor import extract_content, _detect_language, _extract_by_type


class TestExtractor:
    """Test cases for content extraction."""
    
    def test_extract_content_text_file(self) -> None:
        """Test content extraction from a text file."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            # Write test content
            test_content = "This is a test file.\nIt has multiple lines."
            Path(temp_file.name).write_text(test_content)
            
            # Extract content
            content, language = extract_content(Path(temp_file.name))
            
            # Check results
            assert content == test_content
            assert language == "en"
    
    def test_extract_content_with_japanese(self) -> None:
        """Test content extraction with Japanese text."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            # Write test content with Japanese
            test_content = "これはテストファイルです。\n日本語のコンテンツを含んでいます。"
            Path(temp_file.name).write_text(test_content)
            
            # Extract content
            content, language = extract_content(Path(temp_file.name))
            
            # Check results
            assert content == test_content
            assert language == "ja"
    
    def test_extract_by_type(self) -> None:
        """Test extract_by_type function properly extracts text content."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            test_content = "This is a test file for extraction by type."
            Path(temp_file.name).write_text(test_content)
            
            # Test with various file types
            content = _extract_by_type(Path(temp_file.name), "text/plain")
            assert content == test_content
            
            # Non-text types should still be extracted as text in Oboyu
            content = _extract_by_type(Path(temp_file.name), "application/pdf")
            assert content == test_content
    
    def test_extract_content_missing_file(self) -> None:
        """Test extraction with a non-existent file."""
        non_existent_file = Path("/non/existent/file.txt")
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            extract_content(non_existent_file)


class TestLanguageDetection:
    """Test cases for language detection."""
    
    def test_detect_language_english(self) -> None:
        """Test language detection with English text."""
        text = "This is an English text without any Japanese characters."
        assert _detect_language(text) == "en"
    
    def test_detect_language_japanese(self) -> None:
        """Test language detection with Japanese text."""
        text = "これは日本語のテキストです。"
        assert _detect_language(text) == "ja"
    
    def test_detect_language_mixed(self) -> None:
        """Test language detection with mixed text."""
        # Text with some Japanese characters but mostly English
        text = "This text contains some Japanese like こんにちは but is mostly English."
        # Should still detect as English since Japanese is less than 5%
        assert _detect_language(text) == "en"
        
        # Text with more Japanese content
        text = "This text contains a lot of Japanese like これは日本語のテキストです。今日はいい天気ですね。"
        # Should detect as Japanese since it's more than 5%
        assert _detect_language(text) == "ja"