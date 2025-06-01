"""Tests for the content extraction functionality."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from oboyu.crawler.extractor import extract_content, _detect_language, _extract_by_type, _parse_front_matter


class TestExtractor:
    """Test cases for content extraction."""
    
    def test_extract_content_text_file(self) -> None:
        """Test content extraction from a text file."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            # Write test content
            test_content = "This is a test file.\nIt has multiple lines."
            Path(temp_file.name).write_text(test_content)
            
            # Extract content
            content, language, metadata = extract_content(Path(temp_file.name))
            
            # Check results
            assert content == test_content
            assert language == "en"
            assert metadata == {}  # No metadata for plain text
    
    def test_extract_content_with_japanese(self) -> None:
        """Test content extraction with Japanese text."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            # Write test content with Japanese
            test_content = "これはテストファイルです。\n日本語のコンテンツを含んでいます。"
            Path(temp_file.name).write_text(test_content)
            
            # Extract content
            content, language, metadata = extract_content(Path(temp_file.name))
            
            # Check results
            assert content == test_content
            assert language == "ja"
            assert metadata == {}  # No metadata for plain text
    
    def test_extract_by_type(self) -> None:
        """Test extract_by_type function properly extracts text content."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            test_content = "This is a test file for extraction by type."
            Path(temp_file.name).write_text(test_content)
            
            # Test with various file types
            content, metadata = _extract_by_type(Path(temp_file.name), "text/plain")
            assert content == test_content
            assert metadata == {}
            
            # Non-text types should still be extracted as text in Oboyu
            content, metadata = _extract_by_type(Path(temp_file.name), "application/pdf")
            assert content == test_content
            assert metadata == {}
    
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
        # FastText detects this as Japanese due to the presence of Japanese characters
        assert _detect_language(text) == "ja"
        
        # Text with more Japanese content
        text = "This text contains a lot of Japanese like これは日本語のテキストです。今日はいい天気ですね。"
        # Should detect as Japanese since it's more than 5%
        assert _detect_language(text) == "ja"


class TestYamlFrontMatter:
    """Test cases for YAML front matter parsing."""
    
    def test_parse_front_matter_basic(self) -> None:
        """Test parsing basic YAML front matter."""
        content = """---
title: Test Document
author: Test Author
created_at: 2025-05-24T10:00:00Z
updated_at: 2025-05-24T12:00:00Z
uri: https://example.com/test
---
# Main Content

This is the actual content of the document."""
        
        result_content, metadata = _parse_front_matter(content)
        expected_content = """# Main Content

This is the actual content of the document."""
        assert result_content == expected_content
        assert metadata['title'] == 'Test Document'
        assert 'created_at' in metadata
        assert 'updated_at' in metadata
        assert metadata['uri'] == 'https://example.com/test'
    
    def test_parse_front_matter_with_only_title(self) -> None:
        """Test parsing YAML front matter with only title."""
        content = """---
title: Test Document
---
Content here"""
        
        result_content, metadata = _parse_front_matter(content)
        assert result_content == "Content here"
        assert metadata == {'title': 'Test Document'}
    
    def test_no_yaml_front_matter(self) -> None:
        """Test content without YAML front matter remains unchanged."""
        content = """# Regular Markdown

This document has no front matter."""
        
        result_content, metadata = _parse_front_matter(content)
        assert result_content == content
        assert metadata == {}
    
    def test_yaml_front_matter_only_at_start(self) -> None:
        """Test that only YAML front matter at the start is parsed."""
        content = """---
title: Test
---
Some content

---
This should not be removed
---
More content"""
        
        result_content, metadata = _parse_front_matter(content)
        expected = """Some content

---
This should not be removed
---
More content"""
        assert result_content == expected
        assert metadata == {'title': 'Test'}
    
    def test_extract_content_with_yaml_front_matter(self) -> None:
        """Test full extraction pipeline with YAML front matter."""
        with tempfile.NamedTemporaryFile(suffix=".md", mode='w', encoding='utf-8', delete=False) as temp_file:
            # Write test content with YAML front matter
            test_content = """---
title: 日本語のドキュメント
created_at: 2025-05-24T10:00:00Z
updated_at: 2025-05-24T12:00:00Z
uri: https://example.com/japanese
author: テスト著者
tags:
  - japanese
  - test
---
# 日本語のテスト

これはYAMLフロントマターを含むテストファイルです。"""
            
            temp_file.write(test_content)
            temp_file.flush()
            
            # Extract content
            content, language, metadata = extract_content(Path(temp_file.name))
            
            # Check results - should have front matter removed
            expected_content = """# 日本語のテスト

これはYAMLフロントマターを含むテストファイルです。"""
            assert content == expected_content
            assert language == "ja"
            assert metadata['title'] == '日本語のドキュメント'
            assert 'created_at' in metadata
            assert 'updated_at' in metadata
            assert metadata['uri'] == 'https://example.com/japanese'
            
            # Clean up
            Path(temp_file.name).unlink()
    
    def test_empty_yaml_front_matter(self) -> None:
        """Test parsing empty YAML front matter."""
        content = """---
---
# Content starts here"""
        
        result_content, metadata = _parse_front_matter(content)
        assert result_content == "# Content starts here"
        assert metadata == {}
    
    def test_parse_front_matter_datetime_types(self) -> None:
        """Test parsing different datetime formats in front matter."""
        content = """---
title: Test with Dates
created_at: 2025-05-24T10:00:00Z
updated_at: 2025-05-24T12:00:00+09:00
---
Content here"""
        
        result_content, metadata = _parse_front_matter(content)
        assert result_content == "Content here"
        assert metadata['title'] == 'Test with Dates'
        assert isinstance(metadata['created_at'], datetime)
        assert isinstance(metadata['updated_at'], datetime)