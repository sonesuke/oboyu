"""Tests for the main Crawler class."""

import tempfile
from pathlib import Path

import pytest

from oboyu.crawler.crawler import Crawler, CrawlerResult


class TestCrawler:
    """Test cases for the Crawler class."""
    
    def test_crawler_initialization(self) -> None:
        """Test Crawler initialization with default and custom parameters."""
        # Test with default parameters
        crawler = Crawler()
        assert crawler.depth == 10
        assert "*.txt" in crawler.include_patterns
        assert "*/node_modules/*" in crawler.exclude_patterns
        assert crawler.max_file_size == 10 * 1024 * 1024  # 10MB
        assert not crawler.follow_symlinks
        assert "utf-8" in crawler.japanese_encodings
        assert crawler.max_workers == 4  # Default worker count
        
        # Test with custom parameters
        crawler = Crawler(
            depth=5,
            include_patterns=["*.csv"],
            exclude_patterns=["*/temp/*"],
            max_file_size=1024,
            follow_symlinks=True,
            japanese_encodings=["utf-8"],
            max_workers=8,
        )
        assert crawler.depth == 5
        assert crawler.include_patterns == ["*.csv"]
        assert crawler.exclude_patterns == ["*/temp/*"]
        assert crawler.max_file_size == 1024
        assert crawler.follow_symlinks
        assert crawler.japanese_encodings == ["utf-8"]
        assert crawler.max_workers == 8
    
    def test_crawler_crawl(self) -> None:
        """Test crawling a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_dir = Path(temp_dir)
            
            # Create text file
            text_file = test_dir / "test.txt"
            text_file.write_text("This is a test file")
            
            # Create markdown file
            md_file = test_dir / "test.md"
            md_file.write_text("# Test Markdown")
            
            # Create Japanese file
            ja_file = test_dir / "japanese.txt"
            ja_file.write_text("これは日本語のテストファイルです。")
            
            # Create excluded file
            excluded_dir = test_dir / "node_modules"
            excluded_dir.mkdir()
            excluded_file = excluded_dir / "excluded.txt"
            excluded_file.write_text("This should be excluded")
            
            # Create crawler and crawl directory
            crawler = Crawler(
                include_patterns=["*.txt", "*.md"],
                exclude_patterns=["*/node_modules/*"],
            )
            results = crawler.crawl(test_dir)
            
            # Check results
            assert len(results) == 3  # Should find 3 files
            
            # Check paths
            paths = [str(result.path) for result in results]
            assert str(text_file) in paths
            assert str(md_file) in paths
            assert str(ja_file) in paths
            assert str(excluded_file) not in paths
            
            # Check language detection
            for result in results:
                if "japanese" in str(result.path):
                    assert result.language == "ja"
                else:
                    assert result.language == "en"
    
    def test_generate_title(self) -> None:
        """Test title generation from content and filename."""
        crawler = Crawler()
        
        # Test title from metadata
        path = Path("test.txt")
        content = "Title Line\nSecond line\nThird line"
        metadata = {"title": "Metadata Title"}
        title = crawler._generate_title(path, content, metadata)
        assert title == "Metadata Title"
        
        # Test title from content when no metadata
        path = Path("test.txt")
        content = "Title Line\nSecond line\nThird line"
        metadata = {}
        title = crawler._generate_title(path, content, metadata)
        assert title == "Title Line"
        
        # Test title from filename when content doesn't have a good title
        path = Path("document-title.txt")
        content = "This is not a good title because it is too long and contains multiple sentences. It should not be used as a title."
        metadata = {}
        title = crawler._generate_title(path, content, metadata)
        assert title == "document-title"