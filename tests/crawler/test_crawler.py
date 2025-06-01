"""Tests for the main Crawler class."""

import tempfile
from pathlib import Path

import pytest

from oboyu.crawler.crawler import Crawler, CrawlerResult
from oboyu.crawler.services import (
    ContentExtractor,
    EncodingDetector,
    FileDiscoveryService,
    LanguageDetector,
    MetadataExtractor,
)


class TestCrawler:
    """Test cases for the Crawler class."""
    
    def test_crawler_initialization(self) -> None:
        """Test Crawler initialization with default and custom parameters."""
        # Test with default parameters
        crawler = Crawler()
        assert crawler.depth == 10
        assert "*.txt" in crawler.include_patterns
        assert "*/node_modules/*" in crawler.exclude_patterns
        assert crawler.max_workers == 4  # Default worker count
        
        # Check that services are initialized
        assert isinstance(crawler.discovery_service, FileDiscoveryService)
        assert isinstance(crawler.content_extractor, ContentExtractor)
        assert isinstance(crawler.language_detector, LanguageDetector)
        assert isinstance(crawler.encoding_detector, EncodingDetector)
        assert isinstance(crawler.metadata_extractor, MetadataExtractor)
        
        # Test with custom parameters
        crawler = Crawler(
            depth=5,
            include_patterns=["*.csv"],
            exclude_patterns=["*/temp/*"],
            max_workers=8,
        )
        assert crawler.depth == 5
        assert crawler.include_patterns == ["*.csv"]
        assert crawler.exclude_patterns == ["*/temp/*"]
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

    def test_crawler_with_custom_services(self) -> None:
        """Test Crawler initialization with custom services."""
        # Create custom services
        custom_discovery = FileDiscoveryService(max_file_size=5 * 1024 * 1024)
        custom_extractor = ContentExtractor(max_file_size=5 * 1024 * 1024)
        custom_language_detector = LanguageDetector()
        custom_encoding_detector = EncodingDetector()
        custom_metadata_extractor = MetadataExtractor(follow_symlinks=True)

        # Create crawler with custom services
        crawler = Crawler(
            discovery_service=custom_discovery,
            content_extractor=custom_extractor,
            language_detector=custom_language_detector,
            encoding_detector=custom_encoding_detector,
            metadata_extractor=custom_metadata_extractor,
        )

        # Check that custom services are used
        assert crawler.discovery_service is custom_discovery
        assert crawler.content_extractor is custom_extractor
        assert crawler.language_detector is custom_language_detector
        assert crawler.encoding_detector is custom_encoding_detector
        assert crawler.metadata_extractor is custom_metadata_extractor

        # Check custom configurations are preserved
        assert crawler.discovery_service.max_file_size == 5 * 1024 * 1024
        assert crawler.content_extractor.max_file_size == 5 * 1024 * 1024
        assert crawler.metadata_extractor.follow_symlinks is True

    def test_crawler_service_composition(self) -> None:
        """Test that the Crawler properly orchestrates its services."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create test file with front matter
            test_file = test_dir / "test.md"
            test_content = '''---
title: Service Composition Test
---

# Service Composition Test

This tests the orchestration of services.
'''
            test_file.write_text(test_content)

            # Create crawler and crawl
            crawler = Crawler(include_patterns=["*.md"])
            results = crawler.crawl(test_dir)

            # Check that services worked together correctly
            assert len(results) == 1
            result = results[0]

            # Check that content was extracted (without front matter)
            assert "# Service Composition Test" in result.content
            assert "---" not in result.content

            # Check that title was extracted from metadata
            assert result.title == "Service Composition Test"

            # Check that language was detected
            assert result.language == "en"

            # Check that metadata includes both extracted and file metadata
            assert "file_size" in result.metadata  # From file metadata
            assert "title" in result.metadata     # From front matter