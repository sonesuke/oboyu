"""Tests for the document discovery functionality."""

import os
import tempfile
from pathlib import Path

import pytest

from oboyu.crawler.discovery import discover_documents


class TestDiscovery:
    """Test cases for document discovery."""
    
    def test_discover_documents_basic(self) -> None:
        """Test basic document discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_dir = Path(temp_dir)
            
            # Create text file
            text_file = test_dir / "test.txt"
            text_file.write_text("This is a test file")
            
            # Create markdown file
            md_file = test_dir / "test.md"
            md_file.write_text("# Test Markdown")
            
            # Create excluded file
            node_modules_dir = test_dir / "node_modules"
            node_modules_dir.mkdir()
            excluded_file = node_modules_dir / "excluded.txt"
            excluded_file.write_text("This should be excluded")
            
            # Discover documents
            documents = discover_documents(
                directory=test_dir,
                patterns=["*.txt", "*.md"],
                exclude_patterns=["*/node_modules/*"],
                max_depth=5,
            )
            
            # Check results
            assert len(documents) == 2
            paths = [str(doc[0]) for doc in documents]
            assert str(text_file) in paths
            assert str(md_file) in paths
            assert str(excluded_file) not in paths
    
    def test_discover_documents_max_depth(self) -> None:
        """Test max depth in document discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test directory structure
            test_dir = Path(temp_dir)
            
            # Create files at different depths
            level0_file = test_dir / "level0.txt"
            level0_file.write_text("Level 0")
            
            level1_dir = test_dir / "level1"
            level1_dir.mkdir()
            level1_file = level1_dir / "level1.txt"
            level1_file.write_text("Level 1")
            
            level2_dir = level1_dir / "level2"
            level2_dir.mkdir()
            level2_file = level2_dir / "level2.txt"
            level2_file.write_text("Level 2")
            
            # Test with max_depth=1
            documents = discover_documents(
                directory=test_dir,
                patterns=["*.txt"],
                exclude_patterns=[],
                max_depth=1,
            )
            
            # Should only include level0 and level1 files
            assert len(documents) == 2
            paths = [str(doc[0]) for doc in documents]
            assert str(level0_file) in paths
            assert str(level1_file) in paths
            assert str(level2_file) not in paths
            
            # Test with max_depth=2
            documents = discover_documents(
                directory=test_dir,
                patterns=["*.txt"],
                exclude_patterns=[],
                max_depth=2,
            )
            
            # Should include all files
            assert len(documents) == 3
            paths = [str(doc[0]) for doc in documents]
            assert str(level0_file) in paths
            assert str(level1_file) in paths
            assert str(level2_file) in paths
    
    def test_discover_documents_file_size(self) -> None:
        """Test max file size in document discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_dir = Path(temp_dir)
            
            # Create small file
            small_file = test_dir / "small.txt"
            small_file.write_text("Small file")
            
            # Create large file (more than 100 bytes)
            large_file = test_dir / "large.txt"
            large_file.write_text("Large file" * 50)  # ~500 bytes
            
            # Test is no longer valid since max_file_size is hard-coded
            
            # With hard-coded max_file_size=10MB, both files should be included
            documents = discover_documents(
                directory=test_dir,
                patterns=["*.txt"],
                exclude_patterns=[],
                max_depth=5,
            )
            
            # Should include both files (since 10MB > 500 bytes)
            assert len(documents) == 2
            paths = [str(doc[0]) for doc in documents]
            assert str(small_file) in paths
            assert str(large_file) in paths