"""Tests for FileDiscoveryService."""

import tempfile
from pathlib import Path

import pytest

from oboyu.crawler.services.file_discovery_service import FileDiscoveryService


class TestFileDiscoveryService:
    """Test cases for FileDiscoveryService."""

    def test_initialization(self) -> None:
        """Test service initialization with default and custom parameters."""
        # Test with default parameters
        service = FileDiscoveryService()
        assert service.max_file_size == 10 * 1024 * 1024  # 10MB
        assert service.follow_symlinks is False

        # Test with custom parameters
        service = FileDiscoveryService(max_file_size=5 * 1024 * 1024, follow_symlinks=True)
        assert service.max_file_size == 5 * 1024 * 1024
        assert service.follow_symlinks is True

    def test_discover_files_basic(self) -> None:
        """Test basic file discovery functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create test files
            text_file = test_dir / "test.txt"
            text_file.write_text("Test content")

            md_file = test_dir / "test.md"
            md_file.write_text("# Markdown")

            py_file = test_dir / "script.py"
            py_file.write_text("print('hello')")

            # Create excluded file
            excluded_dir = test_dir / "node_modules"
            excluded_dir.mkdir()
            excluded_file = excluded_dir / "excluded.txt"
            excluded_file.write_text("Should be excluded")

            # Test discovery
            service = FileDiscoveryService()
            documents = service.discover_files(
                root_paths=[test_dir],
                include_patterns=["*.txt", "*.md", "*.py"],
                exclude_patterns=["*/node_modules/*"],
                max_depth=10,
            )

            # Check results
            assert len(documents) == 3
            paths = [str(doc[0]) for doc in documents]
            assert str(text_file) in paths
            assert str(md_file) in paths
            assert str(py_file) in paths
            assert str(excluded_file) not in paths

            # Check metadata
            for doc_path, metadata in documents:
                assert "file_size" in metadata
                assert "created_at" in metadata
                assert "modified_at" in metadata
                assert "accessed_at" in metadata
                assert "is_symlink" in metadata

    def test_discover_files_with_subdirectories(self) -> None:
        """Test file discovery with subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create nested structure
            sub_dir = test_dir / "subdir"
            sub_dir.mkdir()
            deep_dir = sub_dir / "deep"
            deep_dir.mkdir()

            # Create files at different levels
            root_file = test_dir / "root.txt"
            root_file.write_text("Root file")

            sub_file = sub_dir / "sub.txt"
            sub_file.write_text("Sub file")

            deep_file = deep_dir / "deep.txt"
            deep_file.write_text("Deep file")

            # Test discovery
            service = FileDiscoveryService()
            documents = service.discover_files(
                root_paths=[test_dir],
                include_patterns=["*.txt"],
                exclude_patterns=[],
                max_depth=10,
            )

            # Check results
            assert len(documents) == 3
            paths = [str(doc[0]) for doc in documents]
            assert str(root_file) in paths
            assert str(sub_file) in paths
            assert str(deep_file) in paths

    def test_discover_files_max_depth(self) -> None:
        """Test file discovery with max depth limitation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create nested structure
            sub_dir = test_dir / "subdir"
            sub_dir.mkdir()
            deep_dir = sub_dir / "deep"
            deep_dir.mkdir()

            # Create files at different levels
            root_file = test_dir / "root.txt"
            root_file.write_text("Root file")

            sub_file = sub_dir / "sub.txt"
            sub_file.write_text("Sub file")

            deep_file = deep_dir / "deep.txt"
            deep_file.write_text("Deep file")

            # Test discovery with depth limit
            service = FileDiscoveryService()
            documents = service.discover_files(
                root_paths=[test_dir],
                include_patterns=["*.txt"],
                exclude_patterns=[],
                max_depth=1,  # Should only reach subdir, not deep
            )

            # Check results - should only find root and sub files, not deep
            assert len(documents) == 2
            paths = [str(doc[0]) for doc in documents]
            assert str(root_file) in paths
            assert str(sub_file) in paths
            assert str(deep_file) not in paths

    def test_discover_files_file_size_limit(self) -> None:
        """Test file discovery with file size limitations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create small file
            small_file = test_dir / "small.txt"
            small_file.write_text("Small content")

            # Create large file
            large_file = test_dir / "large.txt"
            large_content = "x" * (2 * 1024 * 1024)  # 2MB
            large_file.write_text(large_content)

            # Test discovery with small size limit
            service = FileDiscoveryService(max_file_size=1024 * 1024)  # 1MB limit
            documents = service.discover_files(
                root_paths=[test_dir],
                include_patterns=["*.txt"],
                exclude_patterns=[],
                max_depth=10,
            )

            # Check results - should only find small file
            assert len(documents) == 1
            assert str(documents[0][0]) == str(small_file)

    def test_discover_files_multiple_root_paths(self) -> None:
        """Test file discovery with multiple root paths."""
        with tempfile.TemporaryDirectory() as temp_dir1, tempfile.TemporaryDirectory() as temp_dir2:
            dir1 = Path(temp_dir1)
            dir2 = Path(temp_dir2)

            # Create files in both directories
            file1 = dir1 / "file1.txt"
            file1.write_text("Content 1")

            file2 = dir2 / "file2.txt"
            file2.write_text("Content 2")

            # Test discovery
            service = FileDiscoveryService()
            documents = service.discover_files(
                root_paths=[dir1, dir2],
                include_patterns=["*.txt"],
                exclude_patterns=[],
                max_depth=10,
            )

            # Check results
            assert len(documents) == 2
            paths = [str(doc[0]) for doc in documents]
            assert str(file1) in paths
            assert str(file2) in paths

    def test_discover_files_invalid_directory(self) -> None:
        """Test discovery with invalid directory."""
        service = FileDiscoveryService()
        
        with pytest.raises(ValueError, match="Directory does not exist"):
            service.discover_files(
                root_paths=[Path("/nonexistent/path")],
                include_patterns=["*.txt"],
                exclude_patterns=[],
                max_depth=10,
            )

    def test_pattern_matching(self) -> None:
        """Test include and exclude pattern matching."""
        service = FileDiscoveryService()

        # Test include patterns
        assert service._matches_patterns(Path("test.txt"), ["*.txt"])
        assert service._matches_patterns(Path("test.md"), ["*.txt", "*.md"])
        assert not service._matches_patterns(Path("test.py"), ["*.txt", "*.md"])

        # Test exclude patterns - note that Path.match() uses different semantics
        assert service._should_exclude(Path("test/node_modules/file.js"), ["*/node_modules/*"])
        assert not service._should_exclude(Path("src/test.txt"), ["*/node_modules/*"])
        
        # Additional pattern tests
        assert service._should_exclude(Path("some/path/node_modules/file.txt"), ["*/node_modules/*"])
        assert not service._should_exclude(Path("node_modules_backup/file.txt"), ["*/node_modules/*"])