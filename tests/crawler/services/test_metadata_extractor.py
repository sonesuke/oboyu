"""Tests for MetadataExtractor."""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from oboyu.crawler.services.metadata_extractor import MetadataExtractor


class TestMetadataExtractor:
    """Test cases for MetadataExtractor."""

    def test_initialization(self) -> None:
        """Test extractor initialization."""
        # Test with default parameters
        extractor = MetadataExtractor()
        assert extractor.follow_symlinks is False

        # Test with custom parameters
        extractor = MetadataExtractor(follow_symlinks=True)
        assert extractor.follow_symlinks is True

    def test_extract_metadata_basic(self) -> None:
        """Test basic metadata extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create test file
            test_file = test_dir / "test.txt"
            test_content = "This is test content"
            test_file.write_text(test_content)

            # Extract metadata
            extractor = MetadataExtractor()
            metadata = extractor.extract_metadata(test_file)

            # Check metadata fields
            assert "file_size" in metadata
            assert "created_at" in metadata
            assert "modified_at" in metadata
            assert "accessed_at" in metadata
            assert "is_symlink" in metadata

            # Check types
            assert isinstance(metadata["file_size"], int)
            assert isinstance(metadata["created_at"], datetime)
            assert isinstance(metadata["modified_at"], datetime)
            assert isinstance(metadata["accessed_at"], datetime)
            assert isinstance(metadata["is_symlink"], bool)

            # Check values
            assert metadata["file_size"] == len(test_content.encode("utf-8"))
            assert metadata["is_symlink"] is False

    def test_extract_metadata_file_size(self) -> None:
        """Test metadata extraction with different file sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create small file
            small_file = test_dir / "small.txt"
            small_content = "Small"
            small_file.write_text(small_content)

            # Create larger file
            large_file = test_dir / "large.txt"
            large_content = "Large content " * 1000
            large_file.write_text(large_content)

            extractor = MetadataExtractor()

            # Check small file
            small_metadata = extractor.extract_metadata(small_file)
            assert small_metadata["file_size"] == len(small_content.encode("utf-8"))

            # Check large file
            large_metadata = extractor.extract_metadata(large_file)
            assert large_metadata["file_size"] == len(large_content.encode("utf-8"))
            assert large_metadata["file_size"] > small_metadata["file_size"]

    def test_extract_metadata_timestamps(self) -> None:
        """Test metadata extraction with timestamp validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create test file
            test_file = test_dir / "test.txt"
            test_file.write_text("Test content")

            # Get creation time
            creation_time = datetime.now()

            # Extract metadata
            extractor = MetadataExtractor()
            metadata = extractor.extract_metadata(test_file)

            # Check that timestamps are reasonable (within last few seconds)
            time_diff = abs((metadata["created_at"] - creation_time).total_seconds())
            assert time_diff < 60  # Should be within 1 minute

            time_diff = abs((metadata["modified_at"] - creation_time).total_seconds())
            assert time_diff < 60  # Should be within 1 minute

    def test_extract_metadata_invalid_file(self) -> None:
        """Test metadata extraction with invalid file."""
        extractor = MetadataExtractor()
        
        with pytest.raises(ValueError, match="File does not exist"):
            extractor.extract_metadata(Path("/nonexistent/file.txt"))

    def test_extract_metadata_from_direntry(self) -> None:
        """Test metadata extraction from directory entry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create test file
            test_file = test_dir / "test.txt"
            test_content = "Test content for direntry"
            test_file.write_text(test_content)

            # Get directory entry
            entries = list(os.scandir(test_dir))
            test_entry = next(entry for entry in entries if entry.name == "test.txt")

            # Extract metadata from direntry
            extractor = MetadataExtractor()
            metadata = extractor.extract_metadata_from_direntry(test_entry)

            # Check metadata fields
            assert "file_size" in metadata
            assert "created_at" in metadata
            assert "modified_at" in metadata
            assert "accessed_at" in metadata
            assert "is_symlink" in metadata

            # Check values
            assert metadata["file_size"] == len(test_content.encode("utf-8"))
            assert metadata["is_symlink"] is False

    def test_extract_metadata_different_follow_symlinks(self) -> None:
        """Test metadata extraction with different follow_symlinks settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create test file
            test_file = test_dir / "test.txt"
            test_file.write_text("Test content")

            # Test with follow_symlinks=False
            extractor_no_follow = MetadataExtractor(follow_symlinks=False)
            metadata_no_follow = extractor_no_follow.extract_metadata(test_file)

            # Test with follow_symlinks=True
            extractor_follow = MetadataExtractor(follow_symlinks=True)
            metadata_follow = extractor_follow.extract_metadata(test_file)

            # For regular files, results should be the same
            assert metadata_no_follow["file_size"] == metadata_follow["file_size"]
            assert metadata_no_follow["is_symlink"] == metadata_follow["is_symlink"]

    def test_extract_metadata_symlink(self) -> None:
        """Test metadata extraction with symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create original file
            original_file = test_dir / "original.txt"
            original_content = "Original content"
            original_file.write_text(original_content)

            # Create symbolic link
            try:
                symlink_file = test_dir / "symlink.txt"
                symlink_file.symlink_to(original_file)

                # Extract metadata from symlink
                extractor = MetadataExtractor()
                metadata = extractor.extract_metadata(symlink_file)

                # Check that it's detected as a symlink
                assert metadata["is_symlink"] is True

            except OSError:
                # Skip test if symlinks are not supported on this system
                pytest.skip("Symbolic links not supported on this system")

    def test_extract_metadata_direntry_with_symlinks(self) -> None:
        """Test metadata extraction from directory entry with symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create original file
            original_file = test_dir / "original.txt"
            original_file.write_text("Original content")

            try:
                # Create symbolic link
                symlink_file = test_dir / "symlink.txt"
                symlink_file.symlink_to(original_file)

                # Get directory entries
                entries = list(os.scandir(test_dir))
                symlink_entry = next(entry for entry in entries if entry.name == "symlink.txt")

                # Extract metadata from symlink direntry
                extractor = MetadataExtractor()
                metadata = extractor.extract_metadata_from_direntry(symlink_entry)

                # Check that it's detected as a symlink
                assert metadata["is_symlink"] is True

            except OSError:
                # Skip test if symlinks are not supported on this system
                pytest.skip("Symbolic links not supported on this system")

    def test_extract_metadata_consistency(self) -> None:
        """Test consistency between extract_metadata and extract_metadata_from_direntry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create test file
            test_file = test_dir / "test.txt"
            test_content = "Consistency test content"
            test_file.write_text(test_content)

            # Get directory entry
            entries = list(os.scandir(test_dir))
            test_entry = next(entry for entry in entries if entry.name == "test.txt")

            # Extract metadata both ways
            extractor = MetadataExtractor()
            metadata_from_path = extractor.extract_metadata(test_file)
            metadata_from_entry = extractor.extract_metadata_from_direntry(test_entry)

            # Results should be consistent
            assert metadata_from_path["file_size"] == metadata_from_entry["file_size"]
            assert metadata_from_path["is_symlink"] == metadata_from_entry["is_symlink"]
            
            # Timestamps might differ slightly due to access times, but should be close
            time_diff = abs((metadata_from_path["modified_at"] - metadata_from_entry["modified_at"]).total_seconds())
            assert time_diff < 2  # Should be within 2 seconds