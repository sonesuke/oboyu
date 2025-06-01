"""Metadata extraction service for extracting file metadata."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class MetadataExtractor:
    """Service responsible for extracting file metadata (timestamps, size, etc.)."""

    def __init__(self, follow_symlinks: bool = False) -> None:
        """Initialize the metadata extractor.
        
        Args:
            follow_symlinks: Whether to follow symbolic links when extracting metadata

        """
        self.follow_symlinks = follow_symlinks

    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract file metadata (timestamps, size, etc.).
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file metadata

        """
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")

        stat_result = file_path.stat()

        return {
            "file_size": stat_result.st_size,
            "created_at": datetime.fromtimestamp(stat_result.st_ctime),
            "modified_at": datetime.fromtimestamp(stat_result.st_mtime),
            "accessed_at": datetime.fromtimestamp(stat_result.st_atime),
            "is_symlink": file_path.is_symlink(),
        }

    def extract_metadata_from_direntry(self, file_entry: os.DirEntry[str]) -> Dict[str, Any]:
        """Extract metadata from a directory entry.
        
        Args:
            file_entry: Directory entry for the file
            
        Returns:
            Dictionary with file metadata

        """
        stat_result = file_entry.stat(follow_symlinks=self.follow_symlinks)

        return {
            "file_size": stat_result.st_size,
            "created_at": datetime.fromtimestamp(stat_result.st_ctime),
            "modified_at": datetime.fromtimestamp(stat_result.st_mtime),
            "accessed_at": datetime.fromtimestamp(stat_result.st_atime),
            "is_symlink": file_entry.is_symlink(),
        }
