"""Document discovery module for Oboyu.

This module provides utilities for discovering documents in the file system.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

# Type alias for document path and metadata
DocumentInfo = Tuple[Path, Dict[str, object]]


def discover_documents(
    directory: Path,
    patterns: List[str],
    exclude_patterns: List[str],
    max_depth: int = 10,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    follow_symlinks: bool = False,
) -> List[DocumentInfo]:
    """Discover documents in a directory based on patterns.

    Args:
        directory: Root directory to start discovery
        patterns: List of glob patterns to include (e.g., "*.txt")
        exclude_patterns: List of glob patterns to exclude (e.g., "*/node_modules/*")
        max_depth: Maximum directory traversal depth
        max_file_size: Maximum file size in bytes
        follow_symlinks: Whether to follow symbolic links

    Returns:
        List of tuples with document path and metadata

    """
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory does not exist or is not a directory: {directory}")

    # Ensure the directory is absolute
    directory = directory.absolute()

    # Keep track of visited directories to avoid cycles
    visited_dirs: Set[Path] = set()

    # List to store discovered documents
    documents: List[DocumentInfo] = []

    # Walk the directory tree
    for doc_path, metadata in _walk_directory(
        directory=directory,
        patterns=patterns,
        exclude_patterns=exclude_patterns,
        current_depth=0,
        max_depth=max_depth,
        max_file_size=max_file_size,
        follow_symlinks=follow_symlinks,
        visited_dirs=visited_dirs,
    ):
        documents.append((doc_path, metadata))

    return documents


def _walk_directory(
    directory: Path,
    patterns: List[str],
    exclude_patterns: List[str],
    current_depth: int,
    max_depth: int,
    max_file_size: int,
    follow_symlinks: bool,
    visited_dirs: Set[Path],
) -> Iterator[DocumentInfo]:
    """Recursively walk a directory to discover documents.

    Args:
        directory: Current directory to walk
        patterns: List of glob patterns to include
        exclude_patterns: List of glob patterns to exclude
        current_depth: Current traversal depth
        max_depth: Maximum directory traversal depth
        max_file_size: Maximum file size in bytes
        follow_symlinks: Whether to follow symbolic links
        visited_dirs: Set of already visited directories

    Yields:
        Tuples of document path and metadata

    """
    # Check if we've reached the max depth
    if current_depth > max_depth:
        return

    # Add this directory to visited set
    visited_dirs.add(directory)

    try:
        # List directory contents
        for item in os.scandir(directory):
            item_path = Path(item.path)

            # Check if the path should be excluded
            if _should_exclude(item_path, exclude_patterns):
                continue

            # Handle directories
            if item.is_dir(follow_symlinks=follow_symlinks):
                # Skip if we've already visited this directory (to avoid cycles)
                if item_path in visited_dirs:
                    continue

                # Recursively walk subdirectory
                yield from _walk_directory(
                    directory=item_path,
                    patterns=patterns,
                    exclude_patterns=exclude_patterns,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    max_file_size=max_file_size,
                    follow_symlinks=follow_symlinks,
                    visited_dirs=visited_dirs,
                )

            # Handle files
            elif item.is_file(follow_symlinks=follow_symlinks):
                # Check if the file matches any of the include patterns
                if _matches_patterns(item_path, patterns):
                    # Check file size
                    file_size = item.stat(follow_symlinks=follow_symlinks).st_size
                    if file_size <= max_file_size:
                        # Collect metadata
                        metadata = _get_file_metadata(item, follow_symlinks)
                        yield (item_path, metadata)
    except PermissionError:
        # Skip directories we don't have permission to access
        pass


def _should_exclude(path: Path, exclude_patterns: List[str]) -> bool:
    """Check if a path should be excluded based on patterns.

    Args:
        path: Path to check
        exclude_patterns: List of glob patterns to exclude

    Returns:
        True if the path should be excluded, False otherwise

    """
    for pattern in exclude_patterns:
        if path.match(pattern):
            return True
    return False


def _matches_patterns(path: Path, patterns: List[str]) -> bool:
    """Check if a path matches any of the include patterns.

    Args:
        path: Path to check
        patterns: List of glob patterns to include

    Returns:
        True if the path matches any pattern, False otherwise

    """
    for pattern in patterns:
        if path.match(pattern):
            return True
    return False


def _get_file_metadata(file_entry: os.DirEntry[str], follow_symlinks: bool) -> Dict[str, object]:
    """Extract metadata from a file.

    Args:
        file_entry: Directory entry for the file
        follow_symlinks: Whether to follow symbolic links

    Returns:
        Dictionary with file metadata

    """
    stat_result = file_entry.stat(follow_symlinks=follow_symlinks)

    return {
        "file_size": stat_result.st_size,
        "created_at": datetime.fromtimestamp(stat_result.st_ctime),
        "modified_at": datetime.fromtimestamp(stat_result.st_mtime),
        "accessed_at": datetime.fromtimestamp(stat_result.st_atime),
        "is_symlink": file_entry.is_symlink(),
    }
