"""Document discovery module for Oboyu.

This module provides utilities for discovering documents in the file system.
It respects .gitignore files in the directory hierarchy.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

try:
    from gitignore_parser import parse_gitignore

    HAS_GITIGNORE_PARSER = True
except ImportError:
    HAS_GITIGNORE_PARSER = False

# Type alias for document path and metadata
DocumentInfo = Tuple[Path, Dict[str, object]]

# Hard-coded values (no longer configurable)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
FOLLOW_SYMLINKS = False


def discover_documents(
    directory: Path,
    patterns: List[str],
    exclude_patterns: List[str],
    max_depth: int = 10,
    respect_gitignore: bool = True,
) -> List[DocumentInfo]:
    """Discover documents in a directory based on patterns.

    Args:
        directory: Root directory to start discovery
        patterns: List of glob patterns to include (e.g., "*.txt")
        exclude_patterns: List of glob patterns to exclude (e.g., "*/node_modules/*")
        max_depth: Maximum directory traversal depth
        respect_gitignore: Whether to respect .gitignore files (default: True)

    Returns:
        List of tuples with document path and metadata

    Note:
        max_file_size is hard-coded to 10MB and follow_symlinks is hard-coded to False
        for consistency and security reasons.

    """
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory does not exist or is not a directory: {directory}")

    # Ensure the directory is absolute
    directory = directory.absolute()

    # Keep track of visited directories to avoid cycles
    visited_dirs: Set[Path] = set()

    # List to store discovered documents
    documents: List[DocumentInfo] = []

    # Set up .gitignore matching if enabled and library is available
    gitignore_matcher: Optional[Callable[[str], bool]] = None
    if respect_gitignore and HAS_GITIGNORE_PARSER:
        gitignore_path = directory / ".gitignore"
        if gitignore_path.exists():
            gitignore_matcher = parse_gitignore(gitignore_path)

    # Walk the directory tree
    for doc_path, metadata in _walk_directory(
        directory=directory,
        patterns=patterns,
        exclude_patterns=exclude_patterns,
        current_depth=0,
        max_depth=max_depth,
        max_file_size=MAX_FILE_SIZE,
        follow_symlinks=FOLLOW_SYMLINKS,
        visited_dirs=visited_dirs,
        gitignore_matcher=gitignore_matcher,
        root_directory=directory,
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
    gitignore_matcher: Optional[Callable[[str], bool]] = None,
    root_directory: Optional[Path] = None,
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
        gitignore_matcher: Optional function to match paths against gitignore rules
        root_directory: Root directory for relative path calculation for gitignore

    Yields:
        Tuples of document path and metadata

    """
    # Check if we've reached the max depth
    if current_depth > max_depth:
        return

    # Add this directory to visited set
    visited_dirs.add(directory)

    # Check for .gitignore in this directory if we're using gitignore support
    # This enables us to respect nested .gitignore files
    local_gitignore_matcher = gitignore_matcher
    if HAS_GITIGNORE_PARSER:
        gitignore_path = directory / ".gitignore"
        if gitignore_path.exists():
            # Create a new matcher for this directory's .gitignore
            local_gitignore_matcher = parse_gitignore(gitignore_path)

    try:
        # List directory contents - use scandir for better performance
        # Convert to list first to avoid holding directory handle
        items = list(os.scandir(directory))

        for item in items:
            item_path = Path(item.path)

            # Check if the path should be excluded by patterns
            if _should_exclude(item_path, exclude_patterns):
                continue

            # Check if the path should be excluded by .gitignore
            if local_gitignore_matcher is not None:
                # Get the absolute path as a string
                abs_path_str = str(item_path.absolute())
                # If the path matches gitignore rules, skip it
                if local_gitignore_matcher(abs_path_str):
                    continue

            # Handle directories
            if item.is_dir(follow_symlinks=FOLLOW_SYMLINKS):
                # Skip if we've already visited this directory (to avoid cycles)
                if item_path in visited_dirs:
                    continue

                # Recursively walk subdirectory, passing the gitignore matcher
                yield from _walk_directory(
                    directory=item_path,
                    patterns=patterns,
                    exclude_patterns=exclude_patterns,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    max_file_size=max_file_size,
                    follow_symlinks=follow_symlinks,
                    visited_dirs=visited_dirs,
                    gitignore_matcher=local_gitignore_matcher,
                    root_directory=root_directory or directory,
                )

            # Handle files
            elif item.is_file(follow_symlinks=FOLLOW_SYMLINKS):
                # Check if the file matches any of the include patterns
                if _matches_patterns(item_path, patterns):
                    # Check file size
                    file_size = item.stat(follow_symlinks=FOLLOW_SYMLINKS).st_size
                    if file_size <= MAX_FILE_SIZE:
                        # Collect metadata
                        metadata = _get_file_metadata(item, FOLLOW_SYMLINKS)
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
