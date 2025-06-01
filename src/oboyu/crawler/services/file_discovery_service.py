"""File discovery service for finding files that match criteria."""

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

# Hard-coded values for consistency and security
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
FOLLOW_SYMLINKS = False


class FileDiscoveryService:
    """Service responsible for finding files that match criteria."""

    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE,
        follow_symlinks: bool = FOLLOW_SYMLINKS,
    ) -> None:
        """Initialize the file discovery service.
        
        Args:
            max_file_size: Maximum file size in bytes (default: 10MB)
            follow_symlinks: Whether to follow symbolic links (default: False)

        """
        self.max_file_size = max_file_size
        self.follow_symlinks = follow_symlinks

    def discover_files(
        self,
        root_paths: List[Path],
        include_patterns: List[str],
        exclude_patterns: List[str],
        max_depth: int,
        respect_gitignore: bool = True,
    ) -> List[DocumentInfo]:
        """Discover files matching the specified criteria.
        
        Args:
            root_paths: List of root directories to search
            include_patterns: List of glob patterns to include (e.g., "*.txt")
            exclude_patterns: List of glob patterns to exclude (e.g., "*/node_modules/*")
            max_depth: Maximum directory traversal depth
            respect_gitignore: Whether to respect .gitignore files
            
        Returns:
            List of tuples with document path and metadata

        """
        documents: List[DocumentInfo] = []
        
        for directory in root_paths:
            if not directory.exists() or not directory.is_dir():
                raise ValueError(f"Directory does not exist or is not a directory: {directory}")
            
            # Get documents from this directory
            dir_documents = self._discover_in_directory(
                directory=directory.absolute(),
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                max_depth=max_depth,
                respect_gitignore=respect_gitignore,
            )
            documents.extend(dir_documents)
        
        return documents

    def _discover_in_directory(
        self,
        directory: Path,
        include_patterns: List[str],
        exclude_patterns: List[str],
        max_depth: int,
        respect_gitignore: bool,
    ) -> List[DocumentInfo]:
        """Discover documents in a single directory.
        
        Args:
            directory: Directory to search (must be absolute)
            include_patterns: List of glob patterns to include
            exclude_patterns: List of glob patterns to exclude
            max_depth: Maximum directory traversal depth
            respect_gitignore: Whether to respect .gitignore files
            
        Returns:
            List of tuples with document path and metadata

        """
        documents: List[DocumentInfo] = []
        visited_dirs: Set[Path] = set()

        # Set up .gitignore matching if enabled and library is available
        gitignore_matcher: Optional[Callable[[str], bool]] = None
        if respect_gitignore and HAS_GITIGNORE_PARSER:
            gitignore_path = directory / ".gitignore"
            if gitignore_path.exists():
                gitignore_matcher = parse_gitignore(gitignore_path)

        # Walk the directory tree
        for doc_path, metadata in self._walk_directory(
            directory=directory,
            patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            current_depth=0,
            max_depth=max_depth,
            visited_dirs=visited_dirs,
            gitignore_matcher=gitignore_matcher,
            root_directory=directory,
        ):
            documents.append((doc_path, metadata))

        return documents

    def _walk_directory(
        self,
        directory: Path,
        patterns: List[str],
        exclude_patterns: List[str],
        current_depth: int,
        max_depth: int,
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
            visited_dirs: Set of already visited directories
            gitignore_matcher: Optional function to match paths against gitignore rules
            root_directory: Root directory for relative path calculation for gitignore
            
        Yields:
            Tuples with document path and metadata

        """
        # Check if we've reached the max depth
        if current_depth > max_depth:
            return

        # Add this directory to visited set
        visited_dirs.add(directory)

        # Check for .gitignore in this directory if we're using gitignore support
        local_gitignore_matcher = gitignore_matcher
        if HAS_GITIGNORE_PARSER:
            gitignore_path = directory / ".gitignore"
            if gitignore_path.exists():
                # Create a new matcher for this directory's .gitignore
                local_gitignore_matcher = parse_gitignore(gitignore_path)

        try:
            # List directory contents - use scandir for better performance
            items = list(os.scandir(directory))

            for item in items:
                item_path = Path(item.path)

                # Check if the path should be excluded by patterns
                if self._should_exclude(item_path, exclude_patterns):
                    continue

                # Check if the path should be excluded by .gitignore
                if local_gitignore_matcher is not None:
                    abs_path_str = str(item_path.absolute())
                    if local_gitignore_matcher(abs_path_str):
                        continue

                # Handle directories
                if item.is_dir(follow_symlinks=self.follow_symlinks):
                    # Skip if we've already visited this directory (to avoid cycles)
                    if item_path in visited_dirs:
                        continue

                    # Recursively walk subdirectory
                    yield from self._walk_directory(
                        directory=item_path,
                        patterns=patterns,
                        exclude_patterns=exclude_patterns,
                        current_depth=current_depth + 1,
                        max_depth=max_depth,
                        visited_dirs=visited_dirs,
                        gitignore_matcher=local_gitignore_matcher,
                        root_directory=root_directory or directory,
                    )

                # Handle files
                elif item.is_file(follow_symlinks=self.follow_symlinks):
                    # Check if the file matches any of the include patterns
                    if self._matches_patterns(item_path, patterns):
                        # Check file size
                        file_size = item.stat(follow_symlinks=self.follow_symlinks).st_size
                        if file_size <= self.max_file_size:
                            # Collect metadata
                            metadata = self._get_file_metadata(item)
                            yield (item_path, metadata)

        except PermissionError:
            # Skip directories we don't have permission to access
            pass

    def _should_exclude(self, path: Path, exclude_patterns: List[str]) -> bool:
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

    def _matches_patterns(self, path: Path, patterns: List[str]) -> bool:
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

    def _get_file_metadata(self, file_entry: os.DirEntry[str]) -> Dict[str, object]:
        """Extract metadata from a file.
        
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
