"""File change detection for incremental indexing.

This module provides intelligent file change detection strategies for
efficient incremental indexing. It tracks file modifications through
timestamps, size changes, and content hashes.

Key features:
- Multiple change detection strategies (timestamp, hash, smart)
- Persistent file metadata tracking
- Efficient batch processing
- Support for large document collections
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from oboyu.indexer.storage.database_service import DatabaseService

logger = logging.getLogger(__name__)


@dataclass
class ChangeResult:
    """Result of change detection analysis."""

    new_files: List[Path]
    modified_files: List[Path]
    deleted_files: List[Path]

    @property
    def total_changes(self) -> int:
        """Get total number of changes."""
        return len(self.new_files) + len(self.modified_files) + len(self.deleted_files)

    def has_changes(self) -> bool:
        """Check if any changes were detected."""
        return self.total_changes > 0


class FileChangeDetector:
    """Intelligent file change detection for incremental indexing.

    This class provides multiple strategies for detecting file changes:
    - timestamp: Use file modification time only (fastest)
    - hash: Use content hash comparison (most accurate)
    - smart: Combine timestamp + size + hash for optimal balance
    """

    def __init__(self, database: DatabaseService, batch_size: int = 1000) -> None:
        """Initialize change detector.

        Args:
            database: Database connection for metadata storage
            batch_size: Batch size for processing large file sets

        """
        self.db = database
        self.batch_size = batch_size

    def detect_changes(self, file_paths: List[Path], strategy: str = "smart") -> ChangeResult:
        """Detect new, modified, and deleted files.

        Args:
            file_paths: List of file paths to check
            strategy: Detection strategy ('timestamp', 'hash', 'smart')

        Returns:
            ChangeResult with categorized file changes

        """
        logger.info(f"Detecting changes for {len(file_paths)} files using '{strategy}' strategy")

        # Get existing file metadata from database
        existing_metadata = self._get_existing_metadata()
        current_paths = set(str(p) for p in file_paths)
        existing_paths = set(existing_metadata.keys())

        # Categorize files
        new_files = []
        modified_files = []

        # Find new and potentially modified files
        for path in file_paths:
            path_str = str(path)

            if path_str not in existing_paths:
                new_files.append(path)
            else:
                # Check if file is modified based on strategy
                if self._is_file_modified(path, existing_metadata[path_str], strategy):
                    modified_files.append(path)

        # Find deleted files
        deleted_paths = existing_paths - current_paths
        deleted_files = [Path(p) for p in deleted_paths]

        result = ChangeResult(new_files=new_files, modified_files=modified_files, deleted_files=deleted_files)

        logger.info(f"Change detection complete: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_files)} deleted")

        return result

    def _get_existing_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get all existing file metadata from database.

        Returns:
            Dictionary mapping file paths to their metadata

        """
        try:
            conn = self.db._ensure_connection()
            results = conn.execute("""
                SELECT path, last_processed_at, file_modified_at, file_size, content_hash
                FROM file_metadata
                WHERE processing_status = 'completed'
            """).fetchall()

            metadata = {}
            for row in results:
                metadata[row[0]] = {"last_processed_at": row[1], "file_modified_at": row[2], "file_size": row[3], "content_hash": row[4]}

            return metadata

        except Exception as e:
            logger.error(f"Failed to get existing metadata: {e}")
            return {}

    def _is_file_modified(self, file_path: Path, existing_meta: Dict[str, Any], strategy: str) -> bool:
        """Check if a file has been modified.

        Args:
            file_path: Path to file
            existing_meta: Existing metadata for the file
            strategy: Detection strategy to use

        Returns:
            True if file is modified

        """
        try:
            file_stats = file_path.stat()

            if strategy == "timestamp":
                return self._check_timestamp(file_stats, existing_meta)

            elif strategy == "hash":
                return self._check_hash(file_path, existing_meta)

            elif strategy == "smart":
                # First check timestamp and size (fast)
                if self._check_timestamp(file_stats, existing_meta):
                    return True

                if file_stats.st_size != existing_meta.get("file_size", -1):
                    return True

                # If timestamp and size match, check hash for accuracy
                # (only for files under 10MB to avoid performance issues)
                if file_stats.st_size < 10 * 1024 * 1024:  # 10MB
                    return self._check_hash(file_path, existing_meta)

                # For large files, trust timestamp and size
                return False

            else:
                logger.warning(f"Unknown strategy '{strategy}', defaulting to timestamp")
                return self._check_timestamp(file_stats, existing_meta)

        except Exception as e:
            logger.error(f"Error checking if file is modified: {e}")
            # If we can't determine, assume it's modified to be safe
            return True

    def _check_timestamp(self, file_stats: Any, existing_meta: Dict[str, Any]) -> bool:  # noqa: ANN401
        """Check if file modification timestamp has changed.

        Args:
            file_stats: File stat object
            existing_meta: Existing metadata

        Returns:
            True if timestamp is newer

        """
        file_modified_time = datetime.fromtimestamp(file_stats.st_mtime)
        stored_modified_time = existing_meta.get("file_modified_at")

        if stored_modified_time is None:
            return True

        # Parse stored timestamp if it's a string
        if isinstance(stored_modified_time, str):
            parsed_time = datetime.fromisoformat(stored_modified_time)
            return file_modified_time > parsed_time
        elif isinstance(stored_modified_time, datetime):
            return file_modified_time > stored_modified_time
        else:
            # Unknown type, consider as modified
            return True

    def _check_hash(self, file_path: Path, existing_meta: Dict[str, Any]) -> bool:
        """Check if file content hash has changed.

        Args:
            file_path: Path to file
            existing_meta: Existing metadata

        Returns:
            True if hash is different

        """
        stored_hash = existing_meta.get("content_hash")
        if stored_hash is None:
            return True

        current_hash = self.calculate_file_hash(file_path)
        return bool(current_hash != stored_hash)

    @staticmethod
    def calculate_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate SHA-256 hash of file content.

        Args:
            file_path: Path to file
            chunk_size: Size of chunks to read

        Returns:
            Hexadecimal hash string

        """
        hash_obj = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    hash_obj.update(chunk)

            return hash_obj.hexdigest()

        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            # Return a unique value that won't match any stored hash
            return f"error_{datetime.now().timestamp()}"

    def mark_files_for_reprocessing(self, file_paths: List[Path]) -> None:
        """Mark specific files for reprocessing.

        Args:
            file_paths: Files to mark for reprocessing

        """
        try:
            # Update processing status for specified files
            conn = self.db._ensure_connection()
            for path in file_paths:
                conn.execute(
                    """
                    UPDATE file_metadata
                    SET processing_status = 'pending'
                    WHERE path = ?
                """,
                    [str(path)],
                )

            logger.info(f"Marked {len(file_paths)} files for reprocessing")

        except Exception as e:
            logger.error(f"Failed to mark files for reprocessing: {e}")

    def get_processing_stats(self) -> Dict[str, int]:
        """Get statistics about file processing status.

        Returns:
            Dictionary with counts by status

        """
        try:
            conn = self.db._ensure_connection()
            results = conn.execute("""
                SELECT processing_status, COUNT(*) as count
                FROM file_metadata
                GROUP BY processing_status
            """).fetchall()

            stats = {row[0]: row[1] for row in results}
            stats["total"] = sum(stats.values())

            return stats

        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            return {"error": 1}

    def cleanup_deleted_files(self, deleted_files: List[Path]) -> None:
        """Remove metadata for deleted files.

        Args:
            deleted_files: List of deleted file paths

        """
        if not deleted_files:
            return

        try:
            # Remove file metadata
            conn = self.db._ensure_connection()
            for path in deleted_files:
                conn.execute(
                    """
                    DELETE FROM file_metadata
                    WHERE path = ?
                """,
                    [str(path)],
                )

            logger.info(f"Cleaned up metadata for {len(deleted_files)} deleted files")

        except Exception as e:
            logger.error(f"Failed to cleanup deleted files: {e}")
