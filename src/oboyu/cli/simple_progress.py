"""Simple single-line progress display without flickering.

This module provides a clean, non-flickering progress display that uses
simple line overwriting instead of Rich.Live.
"""

import sys
import time
from typing import Callable, Dict, Optional


class SimpleProgressDisplay:
    """Simple single-line progress display."""

    def __init__(self) -> None:
        """Initialize the progress display."""
        self.current_stage: Optional[str] = None
        self.start_time: float = time.time()
        self.last_update: float = 0
        self.stages: Dict[str, str] = {
            "detecting_changes": "Detecting changes",
            "cleaning_deleted": "Cleaning deleted files",
            "crawling": "Scanning files",
            "processing": "Processing files",
            "storing": "Storing chunks",
            "embedding": "Generating embeddings",
            "storing_embeddings": "Storing embeddings",
            "bm25_indexing": "Building search index",
            "bm25_tokenizing": "Tokenizing",
            "bm25_vocabulary": "Building vocabulary",
            "bm25_filtering": "Filtering terms",
            "bm25_store_vocabulary": "Storing vocabulary",
            "bm25_store_inverted_index": "Storing index",
            "bm25_store_document_stats": "Storing statistics",
            "bm25_creating_indexes": "Creating indexes",
            "recompacting": "Optimizing database",
        }

    def update(self, stage: str, current: int, total: int) -> None:
        """Update progress display.

        Args:
            stage: Current stage name
            current: Current progress
            total: Total items

        """
        now = time.time()

        # Update at reasonable intervals
        if stage != self.current_stage or now - self.last_update > 2.0 or current >= total:
            self.current_stage = stage
            self.last_update = now

            # Get stage description
            stage_desc = self.stages.get(stage, stage.replace("_", " ").title())

            # Calculate percentage
            if total > 0:
                percent = (current / total) * 100
                progress_text = f"{stage_desc}... {current}/{total} ({percent:.0f}%)"
            else:
                progress_text = f"{stage_desc}... {current}"

            # Calculate elapsed time
            elapsed = now - self.start_time
            if elapsed > 60:
                time_text = f" [{elapsed / 60:.1f}m]"
            else:
                time_text = f" [{elapsed:.0f}s]"

            # Overwrite current line
            sys.stderr.write(f"\r{progress_text}{time_text}")
            sys.stderr.flush()

            # New line on completion
            if current >= total and total > 0:
                sys.stderr.write("\n")
                sys.stderr.flush()

    def finish(self, message: Optional[str] = None) -> None:
        """Finish progress display.

        Args:
            message: Optional completion message

        """
        if message:
            sys.stderr.write(f"\r{message}\n")
        else:
            sys.stderr.write("\n")
        sys.stderr.flush()


def create_simple_progress_callback() -> Callable[[int, int], None]:
    """Create a simple progress callback function.

    Returns:
        Progress callback function

    """
    display = SimpleProgressDisplay()
    
    def callback(current: int, total: int) -> None:
        """Progress callback wrapper."""
        display.update("Progress", current, total)
    
    return callback
