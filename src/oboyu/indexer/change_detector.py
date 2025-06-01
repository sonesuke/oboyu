"""Legacy import compatibility for change_detector module."""

# Re-export from the new location
from oboyu.indexer.storage.change_detector import ChangeResult, FileChangeDetector

__all__ = ["ChangeResult", "FileChangeDetector"]
