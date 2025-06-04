"""Filesystem port interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from ...domain.entities.document import Document
from ...domain.value_objects.language_code import LanguageCode


class FilesystemPort(ABC):
    """Abstract interface for filesystem operations."""
    
    @abstractmethod
    async def discover_files(self, root_path: Path,
                           include_patterns: Optional[List[str]] = None,
                           exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """Discover files matching patterns."""
        pass
    
    @abstractmethod
    async def read_document(self, path: Path) -> Document:
        """Read a document from filesystem."""
        pass
    
    @abstractmethod
    async def detect_language(self, content: str) -> LanguageCode:
        """Detect the language of content."""
        pass
    
    @abstractmethod
    async def detect_encoding(self, path: Path) -> str:
        """Detect the encoding of a file."""
        pass
    
    @abstractmethod
    def file_exists(self, path: Path) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    def get_file_size(self, path: Path) -> int:
        """Get file size in bytes."""
        pass
    
    @abstractmethod
    def get_modification_time(self, path: Path) -> float:
        """Get file modification timestamp."""
        pass
