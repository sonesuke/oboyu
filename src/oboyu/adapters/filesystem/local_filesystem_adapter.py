"""Local filesystem adapter implementation."""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ...crawler.crawler import Crawler
from ...crawler.services import LanguageDetector
from ...domain.entities.document import Document
from ...domain.value_objects.content_hash import ContentHash
from ...domain.value_objects.language_code import LanguageCode
from ...ports.external.filesystem_port import FilesystemPort

logger = logging.getLogger(__name__)


class LocalFilesystemAdapter(FilesystemPort):
    """Local filesystem implementation of filesystem port."""
    
    def __init__(self, crawler_service: Crawler, language_detector: LanguageDetector) -> None:
        """Initialize with existing services."""
        self._crawler_service = crawler_service
        self._language_detector = language_detector
    
    async def discover_files(self, root_path: Path,
                           include_patterns: Optional[List[str]] = None,
                           exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """Discover files matching patterns."""
        return await self._crawler_service.discover_files(
            root_path, include_patterns, exclude_patterns
        )
    
    async def read_document(self, path: Path) -> Document:
        """Read a document from filesystem."""
        content = path.read_text(encoding='utf-8')
        stat = path.stat()
        
        language = await self.detect_language(content)
        content_hash = ContentHash.from_content(content)
        
        title = self._extract_title_from_path(path)
        
        return Document(
            path=path,
            title=title,
            content=content,
            language=language,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            content_hash=content_hash,
            metadata={
                'file_size': stat.st_size,
                'encoding': 'utf-8',
                'mime_type': self._get_mime_type(path)
            }
        )
    
    async def detect_language(self, content: str) -> LanguageCode:
        """Detect the language of content."""
        detected_lang = await self._language_detector.detect_language(content)
        return LanguageCode.from_string(detected_lang)
    
    async def detect_encoding(self, path: Path) -> str:
        """Detect the encoding of a file."""
        return 'utf-8'
    
    def file_exists(self, path: Path) -> bool:
        """Check if file exists."""
        return path.exists() and path.is_file()
    
    def get_file_size(self, path: Path) -> int:
        """Get file size in bytes."""
        return path.stat().st_size
    
    def get_modification_time(self, path: Path) -> float:
        """Get file modification timestamp."""
        return path.stat().st_mtime
    
    def _extract_title_from_path(self, path: Path) -> str:
        """Extract title from file path."""
        return path.stem.replace('_', ' ').replace('-', ' ').title()
    
    def _get_mime_type(self, path: Path) -> str:
        """Get MIME type from file extension."""
        ext = path.suffix.lower()
        mime_types = {
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.py': 'text/x-python',
            '.rst': 'text/x-rst',
            '.html': 'text/html',
            '.xml': 'text/xml',
            '.json': 'application/json',
            '.yaml': 'text/yaml',
            '.yml': 'text/yaml'
        }
        return mime_types.get(ext, 'text/plain')
