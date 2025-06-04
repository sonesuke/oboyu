"""Document repository port interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from ...domain.entities.document import Document
from ...domain.value_objects.content_hash import ContentHash


class DocumentRepository(ABC):
    """Abstract interface for document repository operations."""
    
    @abstractmethod
    async def store_document(self, document: Document) -> None:
        """Store a document in the repository."""
        pass
    
    @abstractmethod
    async def find_by_path(self, path: Path) -> Optional[Document]:
        """Find a document by its file path."""
        pass
    
    @abstractmethod
    async def find_by_content_hash(self, content_hash: ContentHash) -> Optional[Document]:
        """Find a document by its content hash."""
        pass
    
    @abstractmethod
    async def get_all_documents(self) -> List[Document]:
        """Get all documents in the repository."""
        pass
    
    @abstractmethod
    async def delete_document(self, path: Path) -> None:
        """Delete a document from the repository."""
        pass
    
    @abstractmethod
    async def document_exists(self, path: Path) -> bool:
        """Check if a document exists in the repository."""
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """Get total number of documents in repository."""
        pass
    
    @abstractmethod
    async def has_changed(self, path: Path, current_hash: ContentHash) -> bool:
        """Check if a document has changed since last indexing."""
        pass
