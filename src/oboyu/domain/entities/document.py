"""Document entity - core business object for documents."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..value_objects.content_hash import ContentHash
from ..value_objects.language_code import LanguageCode


@dataclass
class Document:
    """Core document entity with pure business logic."""
    
    path: Path
    title: str
    content: str
    language: LanguageCode
    created_at: datetime
    modified_at: datetime
    content_hash: ContentHash
    metadata: Dict[str, object]
    
    def __post_init__(self) -> None:
        """Validate document consistency."""
        if not self.title.strip():
            raise ValueError("Document title cannot be empty")
        
        if not self.content.strip():
            raise ValueError("Document content cannot be empty")
        
        if self.created_at > self.modified_at:
            raise ValueError("Created date cannot be after modified date")
        
        if not self.content_hash.matches_content(self.content):
            raise ValueError("Content hash does not match document content")
    
    def get_size(self) -> int:
        """Get document size in characters."""
        return len(self.content)
    
    def is_empty(self) -> bool:
        """Check if document has meaningful content."""
        return len(self.content.strip()) == 0
    
    def extract_title_from_content(self) -> str:
        """Extract title from content if not explicitly set."""
        if self.title and self.title.strip():
            return self.title
        
        lines = self.content.split('\n')
        for line in lines:
            cleaned = line.strip()
            if cleaned and not cleaned.startswith('#'):
                return cleaned[:100]
        
        return str(self.path.stem)
    
    def should_be_processed(self) -> bool:
        """Determine if document should be processed for indexing."""
        return (
            not self.is_empty()
            and self.get_size() > 10
            and self.language != LanguageCode.UNKNOWN
        )
    
    def create_updated_copy(self, *, content: Optional[str] = None,
                          title: Optional[str] = None) -> "Document":
        """Create an updated copy of the document."""
        new_content = content if content is not None else self.content
        new_title = title if title is not None else self.title
        new_hash = ContentHash.from_content(new_content)
        
        return Document(
            path=self.path,
            title=new_title,
            content=new_content,
            language=self.language,
            created_at=self.created_at,
            modified_at=datetime.now(),
            content_hash=new_hash,
            metadata=self.metadata.copy()
        )
