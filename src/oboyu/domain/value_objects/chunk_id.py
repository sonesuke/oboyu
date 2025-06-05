"""Chunk ID value object."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChunkId:
    """Immutable chunk identifier value object."""
    
    value: str
    
    def __post_init__(self) -> None:
        """Validate chunk ID format."""
        if not self.value:
            raise ValueError("Chunk ID cannot be empty")
        
        if ':' not in self.value:
            raise ValueError("Chunk ID must contain path and index separated by ':'")
        
        parts = self.value.split(':')
        if len(parts) != 2:
            raise ValueError("Chunk ID must have exactly one ':' separator")
        
        path_part, index_part = parts
        if not path_part:
            raise ValueError("Chunk ID path part cannot be empty")
        
        try:
            index = int(index_part)
            if index < 0:
                raise ValueError("Chunk index cannot be negative")
        except ValueError:
            raise ValueError("Chunk ID index part must be a non-negative integer")
    
    @classmethod
    def create(cls, document_path: Path, chunk_index: int) -> "ChunkId":
        """Create chunk ID from document path and index."""
        if chunk_index < 0:
            raise ValueError("Chunk index cannot be negative")
        
        return cls(f"{document_path}:{chunk_index}")
    
    def get_document_path(self) -> Path:
        """Extract document path from chunk ID."""
        path_part = self.value.split(':')[0]
        return Path(path_part)
    
    def get_chunk_index(self) -> int:
        """Extract chunk index from chunk ID."""
        index_part = self.value.split(':')[1]
        return int(index_part)
    
    def __str__(self) -> str:
        """Return string representation of the chunk ID."""
        return self.value
