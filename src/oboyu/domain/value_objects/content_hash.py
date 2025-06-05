"""Content hash value object."""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ContentHash:
    """Immutable content hash value object."""
    
    value: str
    
    def __post_init__(self) -> None:
        """Validate hash format."""
        if not self.value:
            raise ValueError("Content hash cannot be empty")
        
        if len(self.value) != 64:
            raise ValueError("Content hash must be 64 characters long (SHA-256)")
        
        if not all(c in '0123456789abcdef' for c in self.value.lower()):
            raise ValueError("Content hash must be valid hexadecimal")
    
    @classmethod
    def from_content(cls, content: str) -> "ContentHash":
        """Create hash from content."""
        hash_value = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return cls(hash_value)
    
    def matches_content(self, content: str) -> bool:
        """Check if hash matches given content."""
        expected_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return self.value == expected_hash
    
    def __str__(self) -> str:
        """Return string representation of the content hash."""
        return self.value
