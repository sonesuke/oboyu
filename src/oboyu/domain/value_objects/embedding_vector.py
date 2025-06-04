"""Embedding vector value object."""

import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class EmbeddingVector:
    """Immutable embedding vector value object."""
    
    values: tuple[float, ...]
    dimensions: int
    
    def __post_init__(self) -> None:
        """Validate vector properties."""
        if not self.values:
            raise ValueError("Embedding vector cannot be empty")
        
        if self.dimensions <= 0:
            raise ValueError("Dimensions must be positive")
        
        if len(self.values) != self.dimensions:
            raise ValueError(f"Vector length {len(self.values)} does not match dimensions {self.dimensions}")
        
        for i, value in enumerate(self.values):
            if not isinstance(value, (int, float)) or value != value:  # NaN check
                raise ValueError(f"Invalid vector value at position {i}: {value}")
    
    @classmethod
    def create(cls, values: List[float]) -> "EmbeddingVector":
        """Create embedding vector from list of values."""
        if not values:
            raise ValueError("Vector values cannot be empty")
        
        return cls(tuple(values), len(values))
    
    def normalize(self) -> "EmbeddingVector":
        """Return normalized vector."""
        magnitude = self.get_magnitude()
        if magnitude == 0:
            raise ValueError("Cannot normalize zero vector")
        
        normalized_values = tuple(x / magnitude for x in self.values)
        return EmbeddingVector(normalized_values, self.dimensions)
    
    def get_magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(sum(x * x for x in self.values))
    
    def dot_product(self, other: "EmbeddingVector") -> float:
        """Calculate dot product with another vector."""
        if self.dimensions != other.dimensions:
            raise ValueError("Vectors must have same dimensions for dot product")
        
        return sum(a * b for a, b in zip(self.values, other.values))
    
    def cosine_similarity(self, other: "EmbeddingVector") -> float:
        """Calculate cosine similarity with another vector."""
        if self.dimensions != other.dimensions:
            raise ValueError("Vectors must have same dimensions for cosine similarity")
        
        dot_prod = self.dot_product(other)
        magnitude_product = self.get_magnitude() * other.get_magnitude()
        
        if magnitude_product == 0:
            return 0.0
        
        return dot_prod / magnitude_product
    
    def to_list(self) -> List[float]:
        """Convert to list of floats."""
        return list(self.values)
