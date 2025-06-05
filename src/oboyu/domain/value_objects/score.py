"""Score value object for search results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Score:
    """Immutable score value object for search results."""
    
    value: float
    
    def __post_init__(self) -> None:
        """Validate score value."""
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Score must be between 0.0 and 1.0")
        
        if self.value != self.value:  # NaN check
            raise ValueError("Score cannot be NaN")
    
    @classmethod
    def create(cls, value: float) -> "Score":
        """Create score with normalization."""
        if value < 0.0:
            value = 0.0
        elif value > 1.0:
            value = 1.0
        
        return cls(round(value, 6))
    
    def is_high(self) -> bool:
        """Check if this is a high score (>= 0.7)."""
        return self.value >= 0.7
    
    def is_medium(self) -> bool:
        """Check if this is a medium score (0.3 - 0.7)."""
        return 0.3 <= self.value < 0.7
    
    def is_low(self) -> bool:
        """Check if this is a low score (< 0.3)."""
        return self.value < 0.3
    
    def meets_threshold(self, threshold: float) -> bool:
        """Check if score meets or exceeds threshold."""
        return self.value >= threshold
    
    def __float__(self) -> float:
        """Return float representation of the score."""
        return self.value
    
    def __str__(self) -> str:
        """Return string representation of the score."""
        return f"{self.value:.3f}"
