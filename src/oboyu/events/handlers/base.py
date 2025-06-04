"""Base event handler interface."""

from abc import ABC, abstractmethod

from ..events import IndexEvent


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    def handle(self, event: IndexEvent) -> None:
        """Handle an index event.
        
        Args:
            event: The event to handle

        """
        pass
