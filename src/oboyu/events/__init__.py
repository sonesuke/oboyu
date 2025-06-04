"""Event-driven architecture for index state management."""

from .event_bus import IndexEventBus
from .events import (
    DatabaseClearedEvent,
    DatabaseClearFailedEvent,
    IndexCorruptionDetectedEvent,
    IndexEvent,
    IndexingCompletedEvent,
    IndexingFailedEvent,
    IndexingStartedEvent,
)
from .handlers.base import EventHandler

__all__ = [
    "IndexEventBus",
    "IndexEvent",
    "IndexingStartedEvent",
    "IndexingCompletedEvent",
    "IndexingFailedEvent",
    "DatabaseClearedEvent",
    "DatabaseClearFailedEvent",
    "IndexCorruptionDetectedEvent",
    "EventHandler",
]
