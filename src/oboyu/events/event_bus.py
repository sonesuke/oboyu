"""Core event bus implementation for index state management."""

import logging
from typing import Dict, List

from .events import IndexEvent
from .handlers.base import EventHandler

logger = logging.getLogger(__name__)


class IndexEventBus:
    """Event bus for publishing and handling index-related events."""
    
    def __init__(self) -> None:
        """Initialize the event bus."""
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
    
    def publish(self, event: IndexEvent) -> None:
        """Publish an event to all registered handlers.
        
        Args:
            event: The event to publish

        """
        logger.debug(f"Publishing event: {event.event_type}")
        
        # Handle by specific event type handlers
        for handler in self._handlers.get(event.event_type, []):
            try:
                handler.handle(event)
            except Exception as e:
                logger.error(
                    f"Event handler failed for {event.event_type}: {e}",
                    exc_info=True
                )
        
        # Handle by global handlers (listen to all events)
        for handler in self._global_handlers:
            try:
                handler.handle(event)
            except Exception as e:
                logger.error(
                    f"Global event handler failed for {event.event_type}: {e}",
                    exc_info=True
                )
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe a handler to a specific event type.
        
        Args:
            event_type: The type of event to listen for
            handler: The handler to register

        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed handler {handler.__class__.__name__} to {event_type}")
    
    def subscribe_global(self, handler: EventHandler) -> None:
        """Subscribe a handler to all events.
        
        Args:
            handler: The handler to register for all events

        """
        self._global_handlers.append(handler)
        logger.debug(f"Subscribed global handler {handler.__class__.__name__}")
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe a handler from a specific event type.
        
        Args:
            event_type: The event type to unsubscribe from
            handler: The handler to remove

        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler {handler.__class__.__name__} from {event_type}")
            except ValueError:
                logger.warning(f"Handler {handler.__class__.__name__} not found for {event_type}")
    
    def unsubscribe_global(self, handler: EventHandler) -> None:
        """Unsubscribe a global handler.
        
        Args:
            handler: The handler to remove from global handlers

        """
        try:
            self._global_handlers.remove(handler)
            logger.debug(f"Unsubscribed global handler {handler.__class__.__name__}")
        except ValueError:
            logger.warning(f"Global handler {handler.__class__.__name__} not found")
    
    def get_handlers(self, event_type: str) -> List[EventHandler]:
        """Get all handlers for a specific event type.
        
        Args:
            event_type: The event type to get handlers for
            
        Returns:
            List of handlers for the event type

        """
        return list(self._handlers.get(event_type, []))
    
    def get_global_handlers(self) -> List[EventHandler]:
        """Get all global handlers.
        
        Returns:
            List of global handlers

        """
        return list(self._global_handlers)
    
    def clear_handlers(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._global_handlers.clear()
        logger.debug("Cleared all event handlers")
