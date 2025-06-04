"""Tests for the event bus implementation."""

import pytest
from unittest.mock import Mock

from oboyu.events.event_bus import IndexEventBus
from oboyu.events.events import IndexingStartedEvent, IndexingCompletedEvent
from oboyu.events.handlers.base import EventHandler


class MockEventHandler(EventHandler):
    """Mock event handler for testing."""
    
    def __init__(self):
        self.handled_events = []
        self.handle_exception = None
    
    def handle(self, event):
        if self.handle_exception:
            raise self.handle_exception
        self.handled_events.append(event)


class TestIndexEventBus:
    """Test cases for IndexEventBus."""
    
    def test_event_bus_initialization(self):
        """Test event bus initializes correctly."""
        event_bus = IndexEventBus()
        
        assert len(event_bus._handlers) == 0
        assert len(event_bus._global_handlers) == 0
    
    def test_subscribe_handler(self):
        """Test subscribing a handler to specific event type."""
        event_bus = IndexEventBus()
        handler = MockEventHandler()
        
        event_bus.subscribe("indexing_started", handler)
        
        assert len(event_bus._handlers["indexing_started"]) == 1
        assert handler in event_bus._handlers["indexing_started"]
    
    def test_subscribe_global_handler(self):
        """Test subscribing a global handler."""
        event_bus = IndexEventBus()
        handler = MockEventHandler()
        
        event_bus.subscribe_global(handler)
        
        assert len(event_bus._global_handlers) == 1
        assert handler in event_bus._global_handlers
    
    def test_publish_event_to_specific_handlers(self):
        """Test publishing event to specific handlers."""
        event_bus = IndexEventBus()
        handler1 = MockEventHandler()
        handler2 = MockEventHandler()
        
        event_bus.subscribe("indexing_started", handler1)
        event_bus.subscribe("indexing_completed", handler2)
        
        event = IndexingStartedEvent(document_count=5)
        event_bus.publish(event)
        
        assert len(handler1.handled_events) == 1
        assert len(handler2.handled_events) == 0
        assert handler1.handled_events[0] == event
    
    def test_publish_event_to_global_handlers(self):
        """Test publishing event to global handlers."""
        event_bus = IndexEventBus()
        global_handler = MockEventHandler()
        specific_handler = MockEventHandler()
        
        event_bus.subscribe_global(global_handler)
        event_bus.subscribe("indexing_started", specific_handler)
        
        event = IndexingStartedEvent(document_count=5)
        event_bus.publish(event)
        
        assert len(global_handler.handled_events) == 1
        assert len(specific_handler.handled_events) == 1
        assert global_handler.handled_events[0] == event
        assert specific_handler.handled_events[0] == event
    
    def test_handler_exception_handling(self):
        """Test that handler exceptions don't break event publishing."""
        event_bus = IndexEventBus()
        failing_handler = MockEventHandler()
        succeeding_handler = MockEventHandler()
        
        failing_handler.handle_exception = Exception("Test exception")
        
        event_bus.subscribe("indexing_started", failing_handler)
        event_bus.subscribe("indexing_started", succeeding_handler)
        
        event = IndexingStartedEvent(document_count=5)
        
        # Should not raise exception
        event_bus.publish(event)
        
        # Succeeding handler should still get the event
        assert len(succeeding_handler.handled_events) == 1
        assert succeeding_handler.handled_events[0] == event
    
    def test_unsubscribe_handler(self):
        """Test unsubscribing a handler."""
        event_bus = IndexEventBus()
        handler = MockEventHandler()
        
        event_bus.subscribe("indexing_started", handler)
        assert len(event_bus._handlers["indexing_started"]) == 1
        
        event_bus.unsubscribe("indexing_started", handler)
        assert len(event_bus._handlers["indexing_started"]) == 0
    
    def test_unsubscribe_global_handler(self):
        """Test unsubscribing a global handler."""
        event_bus = IndexEventBus()
        handler = MockEventHandler()
        
        event_bus.subscribe_global(handler)
        assert len(event_bus._global_handlers) == 1
        
        event_bus.unsubscribe_global(handler)
        assert len(event_bus._global_handlers) == 0
    
    def test_unsubscribe_nonexistent_handler(self):
        """Test unsubscribing a handler that doesn't exist."""
        event_bus = IndexEventBus()
        handler = MockEventHandler()
        
        # Should not raise exception
        event_bus.unsubscribe("indexing_started", handler)
        event_bus.unsubscribe_global(handler)
    
    def test_get_handlers(self):
        """Test getting handlers for an event type."""
        event_bus = IndexEventBus()
        handler1 = MockEventHandler()
        handler2 = MockEventHandler()
        
        event_bus.subscribe("indexing_started", handler1)
        event_bus.subscribe("indexing_started", handler2)
        
        handlers = event_bus.get_handlers("indexing_started")
        
        assert len(handlers) == 2
        assert handler1 in handlers
        assert handler2 in handlers
    
    def test_get_global_handlers(self):
        """Test getting global handlers."""
        event_bus = IndexEventBus()
        handler1 = MockEventHandler()
        handler2 = MockEventHandler()
        
        event_bus.subscribe_global(handler1)
        event_bus.subscribe_global(handler2)
        
        handlers = event_bus.get_global_handlers()
        
        assert len(handlers) == 2
        assert handler1 in handlers
        assert handler2 in handlers
    
    def test_clear_handlers(self):
        """Test clearing all handlers."""
        event_bus = IndexEventBus()
        handler1 = MockEventHandler()
        handler2 = MockEventHandler()
        
        event_bus.subscribe("indexing_started", handler1)
        event_bus.subscribe_global(handler2)
        
        event_bus.clear_handlers()
        
        assert len(event_bus._handlers) == 0
        assert len(event_bus._global_handlers) == 0