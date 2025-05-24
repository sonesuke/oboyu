"""Tests for the hierarchical logger module."""

import time
from unittest.mock import Mock, patch

import pytest

from oboyu.cli.hierarchical_logger import (
    HierarchicalLogger,
    OperationNode,
    OperationStatus,
    create_hierarchical_logger,
)


class TestOperationNode:
    """Test the OperationNode class."""

    def test_operation_node_initialization(self):
        """Test OperationNode initialization."""
        node = OperationNode(
            id="test_1",
            description="Test operation",
            status=OperationStatus.ACTIVE,
        )
        
        assert node.id == "test_1"
        assert node.description == "Test operation"
        assert node.status == OperationStatus.ACTIVE
        assert node.children == []
        assert node.details is None
        assert node.expandable is False
        assert node.expanded is False
        assert node.metadata == {}

    def test_operation_node_duration(self):
        """Test duration calculation."""
        node = OperationNode(id="test_1", description="Test")
        
        # Active operation
        node.status = OperationStatus.ACTIVE
        time.sleep(0.1)
        duration = node.duration
        assert duration is not None
        assert duration >= 0.1
        
        # Completed operation
        node.end_time = node.start_time + 1.5
        node.status = OperationStatus.COMPLETE
        assert node.duration == 1.5

    def test_format_duration(self):
        """Test duration formatting."""
        node = OperationNode(id="test_1", description="Test")
        
        # Sub-second duration
        node.end_time = node.start_time + 0.5
        node.status = OperationStatus.COMPLETE
        assert node.format_duration() == "0.5s"
        
        # Multi-second duration
        node.end_time = node.start_time + 2.3
        assert node.format_duration() == "2.3s"


class TestHierarchicalLogger:
    """Test the HierarchicalLogger class."""

    def test_logger_initialization(self):
        """Test HierarchicalLogger initialization."""
        logger = HierarchicalLogger()
        
        assert logger.operations == []
        assert logger.operation_stack == []
        assert logger.live is None
        assert logger._id_counter == 0

    def test_start_operation(self):
        """Test starting an operation."""
        logger = HierarchicalLogger()
        
        # Start root operation
        op_id = logger.start_operation("Root operation", expandable=True, details="Details")
        
        assert len(logger.operations) == 1
        assert len(logger.operation_stack) == 1
        
        op = logger.operations[0]
        assert op.id == op_id
        assert op.description == "Root operation"
        assert op.status == OperationStatus.ACTIVE
        assert op.expandable is True
        assert op.details == "Details"
        
        # Start child operation
        child_id = logger.start_operation("Child operation")
        
        assert len(logger.operations) == 1  # Still one root
        assert len(logger.operation_stack) == 2
        assert len(op.children) == 1
        
        child = op.children[0]
        assert child.id == child_id
        assert child.description == "Child operation"

    def test_complete_operation(self):
        """Test completing an operation."""
        logger = HierarchicalLogger()
        
        op_id = logger.start_operation("Test operation")
        op = logger.operations[0]
        
        # Complete successfully
        logger.complete_operation(op_id)
        assert op.status == OperationStatus.COMPLETE
        assert op.end_time is not None
        assert len(logger.operation_stack) == 0
        
        # Complete with error
        error_id = logger.start_operation("Error operation")
        logger.complete_operation(error_id, error=True)
        error_op = logger.operations[1]
        assert error_op.status == OperationStatus.ERROR

    def test_operation_context_manager(self):
        """Test operation context manager."""
        logger = HierarchicalLogger()
        
        with logger.operation("Context operation") as op_id:
            op = logger.operations[0]
            assert op.id == op_id
            assert op.status == OperationStatus.ACTIVE
        
        # After context exit
        assert op.status == OperationStatus.COMPLETE
        assert len(logger.operation_stack) == 0

    def test_hierarchical_structure(self):
        """Test that operations maintain proper hierarchy."""
        logger = HierarchicalLogger()
        
        # Create nested structure
        root_id = logger.start_operation("Root")
        child1_id = logger.start_operation("Child 1")
        grandchild_id = logger.start_operation("Grandchild")
        logger.complete_operation(grandchild_id)
        logger.complete_operation(child1_id)
        child2_id = logger.start_operation("Child 2")
        logger.complete_operation(child2_id)
        logger.complete_operation(root_id)
        
        # Check structure
        assert len(logger.operations) == 1
        root = logger.operations[0]
        assert len(root.children) == 2
        assert root.children[0].description == "Child 1"
        assert root.children[1].description == "Child 2"
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].description == "Grandchild"

    def test_status_indicators(self):
        """Test that correct status indicators are used."""
        logger = HierarchicalLogger()
        
        # Check indicator mapping
        assert logger.STATUS_INDICATORS[OperationStatus.ACTIVE] == "⏺"
        assert logger.STATUS_INDICATORS[OperationStatus.COMPLETE] == "✓"
        assert logger.STATUS_INDICATORS[OperationStatus.ERROR] == "✗"

    def test_japanese_text_handling(self):
        """Test handling of Japanese text in operations."""
        logger = HierarchicalLogger()
        
        # Test Japanese descriptions
        op_id = logger.start_operation("インデックスを初期化しています...")
        logger.update_operation(op_id, description="初期化完了 ✓")
        
        op = logger.operations[0]
        assert op.description == "初期化完了 ✓"
        
        # Test mixed language
        child_id = logger.start_operation("Processing ファイル: 設計書.md")
        child = op.children[0]
        assert child.description == "Processing ファイル: 設計書.md"

    @patch('oboyu.cli.hierarchical_logger.Live')
    def test_live_display_context_manager(self, mock_live_class):
        """Test live display context manager."""
        # Create a mock that supports context manager protocol
        mock_live = Mock()
        mock_live.__enter__ = Mock(return_value=mock_live)
        mock_live.__exit__ = Mock(return_value=None)
        mock_live_class.return_value = mock_live
        
        logger = HierarchicalLogger()
        
        with logger.live_display():
            assert logger.live is mock_live
            mock_live.__enter__.assert_called_once()
        
        mock_live.__exit__.assert_called_once()
        assert logger.live is None


def test_create_hierarchical_logger():
    """Test the factory function."""
    logger = create_hierarchical_logger()
    assert isinstance(logger, HierarchicalLogger)
    
    # Test with custom console
    from rich.console import Console
    custom_console = Console()
    logger_with_console = create_hierarchical_logger(custom_console)
    assert logger_with_console.console is custom_console