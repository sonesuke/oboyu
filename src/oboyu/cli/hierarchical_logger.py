"""Hierarchical log-style TUI for Oboyu CLI.

This module provides a hierarchical logging interface with tree-style
formatting as described in issue #33.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

from rich.console import Console
from rich.live import Live
from rich.text import Text


class OperationStatus(Enum):
    """Status of an operation in the hierarchy."""

    ACTIVE = "active"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class OperationNode:
    """Represents a single operation in the hierarchical log."""

    id: str
    description: str
    status: OperationStatus = OperationStatus.ACTIVE
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    children: List["OperationNode"] = field(default_factory=list)
    details: Optional[str] = None
    expandable: bool = False
    expanded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    indent_level: int = 0
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate operation duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        elif self.status == OperationStatus.ACTIVE:
            return time.time() - self.start_time
        return None
    
    def format_duration(self) -> str:
        """Format duration for display."""
        duration = self.duration
        if duration is None:
            return ""
        if duration < 1:
            return f"{duration:.1f}s"
        return f"{duration:.1f}s"


class HierarchicalLogger:
    """A hierarchical logging system with tree-style output matching issue #33."""
    
    # Status indicators as per issue #33
    STATUS_INDICATORS = {
        OperationStatus.ACTIVE: "⏺",
        OperationStatus.COMPLETE: "✓",
        OperationStatus.ERROR: "✗",
    }
    
    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialize the hierarchical logger.
        
        Args:
            console: Rich console instance to use for output

        """
        self.console = console or Console()
        self.operations: List[OperationNode] = []
        self.operation_stack: List[OperationNode] = []
        self.live: Optional[Live] = None
        self._id_counter = 0
    
    def _generate_id(self) -> str:
        """Generate a unique operation ID."""
        self._id_counter += 1
        return f"op_{self._id_counter}"
    
    def start_operation(
        self,
        description: str,
        expandable: bool = False,
        details: Optional[str] = None,
        **metadata: Any  # noqa: ANN401
    ) -> str:
        """Start a new operation.
        
        Args:
            description: Description of the operation
            expandable: Whether this operation can be expanded for details
            details: Additional details (shown when expanded)
            **metadata: Additional metadata for the operation
            
        Returns:
            Operation ID for tracking

        """
        op_id = self._generate_id()
        
        # Determine indent level based on stack
        indent_level = len(self.operation_stack)
        
        operation = OperationNode(
            id=op_id,
            description=description,
            status=OperationStatus.ACTIVE,
            expandable=expandable,
            details=details,
            metadata=metadata,
            indent_level=indent_level,
        )
        
        if self.operation_stack:
            # Add as child of current operation
            self.operation_stack[-1].children.append(operation)
        else:
            # Add as root operation
            self.operations.append(operation)
        
        self.operation_stack.append(operation)
        self._refresh_display()
        return op_id
    
    def update_operation(
        self,
        op_id: str,
        description: Optional[str] = None,
        details: Optional[str] = None,
        **metadata: Any  # noqa: ANN401
    ) -> None:
        """Update an existing operation.
        
        Args:
            op_id: Operation ID to update
            description: New description
            details: New details
            **metadata: Additional metadata to update

        """
        operation = self._find_operation(op_id)
        if operation:
            if description:
                operation.description = description
            if details:
                operation.details = details
            operation.metadata.update(metadata)
            self._refresh_display()
    
    def complete_operation(self, op_id: Optional[str] = None, error: bool = False) -> None:
        """Mark an operation as complete.
        
        Args:
            op_id: Operation ID to complete (None for current)
            error: Whether the operation ended in error

        """
        if op_id:
            operation = self._find_operation(op_id)
        else:
            operation = self.operation_stack[-1] if self.operation_stack else None
        
        if operation:
            operation.status = OperationStatus.ERROR if error else OperationStatus.COMPLETE
            operation.end_time = time.time()
            
            # Remove from stack if it's the current operation
            if self.operation_stack and self.operation_stack[-1].id == operation.id:
                self.operation_stack.pop()
            
            self._refresh_display()
    
    def _find_operation(self, op_id: str) -> Optional[OperationNode]:
        """Find an operation by ID."""
        def search(nodes: List[OperationNode]) -> Optional[OperationNode]:
            for node in nodes:
                if node.id == op_id:
                    return node
                found = search(node.children)
                if found:
                    return found
            return None
        
        return search(self.operations)
    
    def _render_operation(self, operation: OperationNode, is_child: bool = False) -> List[Text]:
        """Render an operation and its children in hierarchical style."""
        lines = []
        
        # Build the line
        line = Text()
        
        # Add indentation for children
        if is_child:
            line.append("  ⎿ ")
        
        # Add status indicator
        indicator = self.STATUS_INDICATORS.get(operation.status, "⏺")
        color = "green" if operation.status == OperationStatus.COMPLETE else "yellow" if operation.status == OperationStatus.ACTIVE else "red"
        line.append(f"{indicator} ", style=color)
        
        # Add description
        line.append(operation.description)
        
        # Add duration for completed operations
        if operation.status == OperationStatus.COMPLETE and operation.duration:
            line.append(f" ({operation.format_duration()})", style="dim")
        
        # Add expandable hint
        if operation.expandable and not operation.expanded:
            line.append(" (ctrl+r to expand)", style="dim italic")
        
        lines.append(line)
        
        # Render children with proper indentation
        for child in operation.children:
            child_lines = self._render_operation(child, is_child=True)
            for child_line in child_lines:
                # Add extra indentation for nested children
                if operation.indent_level > 0:
                    indented_line = Text("  " * operation.indent_level)
                    indented_line.append(child_line)
                    lines.append(indented_line)
                else:
                    lines.append(child_line)
        
        return lines
    
    def _refresh_display(self) -> None:
        """Refresh the display with current state."""
        if self.live:
            output = Text()
            for i, operation in enumerate(self.operations):
                if i > 0:
                    output.append("\n")
                op_lines = self._render_operation(operation)
                for line in op_lines:
                    output.append(line)
                    output.append("\n")
            self.live.update(output)
    
    @contextmanager
    def live_display(self) -> Iterator["HierarchicalLogger"]:
        """Context manager for live updating display."""
        self.live = Live(
            "",
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        with self.live:
            self._refresh_display()
            yield self
        self.live = None
    
    @contextmanager
    def operation(
        self,
        description: str,
        expandable: bool = False,
        details: Optional[str] = None,
        **metadata: Any  # noqa: ANN401
    ) -> Iterator[str]:
        """Context manager for an operation."""
        op_id = self.start_operation(description, expandable, details, **metadata)
        try:
            yield op_id
            self.complete_operation(op_id)
        except Exception:
            self.complete_operation(op_id, error=True)
            raise


def create_hierarchical_logger(console: Optional[Console] = None) -> HierarchicalLogger:
    """Create a new hierarchical logger instance.
    
    Args:
        console: Rich console to use for output
        
    Returns:
        HierarchicalLogger instance

    """
    return HierarchicalLogger(console)
