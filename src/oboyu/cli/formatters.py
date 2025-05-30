"""Output formatting utilities for the Oboyu CLI.

This module provides functions for formatting CLI output.
"""

import json
from typing import Any, Dict, List, Optional

from rich.box import ROUNDED
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task as ProgressTask,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from oboyu.cli.utils import format_snippet

# Create console for rich output
console = Console()


class RateColumn(ProgressColumn):
    """Renders the processing rate (items per second)."""

    def __init__(self, unit: str = "items") -> None:
        """Initialize column with unit of measurement.

        Args:
            unit: Unit name to display (default: "items")

        """
        super().__init__()
        self.unit = unit

    def render(self, task: "ProgressTask") -> Text:
        """Calculate and render the processing rate.

        Args:
            task: The task to render the rate for

        Returns:
            Formatted text with processing rate

        """
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None or elapsed == 0:
            rate = 0.0
        else:
            rate = task.completed / elapsed

        return Text(f"({rate:.1f} {self.unit}/sec)", style="dim")


def create_progress_bar(description: str, total: Optional[int] = None, unit: str = "items") -> Progress:
    """Create a progress bar with a custom format.

    Args:
        description: Description of the progress bar
        total: Total number of items to process
        unit: Unit name for rate display (e.g., "files", "chunks")

    Returns:
        Progress bar

    """
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("({task.completed}/{task.total} {task.fields[unit]})", style="dim"),
        RateColumn(unit=unit),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def create_indeterminate_progress(description: str, show_count: bool = False, unit: str = "items") -> Progress:
    """Create an indeterminate progress indicator.

    Args:
        description: Description of the progress
        show_count: Whether to show a counter of processed items
        unit: Unit name for items (e.g., "files", "chunks")

    Returns:
        Progress indicator

    """
    columns = [SpinnerColumn(), TextColumn("{task.description}")]

    if show_count:
        columns.append(TextColumn(f"({{task.completed}} {unit} processed)", style="dim"))
        columns.append(RateColumn(unit=unit))

    columns.append(TimeElapsedColumn())

    return Progress(
        *columns,
        console=console,
    )


class ProgressGroup:
    """A group of progress bars that can be updated together.

    This class helps manage multiple related progress indicators and provides
    a way to track complex multi-stage processes with a cleaner interface.
    """

    def __init__(self, description: str) -> None:
        """Initialize the progress group.

        Args:
            description: Description of the overall process

        """
        self.description = description
        self.progress_bars: Dict[str, Progress] = {}
        self.tasks: Dict[str, TaskID] = {}
        self.active_progress: Optional[str] = None

    def add_progress(
        self,
        name: str,
        description: str,
        total: Optional[int] = None,
        unit: str = "items",
        indeterminate: bool = False,
        show_count: bool = True,
    ) -> None:
        """Add a new progress bar to the group.

        Args:
            name: Unique name for this progress bar
            description: Description for the progress bar
            total: Total number of items to process
            unit: Unit name for rate display
            indeterminate: Whether this is an indeterminate progress
            show_count: For indeterminate progress, whether to show count

        """
        if indeterminate:
            progress = create_indeterminate_progress(description, show_count, unit)
            self.progress_bars[name] = progress
            self.tasks[name] = progress.add_task(description, total=None)
        else:
            progress = create_progress_bar(description, total, unit)
            self.progress_bars[name] = progress
            self.tasks[name] = progress.add_task(description, total=total, unit=unit)

    def start(self, name: str) -> None:
        """Start displaying a specific progress bar.

        Args:
            name: Name of the progress bar to start

        """
        if name in self.progress_bars:
            self.active_progress = name
            self.progress_bars[name].start()

    def update(
        self,
        name: str,
        advance: Optional[int] = None,
        completed: Optional[int] = None,
        description: Optional[str] = None,
    ) -> None:
        """Update a specific progress bar.

        Args:
            name: Name of the progress bar to update
            advance: Number of items to advance by
            completed: Set the completed count directly
            description: New description for the task
            **kwargs: Additional fields to update

        """
        if name in self.progress_bars and name in self.tasks:
            # Update progress bar attributes
            if advance is not None:
                self.progress_bars[name].update(self.tasks[name], advance=advance)
            if completed is not None:
                self.progress_bars[name].update(self.tasks[name], completed=completed)
            if description is not None:
                self.progress_bars[name].update(self.tasks[name], description=description)

    def stop(self, name: Optional[str] = None) -> None:
        """Stop displaying a specific progress bar or the active one.

        Args:
            name: Name of the progress bar to stop (defaults to active)

        """
        target = name if name is not None else self.active_progress
        if target and target in self.progress_bars:
            self.progress_bars[target].stop()
            if target == self.active_progress:
                self.active_progress = None

    def stop_all(self) -> None:
        """Stop all progress bars in the group."""
        for name in self.progress_bars:
            self.progress_bars[name].stop()
        self.active_progress = None


def format_table(headers: List[str], rows: List[List[Any]]) -> Table:
    """Format data as a table.

    Args:
        headers: Column headers
        rows: Table rows

    Returns:
        Formatted table

    """
    table = Table(box=ROUNDED)

    # Add headers
    for header in headers:
        table.add_column(header)

    # Add rows
    for row in rows:
        table.add_row(*[str(cell) for cell in row])

    return table


def format_json(data: object) -> Syntax:
    """Format data as JSON.

    Args:
        data: Data to format

    Returns:
        Formatted JSON

    """
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    return Syntax(json_str, "json", theme="monokai", line_numbers=True)


def format_search_results_text(
    results: List[Dict[str, Any]],
    query: str,
    snippet_length: int = 160,
    highlight: bool = True,
) -> None:
    """Format search results as text.

    Args:
        results: Search results
        query: Search query
        snippet_length: Maximum snippet length
        highlight: Whether to highlight matches

    """
    if not results:
        console.print("No results found.")
        return

    console.print(f'\nResults for: "{query}"\n')

    # Display each result
    for result in results:
        title = result.get("title", "Untitled")
        content = result.get("content", "")
        path = result.get("path", "")
        score = result.get("score", 0.0)

        # Create snippet
        snippet = format_snippet(content, query, snippet_length, highlight)

        # Create structured output
        console.print(f"• {title} (Score: {score:.2f})")
        console.print(f"  {snippet}")
        console.print(f"  Source: {path}")
        console.print("")

    console.print(f"\nRetrieved {len(results)} documents")


def format_search_results_json(results: List[Dict[str, Any]]) -> None:
    """Format search results as JSON.

    Args:
        results: Search results

    """
    # Convert to JSON-serializable format
    json_results = []

    for result in results:
        json_result = {
            "title": result.get("title", ""),
            "snippet": result.get("content", "")[:200] + "...",
            "path": str(result.get("path", "")),
            "score": result.get("score", 0.0),
            "language": result.get("language", ""),
        }
        json_results.append(json_result)

    # Print as JSON
    console.print(format_json(json_results))
