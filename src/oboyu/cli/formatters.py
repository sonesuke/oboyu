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
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax
from rich.table import Table

from oboyu.cli.utils import format_snippet

# Create console for rich output
console = Console()


def create_progress_bar(description: str, total: Optional[int] = None) -> Progress:
    """Create a progress bar with a custom format.

    Args:
        description: Description of the progress bar
        total: Total number of items to process

    Returns:
        Progress bar

    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("[dim]({task.completed}/{task.total})[/dim]"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def create_indeterminate_progress(description: str, show_count: bool = False) -> Progress:
    """Create an indeterminate progress indicator.

    Args:
        description: Description of the progress
        show_count: Whether to show a counter of processed items

    Returns:
        Progress indicator

    """
    columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}")
    ]

    if show_count:
        columns.append(TextColumn("[dim]({task.completed} items processed)[/dim]"))

    columns.append(TimeElapsedColumn())

    return Progress(
        *columns,
        console=console,
    )


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
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"Results for: \"[bold]{query}[/bold]\"")
    console.print("----------------------------------------")
    console.print("")

    # Display each result
    for result in results:
        title = result.get("title", "Untitled")
        content = result.get("content", "")
        path = result.get("path", "")
        score = result.get("score", 0.0)

        # Create snippet
        snippet = format_snippet(content, query, snippet_length, highlight)

        # Create structured output
        console.print(f"• [bold cyan]{title}[/bold cyan] [dim](Score: {score:.2f})[/dim]")
        console.print(f"  {snippet}")
        console.print(f"  Source: [blue underline]{path}[/blue underline]")
        console.print("")

    console.print("----------------------------------------")
    console.print(f"Retrieved {len(results)} documents")


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
