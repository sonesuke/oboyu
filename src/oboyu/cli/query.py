"""Query command implementation for Oboyu CLI.

This module provides the command-line interface for querying indexed documents.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

# Disable tokenizer parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from oboyu.cli.base import BaseCommand
from oboyu.cli.commands.query import QueryCommand
from oboyu.common.types import SearchResult

# Create Typer app
app = typer.Typer(
    help="Search indexed documents",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)

# Create console for rich output
console = Console()

# Define command-specific options not in common_options
QueryOption = Annotated[
    Optional[str],
    typer.Argument(
        help="Search query text",
    ),
]


@app.callback(invoke_without_command=True)
def query(
    ctx: typer.Context,
    query: Optional[str] = None,
    mode: str = "hybrid",
    top_k: Optional[int] = None,
    explain: bool = False,
    format: str = "text",
    rrf_k: Optional[int] = None,
    db_path: Optional[Path] = None,
    rerank: Optional[bool] = None,
) -> None:
    """Search indexed documents.

    This command searches the index for documents matching the query.
    """
    # Create base command for common functionality
    base_command = BaseCommand(ctx)

    # Validate query argument
    if query is None:
        base_command.console.print("❌ Error: Query argument is required", style="red")
        raise typer.Exit(1)

    # Get configuration manager and query service
    config_manager = base_command.get_config_manager()
    query_service = QueryCommand(config_manager)

    try:
        # Execute single query using service
        # At this point, query is guaranteed to be non-None due to validation above
        assert query is not None, "Query should be validated as non-None by this point"

        result = query_service.execute_query_with_context(
            query=query,
            mode=mode,
            top_k=top_k,
            rrf_k=rrf_k,
            db_path=db_path,
            rerank=rerank,
        )

        # Display results
        _display_results(base_command.console, result.results, result.elapsed_time, result.mode, explain, format, result.reranker_used)

    except Exception as e:
        base_command.console.print(f"❌ Search failed: {e}", style="red")
        logging.error(f"Search error: {e}")
        raise typer.Exit(1)


def _display_results(
    console: Console,
    results: list[SearchResult],
    elapsed_time: float,
    mode: str,
    explain: bool,
    format: str,
    reranker_used: bool = False,
) -> None:
    """Display search results."""
    if not results:
        if format == "json":
            # Output empty JSON structure for no results
            json_output = {"results": [], "count": 0, "search_type": f"{mode}{' with reranker' if reranker_used else ''}", "duration": elapsed_time}
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
        else:
            console.print("❌ No results found.")
        return

    if format == "json":
        # Convert results to JSON format
        json_results = []
        for result in results:
            # Create snippet (first 200 chars)
            content = result.content[:200].replace("\n", " ").strip()
            if len(result.content) > 200:
                content += "..."

            json_result = {
                "score": result.score,
                "file_path": str(result.path),
                "title": result.title or "",
                "snippet": content,
                "language": getattr(result, "language", "en"),  # Default to 'en' if not available
            }

            # Add chunk index if explain mode is enabled
            if explain:
                json_result["chunk_index"] = result.chunk_index

            json_results.append(json_result)

        # Create final JSON output structure
        json_output = {
            "results": json_results,
            "count": len(results),
            "search_type": f"{mode}{' with reranker' if reranker_used else ''}",
            "duration": elapsed_time,
        }

        # Output JSON using print to avoid Rich formatting
        print(json.dumps(json_output, indent=2, ensure_ascii=False))
    else:
        # Original text format
        # Header with reranker indication
        reranker_suffix = " with reranker" if reranker_used else ""
        console.print(f"\n🎯 Found [bold green]{len(results)}[/bold green] results ([dim]{mode} search{reranker_suffix}, {elapsed_time:.3f}s[/dim])\n")

        # Display results
        for i, result in enumerate(results, 1):
            # Score color coding
            score = result.score
            if score >= 0.8:
                score_color = "bright_green"
            elif score >= 0.6:
                score_color = "green"
            elif score >= 0.4:
                score_color = "yellow"
            else:
                score_color = "red"

            # Display result
            console.print(f"[bold blue]{i:2d}.[/bold blue] [{score_color}]{score:.3f}[/{score_color}] [dim]{result.path}[/dim]")

            if result.title:
                console.print(f"    [bold]{result.title}[/bold]")

            # Content preview
            content = result.content[:200].replace("\n", " ").strip()
            if len(result.content) > 200:
                content += "..."
            console.print(f"    {content}")

            if explain:
                console.print(f"    [dim]Chunk index: {result.chunk_index}[/dim]")

            console.print()  # Empty line between results
