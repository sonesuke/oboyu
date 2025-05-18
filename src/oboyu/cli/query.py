"""Query command implementation for Oboyu CLI.

This module provides the command-line interface for querying indexed documents.
"""

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
from typing_extensions import Annotated

from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.indexer import Indexer, SearchResult

# Create Typer app
app = typer.Typer(help="Search indexed documents")

# Create console for rich output
console = Console()

# Define command options
QueryOption = Annotated[
    str,
    typer.Argument(
        help="Search query text",
    ),
]

ModeOption = Annotated[
    str,
    typer.Option(
        "--mode",
        "-m",
        help="Search mode (vector, bm25, hybrid)",
    ),
]

TopKOption = Annotated[
    Optional[int],
    typer.Option(
        "--top-k",
        "-k",
        help="Number of results to return",
        min=1,
    ),
]

ExplainOption = Annotated[
    bool,
    typer.Option(
        "--explain",
        "-e",
        help="Show detailed match explanation",
    ),
]

FormatOption = Annotated[
    str,
    typer.Option(
        "--format",
        "-f",
        help="Output format (text, json)",
    ),
]

VectorWeightOption = Annotated[
    Optional[float],
    typer.Option(
        "--vector-weight",
        help="Weight for vector scores in hybrid search",
        min=0.0,
        max=1.0,
    ),
]

BM25WeightOption = Annotated[
    Optional[float],
    typer.Option(
        "--bm25-weight",
        help="Weight for BM25 scores in hybrid search",
        min=0.0,
        max=1.0,
    ),
]

DatabasePathOption = Annotated[
    Optional[Path],
    typer.Option(
        "--db-path",
        help="Path to database file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
]


def format_search_result(result: SearchResult, query: str, show_explanation: bool = False) -> Panel:
    """Format a search result for display.

    Args:
        result: Search result to format
        query: Original search query
        show_explanation: Whether to show detailed explanation

    Returns:
        Formatted panel for display

    """
    # Create title
    title = Text()
    title.append(f"{result.title}", style="bold cyan")

    # Create content
    content = Text()

    # Add score
    content.append(f"Score: {result.score:.4f}", style="bold green")
    content.append("\n\n")

    # Add snippet
    content.append(escape(result.content[:200] + "..."), style="")

    # Add metadata
    content.append("\n\n")
    content.append("Path: ", style="dim")
    content.append(str(result.path), style="blue underline")

    # Add language
    content.append("\nLanguage: ", style="dim")
    content.append(result.language, style="")

    # Add explanation if requested
    if show_explanation:
        content.append("\n\nExplanation:\n", style="dim")
        content.append(f"Chunk ID: {result.chunk_id}\n", style="")
        content.append(f"Chunk Index: {result.chunk_index}\n", style="")
        # Add more explanation here in the future

    # Create panel
    return Panel(content, title=title, border_style="blue")


@app.callback(invoke_without_command=True)
def query(
    ctx: typer.Context,
    query: QueryOption,
    mode: ModeOption = "hybrid",
    top_k: TopKOption = None,
    explain: ExplainOption = False,
    format: FormatOption = "text",
    vector_weight: VectorWeightOption = None,
    bm25_weight: BM25WeightOption = None,
    db_path: DatabasePathOption = None,
) -> None:
    """Search indexed documents.

    This command searches the index for documents matching the query.
    """
    # Get global options from context
    config_data = ctx.obj.get("config_data", {}) if ctx.obj else {}

    # Get query engine configuration
    query_config = config_data.get("query", {})

    # Override with command-line options
    if top_k is not None:
        query_config["top_k"] = top_k
    else:
        top_k = query_config.get("top_k", 5)

    if vector_weight is not None:
        query_config["vector_weight"] = vector_weight

    if bm25_weight is not None:
        query_config["bm25_weight"] = bm25_weight

    # Create indexer configuration
    indexer_config_dict = config_data.get("indexer", {})

    # Override with command-line options
    if db_path is not None:
        indexer_config_dict["db_path"] = str(db_path)

    # Create indexer configuration object
    indexer_config = IndexerConfig(config_dict={"indexer": indexer_config_dict})

    # Create indexer
    indexer = Indexer(config=indexer_config)

    # Perform search
    start_time = time.time()
    results = indexer.search(query, limit=top_k)
    elapsed_time = time.time() - start_time

    # Display results
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"[bold green]Found {len(results)} results in {elapsed_time:.2f} seconds:[/bold green]\n")

    # Format and display each result
    for i, result in enumerate(results, start=1):
        console.print(f"[bold]Result {i}:[/bold]")
        panel = format_search_result(result, query, show_explanation=explain)
        console.print(panel)
        console.print("")  # Add spacing between results
