"""Query command implementation for Oboyu CLI.

This module provides the command-line interface for querying indexed documents.
"""

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.text import Text
from typing_extensions import Annotated

from oboyu.common.paths import DEFAULT_DB_PATH
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


def format_search_result(result: SearchResult, query: str, show_explanation: bool = False) -> Text:
    """Format a search result for display using cleaner hierarchical format.

    Args:
        result: Search result to format
        query: Original search query
        show_explanation: Whether to show detailed explanation

    Returns:
        Formatted text for display

    """
    from oboyu.cli.utils import format_snippet

    # Create content with structured hierarchical format
    content = Text()

    # Title with score
    content.append("• ", style="bold")
    content.append(f"{result.title}", style="bold cyan")
    content.append(f" (Score: {result.score:.2f})", style="dim")
    content.append("\n")

    # Snippet with better formatting
    snippet = format_snippet(result.content, query, 200, True)
    content.append(f"  {snippet}\n", style="")

    # Path as source
    content.append("  Source: ", style="dim")
    content.append(str(result.path), style="blue underline")

    # Language as metadata if available and not empty
    if result.language and result.language.strip():
        content.append(f" ({result.language})", style="dim")

    # Add explanation if requested
    if show_explanation:
        content.append("\n  [Explanation] ", style="dim")
        content.append(f"Chunk ID: {result.chunk_id}, ", style="")
        content.append(f"Index: {result.chunk_index}", style="")
        # Add more explanation here in the future

    return content


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

    # Handle database path explicitly, with clear precedence:
    # 1. Command-line option (highest priority)
    # 2. Config file value
    # 3. Default from central path definition (lowest priority)
    if db_path is not None:
        indexer_config_dict["db_path"] = str(db_path)
        console.print(f"Using explicitly specified database path: [cyan]{db_path}[/cyan]")
    elif "db_path" in indexer_config_dict:
        console.print(f"Using configured database path: [cyan]{indexer_config_dict['db_path']}[/cyan]")
    else:
        # Use the default path from central definition
        indexer_config_dict["db_path"] = str(DEFAULT_DB_PATH)
        console.print(f"Using default database path: [cyan]{DEFAULT_DB_PATH}[/cyan]")

    # Create indexer configuration object
    indexer_config = IndexerConfig(config_dict={"indexer": indexer_config_dict})

    # Provide immediate feedback
    console.print(f"Searching for: \"[bold]{query}[/bold]\"")
    console.print(f"Search mode: [cyan]{mode}[/cyan]")

    # Import progress indicator
    from oboyu.cli.formatters import create_indeterminate_progress

    # Show progress during indexer initialization (model loading and database setup)
    with create_indeterminate_progress("Initializing...") as init_progress:
        init_task = init_progress.add_task("Loading embedding model and setting up database...", total=None)

        # Create indexer (this loads the model and sets up the database)
        indexer = Indexer(config=indexer_config)

        # Mark initialization as complete
        init_progress.update(init_task, description="[green]✓[/green] Initialization complete")

    start_time = time.time()
    with create_indeterminate_progress("Searching database...") as progress:
        progress.add_task("Searching...", total=None)
        results = indexer.search(query, limit=top_k)

    elapsed_time = time.time() - start_time

    # Display results
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    # Display results header with divider
    console.print("\nResults for: \"[bold]{query}[/bold]\"")
    console.print("----------------------------------------")

    # Format and display each result
    for result in results:
        formatted_result = format_search_result(result, query, show_explanation=explain)
        console.print(formatted_result)
        console.print("")  # Add spacing between results

    # Display footer with divider
    console.print("----------------------------------------")
    console.print(f"Retrieved {len(results)} documents in {elapsed_time:.2f} seconds")
