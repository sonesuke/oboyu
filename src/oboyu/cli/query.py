"""Query command implementation for Oboyu CLI.

This module provides the command-line interface for querying indexed documents.
"""

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
from oboyu.cli.interactive_session import InteractiveQuerySession
from oboyu.cli.services.query_service import QueryService
from oboyu.indexer.search.search_result import SearchResult

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
    vector_weight: Optional[float] = None,
    bm25_weight: Optional[float] = None,
    db_path: Optional[Path] = None,
    rerank: Optional[bool] = None,
    interactive: bool = False,
) -> None:
    """Search indexed documents.

    This command searches the index for documents matching the query.
    """
    # Create base command for common functionality
    base_command = BaseCommand(ctx)

    # Check if interactive mode requested
    if interactive:
        if query is not None:
            base_command.console.print("⚠️  Warning: Query argument ignored in interactive mode", style="yellow")
    elif query is None:
        base_command.console.print("❌ Error: Query argument is required (or use --interactive)", style="red")
        raise typer.Exit(1)

    # Get configuration manager and query service
    config_manager = base_command.get_config_manager()
    query_service = QueryService(config_manager)

    try:
        if interactive:
            # Get query configuration for interactive session
            query_config = query_service.get_query_config(top_k, vector_weight, bm25_weight, rerank)
            database_path = query_service.get_database_path(db_path)
            
            # Get indexer configuration and create indexer with proper config for interactive session
            indexer_config = config_manager.get_section("indexer")
            
            from oboyu.indexer import Indexer
            from oboyu.indexer.config.indexer_config import IndexerConfig
            
            config = IndexerConfig()
            config.db_path = Path(database_path)
            
            # Apply configuration from file
            if indexer_config.get("use_reranker", False):
                assert config.search is not None, "SearchConfig should be initialized"
                assert config.model is not None, "ModelConfig should be initialized"
                config.search.use_reranker = True
                config.model.use_reranker = True
            
            # Initialize indexer with proper configuration
            indexer = Indexer(config)
            
            # Start interactive session
            session_config = {
                "mode": mode,
                "top_k": query_config.get("top_k", 10),
                "vector_weight": query_config.get("vector_weight", 0.7),
                "bm25_weight": query_config.get("bm25_weight", 0.3),
                "rerank": query_config.get("use_reranker", False),
            }
            session = InteractiveQuerySession(indexer, session_config, base_command.console)
            session.run()
        else:
            # Execute single query using service
            # At this point, query is guaranteed to be non-None due to validation above
            assert query is not None, "Query should be validated as non-None by this point"
            
            result = query_service.execute_query(
                query=query,
                mode=mode,
                top_k=top_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                db_path=db_path,
                rerank=rerank,
            )

            # Display results
            _display_results(base_command.console, result.results, result.elapsed_time, result.mode, explain, format)

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
) -> None:
    """Display search results."""
    if not results:
        console.print("❌ No results found.")
        return

    # Header
    console.print(
        f"\n🎯 Found [bold green]{len(results)}[/bold green] results "
        f"([dim]{mode} search, {elapsed_time:.3f}s[/dim])\n"
    )

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
        content = result.content[:200].replace('\n', ' ').strip()
        if len(result.content) > 200:
            content += "..."
        console.print(f"    {content}")

        if explain:
            console.print(f"    [dim]Chunk index: {result.chunk_index}[/dim]")

        console.print()  # Empty line between results
