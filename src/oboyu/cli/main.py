"""Main CLI entry point for Oboyu.

This module provides the main command-line interface for Oboyu,
a Japanese-enhanced semantic search system for local documents.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from oboyu import __version__
from oboyu.cli.config import load_config
from oboyu.cli.index import app as index_app
from oboyu.cli.paths import DEFAULT_DB_PATH, ensure_config_dirs
from oboyu.cli.query import app as query_app
from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.indexer import Indexer

# Create Typer app
app = typer.Typer(
    name="oboyu",
    help="A Japanese-enhanced semantic search system for your local documents.",
    add_completion=False,
    rich_markup_mode="rich",
)

# Create console for rich output
console = Console()

# Add subcommands
app.add_typer(index_app, name="index", help="Index documents for search")
app.add_typer(query_app, name="query", help="Search indexed documents")

# Define global options
ConfigOption = Annotated[
    Optional[Path],
    typer.Option(
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
]

DatabasePathOption = Annotated[
    Optional[Path],
    typer.Option(
        "--db-path",
        help="Path to database file",
        file_okay=True,
        dir_okay=False,
    ),
]

VerboseOption = Annotated[
    bool,
    typer.Option(
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
]

ForceOption = Annotated[
    bool,
    typer.Option(
        "--force",
        "-f",
        help="Force operation without confirmation",
    ),
]


@app.callback()
def callback(
    ctx: typer.Context,
    config: ConfigOption = None,
    db_path: DatabasePathOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Set up global options for the CLI."""
    # Store configuration and verbosity in context
    ctx.obj = {"verbose": verbose, "config": config, "db_path": db_path}

    # Load configuration if specified
    if config:
        ctx.obj["config_data"] = load_config(config)

    # Initialize config_data if not already present
    if "config_data" not in ctx.obj:
        ctx.obj["config_data"] = {}

    # Initialize indexer config if not already present
    if "indexer" not in ctx.obj["config_data"]:
        ctx.obj["config_data"]["indexer"] = {}

    # Apply database path from CLI option if specified
    if db_path:
        ctx.obj["config_data"]["indexer"]["db_path"] = str(db_path)


@app.command()
def version() -> None:
    """Display version information."""
    console.print(f"Oboyu version: [bold green]{__version__}[/bold green]")


@app.command()
def clear(
    ctx: typer.Context,
    db_path: DatabasePathOption = None,
    force: ForceOption = False,
) -> None:
    """Clear all data from the index database.

    This command removes all indexed documents and their embeddings from the database
    while preserving the database schema and structure.
    """
    # Get global options from context
    config_data = ctx.obj.get("config_data", {}) if ctx.obj else {}

    # Create indexer configuration
    indexer_config_dict = config_data.get("indexer", {})

    # Handle database path explicitly, with clear precedence
    if db_path is not None:
        indexer_config_dict["db_path"] = str(db_path)
        console.print(f"Using explicitly specified database path: [cyan]{db_path}[/cyan]")
    elif "db_path" in indexer_config_dict:
        console.print(f"Using configured database path: [cyan]{indexer_config_dict['db_path']}[/cyan]")
    else:
        # Use the default path from central definition
        indexer_config_dict["db_path"] = str(DEFAULT_DB_PATH)
        console.print(f"Using default database path: [cyan]{DEFAULT_DB_PATH}[/cyan]")

    # Create configuration object
    indexer_config = IndexerConfig(config_dict={"indexer": indexer_config_dict})

    # Confirm before clearing if not forced
    if not force:
        console.print("[bold yellow]Warning:[/bold yellow] This will remove all indexed documents and search data.")
        confirm = typer.confirm("Are you sure you want to continue?")
        if not confirm:
            console.print("Operation cancelled.")
            return

    # Import and use progress indicator for better UX
    from oboyu.cli.formatters import create_indeterminate_progress

    # Show progress during indexer initialization
    with create_indeterminate_progress("Initializing...") as init_progress:
        init_task = init_progress.add_task("Loading embedding model and setting up database...", total=None)

        # Create indexer (this loads the model and sets up the database)
        indexer = Indexer(config=indexer_config)

        # Mark initialization as complete
        init_progress.update(init_task, description="[green]✓[/green] Initialization complete")

    # Clear the index with progress indicator
    console.print("Clearing index database...")
    with create_indeterminate_progress("Clearing...") as clear_progress:
        clear_task = clear_progress.add_task("Removing indexed data...", total=None)

        # Clear the index
        indexer.clear_index()

        # Mark clearing as complete
        clear_progress.update(clear_task, description="[green]✓[/green] Database cleared")

    console.print("[bold green]Index database cleared successfully![/bold green]")


def run() -> None:
    """Run the CLI application."""
    # Display a welcome banner for better UX
    console.print("""
[bold cyan]=====================================[/bold cyan]
[bold green]  Oboyu - Japanese-Enhanced Search[/bold green]
[bold cyan]=====================================[/bold cyan]
""")

    # Ensure config directory exists
    ensure_config_dirs()

    # Run the app
    try:
        app()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    run()
