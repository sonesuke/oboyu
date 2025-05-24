"""Main CLI entry point for Oboyu.

This module provides the main command-line interface for Oboyu,
a Japanese-enhanced semantic search system for local documents.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Disable Rich's error formatting for all Typer output
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
os.environ["_TYPER_COMPLETE_SHOW_ERRORS"] = "0"

import click
import typer
from rich.console import Console
from typing_extensions import Annotated

from oboyu import __version__
from oboyu.cli.config import load_config
from oboyu.cli.index import app as index_app
from oboyu.cli.mcp import app as mcp_app
from oboyu.cli.query import app as query_app
from oboyu.common.paths import DEFAULT_DB_PATH, ensure_config_dirs
from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.indexer import Indexer

# Create Typer app
app = typer.Typer(
    name="oboyu",
    help="A Japanese-enhanced semantic search system for your local documents.",
    add_completion=False,
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)

# Create console for rich output
console = Console()

# Add subcommands
app.add_typer(index_app, name="index", help="Index documents for search")
app.add_typer(query_app, name="query", help="Search indexed documents")
app.add_typer(mcp_app, name="mcp", help="Run an MCP server for AI assistant integration")

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
    console.print(f"Oboyu version: {__version__}")


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
        console.print(f"Using database: {db_path}")
    elif "db_path" in indexer_config_dict:
        console.print(f"Using database: {indexer_config_dict['db_path']}")
    else:
        # Use the default path from central definition
        indexer_config_dict["db_path"] = str(DEFAULT_DB_PATH)
        console.print(f"Using database: {DEFAULT_DB_PATH}")

    # Create configuration object
    indexer_config = IndexerConfig(config_dict={"indexer": indexer_config_dict})

    # Confirm before clearing if not forced
    if not force:
        console.print("Warning: This will remove all indexed documents and search data.")
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
        init_progress.update(init_task, description="✓ Initialization complete")

    # Clear the index with progress indicator
    console.print("Clearing index database...")
    with create_indeterminate_progress("Clearing...") as clear_progress:
        clear_task = clear_progress.add_task("Removing indexed data...", total=None)

        # Clear the index
        indexer.clear_index()

        # Mark clearing as complete
        clear_progress.update(clear_task, description="✓ Database cleared")

    console.print("\nIndex database cleared successfully!")


def run() -> None:
    """Run the CLI application."""
    # Ensure config directory exists
    ensure_config_dirs()

    # Monkey-patch to disable Rich error boxes
    try:
        # Try to import typer's rich module and disable it
        import typer.rich_utils
        typer.rich_utils.FORCE_TERMINAL = False
        if hasattr(typer.rich_utils, "SHOW_ARGUMENTS"):
            typer.rich_utils.SHOW_ARGUMENTS = False
        
        # Also try to disable click's rich integration
        if hasattr(click, "rich"):
            click.rich = None
    except ImportError:
        pass
    
    # Custom exception handler for Click errors
    def handle_click_exception(e: click.ClickException) -> None:
        """Handle Click exceptions without Rich formatting."""
        if isinstance(e, click.UsageError):
            if e.ctx:
                print(f"Usage: {e.ctx.command_path} [OPTIONS] COMMAND [ARGS]...")
            else:
                print("Usage: oboyu [OPTIONS] COMMAND [ARGS]...")
            print("Try 'oboyu --help' for help.")
            print(f"\nError: {e.message}")
        else:
            print(f"Error: {e.message}")
        sys.exit(e.exit_code)
    
    # Run the app
    try:
        app(standalone_mode=False)
    except click.UsageError as e:
        handle_click_exception(e)
    except click.ClickException as e:
        handle_click_exception(e)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run()
