"""Main CLI entry point for Oboyu.

This module provides the main command-line interface for Oboyu,
a Japanese-enhanced semantic search system for local documents.
"""

import os
import sys

# Disable Rich's error formatting for all Typer output
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
os.environ["_TYPER_COMPLETE_SHOW_ERRORS"] = "0"

import click
import typer
from rich.console import Console
from typing_extensions import Annotated

from oboyu import __version__
from oboyu.cli.common_options import ConfigOption, DatabasePathOption, VerboseOption
from oboyu.cli.health import app as health_app
from oboyu.cli.index import app as index_app
from oboyu.cli.manage import app as manage_app
from oboyu.cli.mcp import app as mcp_app
from oboyu.cli.query import app as query_app
from oboyu.common.config import ConfigManager
from oboyu.common.paths import ensure_config_dirs

# Create Typer app
app = typer.Typer(
    name="oboyu",
    help="A Japanese-enhanced semantic search system for your local documents.",
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)

# Create console for rich output
console = Console()

# Add subcommands
app.add_typer(index_app, name="index", help="Index documents for search")
app.add_typer(query_app, name="query", help="Search indexed documents")
app.add_typer(manage_app, name="manage", help="Manage the index database")
app.add_typer(health_app, name="health", help="Health monitoring and diagnostics")
app.add_typer(mcp_app, name="mcp", help="Run an MCP server for AI assistant integration")


@app.callback()
def callback(
    ctx: typer.Context,
    config: ConfigOption = None,
    db_path: DatabasePathOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Set up global options for the CLI."""
    # Configure logging based on verbosity level
    import logging

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        # Set to ERROR level to suppress INFO, DEBUG, and WARNING messages
        logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
        # Disable lower level logs from all loggers
        logging.getLogger().setLevel(logging.ERROR)

    # Use ConfigManager for unified configuration handling
    config_manager = ConfigManager(config)

    # Store configuration manager and options in context
    ctx.obj = {"verbose": verbose, "config_manager": config_manager, "config": config, "db_path": db_path, "config_data": config_manager.load_config()}


@app.command()
def version() -> None:
    """Display version information."""
    console.print(f"Oboyu version: {__version__}")


@app.command()
def clear(
    ctx: typer.Context,
    db_path: DatabasePathOption = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Clear all data from the index database while preserving the database schema and structure.

    This command completely removes the database files, providing a thorough cleanup
    that resets the file size and removes all Write-Ahead Log files.
    """
    import os
    from pathlib import Path

    from oboyu.common.config import ConfigManager

    # Get configuration manager from context or create new one
    config_manager = ctx.obj.get("config_manager") if ctx.obj else ConfigManager()

    # Resolve database path
    indexer_config_dict = config_manager.get_section("indexer")
    resolved_db_path = config_manager.resolve_db_path(Path(db_path) if db_path else None, indexer_config_dict)

    # Show database path
    console.print(f"Using database: {resolved_db_path}")

    # Confirm operation
    if not force:
        console.print("Warning: This will completely remove the index database files.")
        confirm = typer.confirm("Are you sure you want to continue?")
        if not confirm:
            console.print("Operation cancelled.")
            return

    # Delete database files
    files_to_delete = [
        str(resolved_db_path),  # Main database file
        str(resolved_db_path) + "-wal",  # Write-Ahead Log file
        str(resolved_db_path) + "-shm",  # Shared memory file
    ]

    deleted_files = []
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(file_path)
        except OSError as e:
            console.print(f"Warning: Could not delete {file_path}: {e}")

    if deleted_files:
        console.print(f"\nDeleted {len(deleted_files)} database file(s):")
        for file_path in deleted_files:
            console.print(f"  - {file_path}")
        console.print("\nIndex database cleared successfully!")
    else:
        console.print("\nNo database files found to delete.")
        console.print("Index database was already clear.")


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
