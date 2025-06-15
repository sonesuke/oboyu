"""Main CLI entry point for Oboyu.

This module provides the simplified command-line interface for Oboyu,
a Japanese-enhanced semantic search system with integrated GraphRAG functionality.
"""

import os
import sys

# Disable Rich's error formatting for all Typer output
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
os.environ["_TYPER_COMPLETE_SHOW_ERRORS"] = "0"

import click
import typer
from rich.console import Console

from oboyu import __version__
from oboyu.cli.build_kg import app as build_kg_app
from oboyu.cli.clear import clear
from oboyu.cli.common_options import ConfigOption, DatabasePathOption, VerboseOption
from oboyu.cli.deduplicate import app as deduplicate_app
from oboyu.cli.enrich import enrich
from oboyu.cli.index import app as index_app
from oboyu.cli.mcp import app as mcp_app
from oboyu.cli.search import app as search_app
from oboyu.cli.status import status
from oboyu.common.config import ConfigManager
from oboyu.common.paths import ensure_config_dirs

# Create Typer app
app = typer.Typer(
    name="oboyu",
    help="A Japanese-enhanced semantic search system with integrated GraphRAG functionality.",
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)

# Create console for rich output
console = Console()

# Main commands (simplified structure)
app.add_typer(index_app, name="index", help="Index documents for search")
app.add_typer(search_app, name="search", help="Search documents with GraphRAG enhancement")
app.command("enrich")(enrich)
app.add_typer(build_kg_app, name="build-kg", help="Build knowledge graph from indexed documents")
app.add_typer(deduplicate_app, name="deduplicate", help="Deduplicate entities in knowledge graph")
app.add_typer(mcp_app, name="mcp", help="Run MCP server for AI assistant integration")

# Top-level commands
app.command("clear")(clear)
app.command("status")(status)


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
