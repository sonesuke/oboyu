"""Main CLI entry point for Oboyu.

This module provides the main command-line interface for Oboyu,
a Japanese-enhanced semantic search system for local documents.
"""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from oboyu import __version__
from oboyu.cli.config import load_config
from oboyu.cli.index import app as index_app
from oboyu.cli.query import app as query_app

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

VerboseOption = Annotated[
    bool,
    typer.Option(
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
]


@app.callback()
def callback(
    ctx: typer.Context,
    config: ConfigOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Set up global options for the CLI."""
    # Store configuration and verbosity in context
    ctx.obj = {"verbose": verbose, "config": config}

    # Load configuration if specified
    if config:
        ctx.obj["config_data"] = load_config(config)


@app.command()
def version() -> None:
    """Display version information."""
    console.print(f"Oboyu version: [bold green]{__version__}[/bold green]")


def run() -> None:
    """Run the CLI application."""
    # Ensure config directory exists
    config_dir = Path.home() / ".oboyu"
    os.makedirs(config_dir, exist_ok=True)

    # Run the app
    try:
        app()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    run()
