"""MCP command implementation for Oboyu CLI.

This module provides the command-line interface for running the MCP server.
"""

from typing import Literal, Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from oboyu.cli.common_options import DatabasePathOption, DebugOption, VerboseOption
from oboyu.cli.formatters import create_indeterminate_progress
from oboyu.cli.paths import DEFAULT_DB_PATH
from oboyu.mcp.context import db_path_global, mcp

# Create Typer app
app = typer.Typer(help="Run an MCP server for semantic search")

# Create console for rich output
console = Console()

TransportOption = Annotated[
    str,  # Using str for Typer but we'll validate values ourselves
    typer.Option(
        "--transport", "-t",
        help="Transport mechanism (stdio, sse, streamable-http)",
    ),
]

PortOption = Annotated[
    Optional[int],
    typer.Option(
        "--port", "-p",
        help="Port number for HTTP or WebSocket transport",
        min=1024,
        max=65535,
    ),
]


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    db_path: DatabasePathOption = None,
    verbose: VerboseOption = False,
    debug: DebugOption = False,
    transport: TransportOption = "stdio",
    port: PortOption = None,
) -> None:
    """Start the MCP server to provide semantic search via AI assistants.

    The server uses the Model Context Protocol (MCP) to expose Oboyu's
    Japanese-enhanced semantic search capabilities to AI assistants.
    """
    # Validate transport is one of the allowed values
    valid_transports: list[Literal["stdio", "sse", "streamable-http"]] = ["stdio", "sse", "streamable-http"]
    if transport not in valid_transports:
        raise typer.BadParameter(f"Transport must be one of: {', '.join(valid_transports)}")
    # Get global options from context
    config_data = ctx.obj.get("config_data", {}) if ctx.obj else {}

    # Handle database path explicitly
    indexer_config_dict = config_data.get("indexer", {}).copy()
    if db_path:
        indexer_config_dict["db_path"] = str(db_path)
        if verbose:
            console.print(f"Using database path: [cyan]{db_path}[/cyan]")
    elif "db_path" in indexer_config_dict:
        if verbose:
            console.print(f"Using configured database path: [cyan]{indexer_config_dict['db_path']}[/cyan]")
    else:
        # Use default path
        indexer_config_dict["db_path"] = str(DEFAULT_DB_PATH)
        if verbose:
            console.print(f"Using default database path: [cyan]{DEFAULT_DB_PATH}[/cyan]")

    # Provide immediate feedback
    if verbose:
        console.print("[bold green]Starting MCP server...[/bold green]")
        console.print(f"Transport: [cyan]{transport}[/cyan]")
        if port and transport in ["sse", "streamable-http"]:
            console.print(f"Port: [cyan]{port}[/cyan]")
        console.print(f"Database path: [cyan]{indexer_config_dict['db_path']}[/cyan]")

    # Store DB path in a global variable that our tools can access
    db_path_global.value = indexer_config_dict.get("db_path")

    # Start the MCP server with progress indicator
    with create_indeterminate_progress("Starting MCP server...") as progress:
        progress.add_task("Initializing...", total=None)

        try:
            # FastMCP.run() accepts only transport as a parameter (confirmed via docs)
            # Cast to the correct type for mypy type checking
            mcp_transport: Literal["stdio", "sse", "streamable-http"] = transport  # type: ignore

            # Now we can run with the proper type
            mcp.run(mcp_transport)
        except Exception as e:
            console.print(f"[bold red]Error starting MCP server:[/bold red] {str(e)}")
            raise typer.Exit(code=1)
