"""MCP command implementation for Oboyu CLI.

This module provides the command-line interface for running the MCP server.
"""

from pathlib import Path
from typing import Literal, Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from oboyu.cli.base import BaseCommand
from oboyu.cli.services.mcp_service import MCPService

# Create Typer app
app = typer.Typer(
    help="Run an MCP server for semantic search",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)

# Create console for rich output
console = Console()

TransportOption = Annotated[
    str,  # Using str for Typer but we'll validate values ourselves
    typer.Option(
        "--transport",
        "-t",
        help="Transport mechanism (stdio, sse, streamable-http)",
    ),
]

PortOption = Annotated[
    Optional[int],
    typer.Option(
        "--port",
        "-p",
        help="Port number for HTTP or WebSocket transport",
        min=1024,
        max=65535,
    ),
]


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    db_path: Optional[Path] = None,
    verbose: bool = False,
    debug: bool = False,
    transport: str = "stdio",
    port: Optional[int] = None,
) -> None:
    """Start the MCP server to provide semantic search via AI assistants.

    The server uses the Model Context Protocol (MCP) to expose Oboyu's
    Japanese-enhanced semantic search capabilities to AI assistants.
    """
    # Create base command for common functionality
    base_command = BaseCommand(ctx)

    # Validate transport is one of the allowed values
    valid_transports: list[Literal["stdio", "sse", "streamable-http"]] = ["stdio", "sse", "streamable-http"]
    if transport not in valid_transports:
        raise typer.BadParameter(f"Transport must be one of: {', '.join(valid_transports)}")

    # Get configuration manager and MCP service
    config_manager = base_command.get_config_manager()
    mcp_service = MCPService(config_manager)

    # Get resolved database path
    resolved_db_path = mcp_service.get_database_path(db_path)

    if verbose:
        base_command.print_database_path(resolved_db_path)

    # Use hierarchical logger for MCP server startup
    if verbose:
        with base_command.logger.live_display():
            # Start MCP server operation
            server_op = base_command.logger.start_operation("Starting Oboyu MCP Server...")

            # Load configuration
            config_op = base_command.logger.start_operation("Loading configuration...")
            base_command.logger.complete_operation(config_op)

            # Initialize database
            db_op = base_command.logger.start_operation("Initializing database...")
            # Simulated - actual init happens when first query is made
            base_command.logger.complete_operation(db_op)

            # Load embedding model
            model_op = base_command.logger.start_operation("Loading embedding model...")
            base_command.logger.complete_operation(model_op)

            # Start transport
            transport_op = base_command.logger.start_operation(f"Starting {transport} transport...")
            base_command.logger.complete_operation(transport_op)

            base_command.logger.complete_operation(server_op)

            # Add listening operation
            base_command.logger.start_operation(
                "Listening for MCP requests... (ctrl+c to stop)",
                expandable=True,
                details=f"Transport: {transport}\nDatabase: {resolved_db_path}\nPort: {port if port else 'N/A'}",
            )

    # Run the server
    try:
        # Cast to the correct type for mypy type checking
        mcp_transport: Literal["stdio", "sse", "streamable-http"] = transport  # type: ignore

        # Start the server using the service
        mcp_service.start_server(db_path, mcp_transport, port)
    except Exception as e:
        base_command.console.print(f"Error starting MCP server: {str(e)}", style="red")
        raise typer.Exit(code=1)
