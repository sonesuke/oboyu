"""MCP command implementation for Oboyu CLI.

This module provides the command-line interface for running the MCP server.
"""

from typing import Literal, Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from oboyu.cli.common_options import DatabasePathOption, DebugOption, VerboseOption
from oboyu.cli.hierarchical_logger import create_hierarchical_logger
from oboyu.common.config import ConfigManager
from oboyu.mcp.context import db_path_global, mcp

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
    # Get configuration manager from context
    config_manager = ctx.obj.get("config_manager") if ctx.obj else ConfigManager()

    # Get indexer configuration
    indexer_config_dict = config_manager.get_section("indexer")
    
    # Resolve database path using ConfigManager
    resolved_db_path = config_manager.resolve_db_path(db_path, indexer_config_dict)
    indexer_config_dict["db_path"] = str(resolved_db_path)
    
    if verbose:
        console.print(f"Using database: {resolved_db_path}")

    # Store DB path in a global variable that our tools can access
    db_path_global.value = str(resolved_db_path)

    # Use hierarchical logger for MCP server startup
    if verbose:
        logger = create_hierarchical_logger(console)

        with logger.live_display():
            # Start MCP server operation
            server_op = logger.start_operation("Starting Oboyu MCP Server...")
            
            # Load configuration
            config_op = logger.start_operation("Loading configuration...")
            logger.complete_operation(config_op)

            # Initialize database
            db_op = logger.start_operation("Initializing database...")
            # Simulated - actual init happens when first query is made
            logger.complete_operation(db_op)

            # Load embedding model
            model_op = logger.start_operation("Loading embedding model...")
            logger.complete_operation(model_op)

            # Start transport
            transport_op = logger.start_operation(f"Starting {transport} transport...")
            logger.complete_operation(transport_op)
            
            logger.complete_operation(server_op)

            # Add listening operation
            logger.start_operation(
                "Listening for MCP requests... (ctrl+c to stop)",
                expandable=True,
                details=f"Transport: {transport}\nDatabase: {indexer_config_dict.get('db_path')}\nPort: {port if port else 'N/A'}",
            )

    # Run the server
    try:
        # Cast to the correct type for mypy type checking
        mcp_transport: Literal["stdio", "sse", "streamable-http"] = transport  # type: ignore

        # Now we can run with the proper type
        mcp.run(mcp_transport)
    except Exception as e:
        console.print(f"Error starting MCP server: {str(e)}", style="red")
        raise typer.Exit(code=1)
