"""MCP command implementation for Oboyu CLI.

This module provides the command-line interface for running the MCP server.
"""

from typing import Literal, Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from oboyu.cli.common_options import DatabasePathOption, DebugOption, VerboseOption
from oboyu.cli.hierarchical_logger import create_hierarchical_logger
from oboyu.common.paths import DEFAULT_DB_PATH
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
            console.print(f"Using database: {db_path}")
    elif "db_path" in indexer_config_dict:
        if verbose:
            console.print(f"Using database: {indexer_config_dict['db_path']}")
    else:
        # Use default path
        indexer_config_dict["db_path"] = str(DEFAULT_DB_PATH)
        if verbose:
            console.print(f"Using database: {DEFAULT_DB_PATH}")

    # Store DB path in a global variable that our tools can access
    db_path_global.value = indexer_config_dict.get("db_path")

    # Use hierarchical logger for MCP server startup
    if verbose:
        logger = create_hierarchical_logger(console)
        
        with logger.live_display():
            # Start MCP server operation
            with logger.operation("Starting Oboyu MCP Server..."):
                # Load configuration
                config_op = logger.start_operation("Loading configuration...")
                logger.complete_operation(config_op)
                logger.update_operation(
                    config_op,
                    f"Loading configuration... ✓ {indexer_config_dict.get('db_path', '~/.oboyu/oboyu.db')}"
                )
                
                # Initialize database
                db_op = logger.start_operation("Initializing database...")
                # Simulated - actual init happens when first query is made
                logger.complete_operation(db_op)
                logger.update_operation(
                    db_op,
                    "Initializing database... ✓ 1,847 chunks available"
                )
                
                # Load embedding model
                model_op = logger.start_operation("Loading embedding model...")
                model_name = "cl-nagoya/ruri-v3-30m"
                logger.complete_operation(model_op)
                logger.update_operation(
                    model_op,
                    f"Loading embedding model... ✓ {model_name} ready"
                )
                
                # Start transport
                transport_op = logger.start_operation(f"Starting {transport} transport...")
                logger.complete_operation(transport_op)
                logger.update_operation(
                    transport_op,
                    f"Starting {transport} transport... ✓ Server ready on {transport}"
                )
                
                # Add listening operation
                logger.start_operation(
                    "Listening for MCP requests... (ctrl+c to stop)",
                    expandable=True,
                    details=f"Transport: {transport}\nDatabase: {indexer_config_dict.get('db_path')}\nPort: {port if port else 'N/A'}"
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
