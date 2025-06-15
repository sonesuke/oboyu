"""Clear command implementation for Oboyu CLI.

This module provides the command-line interface for clearing the index database.
"""

from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from oboyu.cli.base import BaseCommand
from oboyu.cli.common_options import ConfigOption
from oboyu.cli.services.indexing_service import IndexingService


def clear(
    ctx: typer.Context,
    config: ConfigOption = None,
    db_path: Annotated[
        Optional[Path],
        typer.Option(
            "--db-path",
            "-d",
            help="Path to the database file (default: from config)",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Clear all data from the index database while preserving the database schema and structure.

    This command clears the index database, removing all indexed file metadata and content.
    Unlike the clear-db command which deletes the database files entirely, this command
    preserves the database structure and only removes the data within it.
    """
    cmd = BaseCommand(ctx)

    # Get config manager from context or create new one
    config_manager = cmd.get_config_manager()

    # Create indexing service
    indexing_service = IndexingService(config_manager, cmd.services.indexer_factory, cmd.services.console_manager)

    # Resolve database path
    resolved_db_path = indexing_service.get_database_path(db_path)

    # Show database path
    cmd.print_database_path(resolved_db_path)

    # Confirm operation
    if not cmd.confirm_database_operation("remove all indexed documents and search data from", force, resolved_db_path):
        return

    # Perform clear operation with progress tracking
    with cmd.logger.live_display():
        clear_op = cmd.logger.start_operation("Clearing index database...")
        indexing_service.clear_index()
        cmd.logger.complete_operation(clear_op)

    cmd.console.print("\nIndex database cleared successfully!")
    cmd.console.print("The database structure has been preserved.")
