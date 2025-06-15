"""Status command implementation for Oboyu CLI.

This module provides the command-line interface for showing indexing status.
"""

from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

from oboyu.cli.base import BaseCommand
from oboyu.cli.common_options import ConfigOption
from oboyu.cli.services.indexing_service import IndexingService

DirectoryOption = Annotated[
    List[Path],
    typer.Argument(
        help="Directories to check status for",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
]


def status(
    ctx: typer.Context,
    directories: DirectoryOption,
    config: ConfigOption = None,
    db_path: Annotated[
        Optional[Path],
        typer.Option(
            "--db-path",
            "-p",
            help="Path to the database file (default: from config)",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    detailed: Annotated[
        bool,
        typer.Option("--detailed", "-d", help="Show detailed file-by-file status"),
    ] = False,
) -> None:
    """Show indexing status for specified directories.

    This command shows which files are indexed, modified, or new in the specified directories.
    """
    base_command = BaseCommand(ctx)
    config_manager = base_command.get_config_manager()
    indexing_service = IndexingService(config_manager, base_command.services.indexer_factory, base_command.services.console_manager)

    # Get database path and display it
    resolved_db_path = indexing_service.get_database_path(db_path)
    base_command.print_database_path(resolved_db_path)

    # Get status for all directories
    status_results = indexing_service.get_status(directories, db_path)

    for status_result in status_results:
        base_command.console.print(f"\n[bold]Status for {status_result.directory}:[/bold]")
        base_command.console.print(f"  New files: {status_result.new_files}")
        base_command.console.print(f"  Modified files: {status_result.modified_files}")
        base_command.console.print(f"  Deleted files: {status_result.deleted_files}")
        base_command.console.print(f"  Total indexed: {status_result.total_indexed}")

        if detailed:
            # Get detailed diff for this directory
            diff_results = indexing_service.get_diff([status_result.directory], db_path)
            if diff_results:
                diff_result = diff_results[0]

                if diff_result.new_files:
                    base_command.console.print("\n  [green]New files:[/green]")
                    for f in diff_result.new_files[:10]:  # Show first 10
                        base_command.console.print(f"    + {f}")
                    if len(diff_result.new_files) > 10:
                        base_command.console.print(f"    ... and {len(diff_result.new_files) - 10} more")

                if diff_result.modified_files:
                    base_command.console.print("\n  [yellow]Modified files:[/yellow]")
                    for f in diff_result.modified_files[:10]:  # Show first 10
                        base_command.console.print(f"    ~ {f}")
                    if len(diff_result.modified_files) > 10:
                        base_command.console.print(f"    ... and {len(diff_result.modified_files) - 10} more")

                if diff_result.deleted_files:
                    base_command.console.print("\n  [red]Deleted files:[/red]")
                    for f in diff_result.deleted_files[:10]:  # Show first 10
                        base_command.console.print(f"    - {f}")
                    if len(diff_result.deleted_files) > 10:
                        base_command.console.print(f"    ... and {len(diff_result.deleted_files) - 10} more")
