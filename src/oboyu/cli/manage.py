"""Manage command implementation for Oboyu CLI.

This module provides the command-line interface for managing the index database.
"""

from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

from oboyu.cli.base import BaseCommand
from oboyu.cli.common_options import ConfigOption
from oboyu.cli.services.indexing_service import IndexingService

app = typer.Typer(
    help="Manage the index database",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    context_settings={
        "allow_interspersed_args": True,
    },
)

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


@app.command(name="clear")
def clear(
    ctx: typer.Context,
    config: ConfigOption = None,
    db_path: Optional[Path] = None,
    force: bool = False,
) -> None:
    """Clear all data from the index database.

    This command removes all indexed documents and their embeddings from the database
    while preserving the database schema and structure.
    """
    base_command = BaseCommand(ctx)
    config_manager = base_command.get_config_manager()
    indexing_service = IndexingService(
        config_manager,
        base_command.services.indexer_factory,
        base_command.services.console_manager
    )
    
    # Get database path
    resolved_db_path = indexing_service.get_database_path(db_path)
    base_command.print_database_path(resolved_db_path)
    
    # Confirm operation
    if not base_command.confirm_database_operation(
        "remove all indexed documents and search data from",
        force,
        resolved_db_path
    ):
        return
    
    # Perform clear operation with progress tracking
    with base_command.logger.live_display():
        clear_op = base_command.logger.start_operation("Clearing index database...")
        indexing_service.clear_index(db_path)
        base_command.logger.complete_operation(clear_op)
    
    base_command.console.print("\nIndex database cleared successfully!")


@app.command(name="status")
def status(
    ctx: typer.Context,
    directories: DirectoryOption,
    config: ConfigOption = None,
    db_path: Optional[Path] = None,
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
    indexing_service = IndexingService(
        config_manager,
        base_command.services.indexer_factory,
        base_command.services.console_manager
    )

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


@app.command(name="diff")
def diff(
    ctx: typer.Context,
    directories: DirectoryOption,
    config: ConfigOption = None,
    db_path: Optional[Path] = None,
    change_detection: Annotated[
        Optional[str],
        typer.Option(
            "--change-detection",
            help="Strategy for detecting changes: timestamp, hash, or smart (default: smart)",
        ),
    ] = None,
) -> None:
    """Show what would be updated if indexing were run now.

    This is a dry-run that shows which files would be added, updated, or removed
    without actually performing any indexing.
    """
    base_command = BaseCommand(ctx)
    config_manager = base_command.get_config_manager()
    indexing_service = IndexingService(
        config_manager,
        base_command.services.indexer_factory,
        base_command.services.console_manager
    )

    # Get database path and display it
    resolved_db_path = indexing_service.get_database_path(db_path)
    base_command.print_database_path(resolved_db_path)
    
    # Get diff results for all directories
    diff_results = indexing_service.get_diff(directories, db_path, change_detection)
    
    total_new = 0
    total_modified = 0
    total_deleted = 0

    for diff_result in diff_results:
        base_command.console.print(f"\n[bold]Diff for {diff_result.directory}:[/bold]")

        total_new += len(diff_result.new_files)
        total_modified += len(diff_result.modified_files)
        total_deleted += len(diff_result.deleted_files)

        if diff_result.new_files:
            base_command.console.print(f"\n  [green]Files to be added ({len(diff_result.new_files)}):[/green]")
            for f in diff_result.new_files:
                base_command.console.print(f"    + {f}")

        if diff_result.modified_files:
            base_command.console.print(f"\n  [yellow]Files to be updated ({len(diff_result.modified_files)}):[/yellow]")
            for f in diff_result.modified_files:
                base_command.console.print(f"    ~ {f}")

        if diff_result.deleted_files:
            base_command.console.print(f"\n  [red]Files to be removed ({len(diff_result.deleted_files)}):[/red]")
            for f in diff_result.deleted_files:
                base_command.console.print(f"    - {f}")

        if not (diff_result.new_files or diff_result.modified_files or diff_result.deleted_files):
            base_command.console.print("  [dim]No changes detected[/dim]")

    base_command.console.print("\n[bold]Summary:[/bold]")
    base_command.console.print(f"  Total files to add: {total_new}")
    base_command.console.print(f"  Total files to update: {total_modified}")
    base_command.console.print(f"  Total files to remove: {total_deleted}")

    if total_new + total_modified + total_deleted == 0:
        base_command.console.print("\n[green]✓[/green] Index is up to date")
    else:
        base_command.console.print(f"\n[yellow]![/yellow] Index needs updating ({total_new + total_modified + total_deleted} changes)")
