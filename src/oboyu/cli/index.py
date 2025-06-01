"""Index command implementation for Oboyu CLI.

This module provides the command-line interface for indexing documents.
"""

from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

from oboyu.cli.base import BaseCommand
from oboyu.cli.index_config import (
    build_status_indexer_config,
    create_crawler_config,
    create_indexer_config,
)
from oboyu.cli.index_operations import execute_indexing_operation
from oboyu.indexer import Indexer

app = typer.Typer(
    help="Index documents for search and manage the index",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)

manage_app = typer.Typer(
    help="Manage the index database",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)
app.add_typer(manage_app, name="manage", help="Manage the index database")

DirectoryOption = Annotated[
    List[Path],
    typer.Argument(
        help="Directories to index",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
]

EncodingDetectionOption = Annotated[
    bool,
    typer.Option(
        "--encoding-detection/--no-encoding-detection",
        help="Enable/disable Japanese encoding detection",
    ),
]

@manage_app.command(name="clear")
def clear(
    ctx: typer.Context,
    db_path: Optional[Path] = None,
    force: bool = False,
) -> None:
    """Clear all data from the index database.

    This command removes all indexed documents and their embeddings from the database
    while preserving the database schema and structure.
    """
    base_command = BaseCommand(ctx)
    base_command.handle_clear_operation(db_path=str(db_path) if db_path else None, force=force)

@manage_app.command(name="status")
def status(
    ctx: typer.Context,
    directories: DirectoryOption,
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

    indexer_config_dict = create_indexer_config(
        config_manager,
        None,  # chunk_size
        None,  # chunk_overlap
        None,  # embedding_model
        db_path,
    )

    base_command.print_database_path(indexer_config_dict["db_path"])
    indexer_config = build_status_indexer_config(indexer_config_dict)
    
    indexer = Indexer(config=indexer_config)

    try:
        from oboyu.crawler.config import load_default_config
        from oboyu.crawler.discovery import discover_documents

        crawler_config = load_default_config()

        for directory in directories:
            base_command.console.print(f"\n[bold]Status for {directory}:[/bold]")

            discovered_paths = discover_documents(
                Path(directory),
                patterns=crawler_config.include_patterns,
                exclude_patterns=crawler_config.exclude_patterns,
                max_depth=crawler_config.depth,
            )

            paths_only = [path for path, metadata in discovered_paths]
            changes = indexer.change_detector.detect_changes(paths_only, strategy="smart")

            stats = indexer.change_detector.get_processing_stats()

            base_command.console.print(f"  New files: {len(changes.new_files)}")
            base_command.console.print(f"  Modified files: {len(changes.modified_files)}")
            base_command.console.print(f"  Deleted files: {len(changes.deleted_files)}")
            base_command.console.print(f"  Total indexed: {stats.get('completed', 0)}")

            if detailed:
                if changes.new_files:
                    base_command.console.print("\n  [green]New files:[/green]")
                    for f in sorted(changes.new_files)[:10]:  # Show first 10
                        base_command.console.print(f"    + {f}")
                    if len(changes.new_files) > 10:
                        base_command.console.print(f"    ... and {len(changes.new_files) - 10} more")

                if changes.modified_files:
                    base_command.console.print("\n  [yellow]Modified files:[/yellow]")
                    for f in sorted(changes.modified_files)[:10]:  # Show first 10
                        base_command.console.print(f"    ~ {f}")
                    if len(changes.modified_files) > 10:
                        base_command.console.print(f"    ... and {len(changes.modified_files) - 10} more")

                if changes.deleted_files:
                    base_command.console.print("\n  [red]Deleted files:[/red]")
                    for f in sorted(changes.deleted_files)[:10]:  # Show first 10
                        base_command.console.print(f"    - {f}")
                    if len(changes.deleted_files) > 10:
                        base_command.console.print(f"    ... and {len(changes.deleted_files) - 10} more")

    finally:
        indexer.close()

@manage_app.command(name="diff")
def diff(
    ctx: typer.Context,
    directories: DirectoryOption,
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

    indexer_config_dict = create_indexer_config(
        config_manager,
        None,  # chunk_size
        None,  # chunk_overlap
        None,  # embedding_model
        db_path,
    )

    base_command.print_database_path(indexer_config_dict["db_path"])
    indexer_config = build_status_indexer_config(indexer_config_dict)
    
    indexer = Indexer(config=indexer_config)

    try:
        from oboyu.crawler.config import load_default_config
        from oboyu.crawler.discovery import discover_documents

        crawler_config = load_default_config()
        detection_strategy = change_detection or "smart"

        total_new = 0
        total_modified = 0
        total_deleted = 0

        for directory in directories:
            base_command.console.print(f"\n[bold]Diff for {directory}:[/bold]")

            discovered_paths = discover_documents(
                Path(directory),
                patterns=crawler_config.include_patterns,
                exclude_patterns=crawler_config.exclude_patterns,
                max_depth=crawler_config.depth,
            )

            paths_only = [path for path, metadata in discovered_paths]
            changes = indexer.change_detector.detect_changes(paths_only, strategy=detection_strategy)

            total_new += len(changes.new_files)
            total_modified += len(changes.modified_files)
            total_deleted += len(changes.deleted_files)

            if changes.new_files:
                base_command.console.print(f"\n  [green]Files to be added ({len(changes.new_files)}):[/green]")
                for f in sorted(changes.new_files):
                    base_command.console.print(f"    + {f}")

            if changes.modified_files:
                base_command.console.print(f"\n  [yellow]Files to be updated ({len(changes.modified_files)}):[/yellow]")
                for f in sorted(changes.modified_files):
                    base_command.console.print(f"    ~ {f}")

            if changes.deleted_files:
                base_command.console.print(f"\n  [red]Files to be removed ({len(changes.deleted_files)}):[/red]")
                for f in sorted(changes.deleted_files):
                    base_command.console.print(f"    - {f}")

            if not changes.has_changes():
                base_command.console.print("  [dim]No changes detected[/dim]")

        base_command.console.print("\n[bold]Summary:[/bold]")
        base_command.console.print(f"  Total files to add: {total_new}")
        base_command.console.print(f"  Total files to update: {total_modified}")
        base_command.console.print(f"  Total files to remove: {total_deleted}")

        if total_new + total_modified + total_deleted == 0:
            base_command.console.print("\n[green]✓[/green] Index is up to date")
        else:
            base_command.console.print(f"\n[yellow]![/yellow] Index needs updating ({total_new + total_modified + total_deleted} changes)")

    finally:
        indexer.close()


@app.callback(invoke_without_command=True)
def index(
    ctx: typer.Context,
    directories: DirectoryOption,
    recursive: Optional[bool] = True,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
    force: bool = False,
    encoding_detection: bool = True,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_model: Optional[str] = None,
    db_path: Optional[Path] = None,
    change_detection: Annotated[
        Optional[str],
        typer.Option(
            "--change-detection",
            help="Strategy for detecting changes: timestamp, hash, or smart (default: smart)",
        ),
    ] = None,
    cleanup_deleted: Annotated[
        Optional[bool],
        typer.Option(
            "--cleanup-deleted/--no-cleanup-deleted",
            help="Remove deleted files from index during incremental update",
        ),
    ] = None,
    verify_integrity: Annotated[
        bool,
        typer.Option(
            "--verify-integrity",
            help="Verify file integrity using content hashes (slower but more accurate)",
        ),
    ] = False,
    quiet_progress: Annotated[
        bool,
        typer.Option(
            "--quiet-progress",
            "-q",
            help="Minimal progress output to avoid screen flickering",
        ),
    ] = False,
) -> None:
    """Index documents for search.

    This command indexes documents in the specified directories, making them searchable.
    When run without --force, it performs incremental indexing by default, only processing
    new or modified files.
    """
    base_command = BaseCommand(ctx)
    config_manager = base_command.get_config_manager()

    create_crawler_config(
        config_manager,
        recursive,
        max_depth,
        include_patterns,
        exclude_patterns,
    )

    execute_indexing_operation(
        base_command=base_command,
        directories=directories,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        db_path=db_path,
        change_detection=change_detection,
        cleanup_deleted=cleanup_deleted,
        verify_integrity=verify_integrity,
        quiet_progress=quiet_progress,
        force=force,
        max_depth=max_depth,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
