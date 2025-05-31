"""Index command implementation for Oboyu CLI.

This module provides the command-line interface for indexing documents.
"""

import time
from pathlib import Path
from typing import Any, List, Optional

import typer
from typing_extensions import Annotated

from oboyu.cli.base import BaseCommand
from oboyu.cli.progress import create_indexer_progress_callback
from oboyu.common.config import ConfigManager
from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.indexer import Indexer

# Create Typer app
app = typer.Typer(
    help="Index documents for search and manage the index",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)

# Create manage subcommand app
manage_app = typer.Typer(
    help="Manage the index database",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)
app.add_typer(manage_app, name="manage", help="Manage the index database")

# Define command-specific options not in common_options
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

JapaneseEncodingsOption = Annotated[
    Optional[List[str]],
    typer.Option(
        "--japanese-encodings",
        "-j",
        help="Japanese encodings to detect (e.g., 'utf-8,shift-jis,euc-jp')",
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
    
    # Create indexer config
    indexer_config_dict = _create_indexer_config(
        config_manager,
        None,  # chunk_size
        None,  # chunk_overlap
        None,  # embedding_model
        db_path,
    )
    
    # Display database path
    base_command.print_database_path(indexer_config_dict['db_path'])
    
    # Create indexer
    combined_config = {"indexer": indexer_config_dict}
    indexer_config = IndexerConfig(config_dict=combined_config)
    indexer = Indexer(config=indexer_config)
    
    try:
        from oboyu.crawler.config import load_default_config
        from oboyu.crawler.discovery import discover_documents
        
        crawler_config = load_default_config()
        
        for directory in directories:
            base_command.console.print(f"\n[bold]Status for {directory}:[/bold]")
            
            # Discover all documents
            discovered_paths = discover_documents(
                Path(directory),
                depth=crawler_config.depth,
                include_patterns=crawler_config.include_patterns,
                exclude_patterns=crawler_config.exclude_patterns,
                follow_symlinks=crawler_config.follow_symlinks,
            )
            
            # Detect changes
            changes = indexer.change_detector.detect_changes(
                list(discovered_paths),
                strategy=indexer_config.change_detection_strategy
            )
            
            # Get processing stats
            stats = indexer.change_detector.get_processing_stats()
            
            # Display summary
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
    
    # Create indexer config
    indexer_config_dict = _create_indexer_config(
        config_manager,
        None,  # chunk_size
        None,  # chunk_overlap
        None,  # embedding_model
        db_path,
    )
    
    # Display database path
    base_command.print_database_path(indexer_config_dict['db_path'])
    
    # Create indexer
    combined_config = {"indexer": indexer_config_dict}
    indexer_config = IndexerConfig(config_dict=combined_config)
    indexer = Indexer(config=indexer_config)
    
    try:
        from oboyu.crawler.config import load_default_config
        from oboyu.crawler.discovery import discover_documents
        
        crawler_config = load_default_config()
        detection_strategy = change_detection or indexer_config.change_detection_strategy
        
        total_new = 0
        total_modified = 0
        total_deleted = 0
        
        for directory in directories:
            base_command.console.print(f"\n[bold]Diff for {directory}:[/bold]")
            
            # Discover all documents
            discovered_paths = discover_documents(
                Path(directory),
                depth=crawler_config.depth,
                include_patterns=crawler_config.include_patterns,
                exclude_patterns=crawler_config.exclude_patterns,
                follow_symlinks=crawler_config.follow_symlinks,
            )
            
            # Detect changes
            changes = indexer.change_detector.detect_changes(
                list(discovered_paths),
                strategy=detection_strategy
            )
            
            # Update totals
            total_new += len(changes.new_files)
            total_modified += len(changes.modified_files)
            total_deleted += len(changes.deleted_files)
            
            # Display changes
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
        
        # Show summary
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


def _create_crawler_config(
    config_manager: ConfigManager,
    recursive: Optional[bool],
    max_depth: Optional[int],
    include_patterns: Optional[List[str]],
    exclude_patterns: Optional[List[str]],
    japanese_encodings: Optional[List[str]],
) -> dict[str, Any]:
    """Create crawler configuration from config data and command-line options."""
    crawler_config_dict = config_manager.get_section("crawler")

    # Override with command-line options
    if recursive is not None:
        crawler_config_dict["depth"] = 0 if not recursive else (max_depth or 10)
    elif max_depth is not None:
        crawler_config_dict["depth"] = max_depth

    if include_patterns:
        crawler_config_dict["include_patterns"] = include_patterns
    if exclude_patterns:
        crawler_config_dict["exclude_patterns"] = exclude_patterns
    if japanese_encodings:
        crawler_config_dict["japanese_encodings"] = japanese_encodings

    return dict(crawler_config_dict)


# All the complex callback functions have been removed - they are no longer needed


def _create_indexer_config(
    config_manager: ConfigManager,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    embedding_model: Optional[str],
    db_path: Optional[Path],
) -> dict[str, Any]:
    """Create indexer configuration from config data and command-line options."""
    # Use merge_cli_overrides for proper precedence handling
    cli_overrides: dict[str, Any] = {}
    
    if chunk_size is not None:
        cli_overrides["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        cli_overrides["chunk_overlap"] = chunk_overlap
    if embedding_model is not None:
        cli_overrides["embedding_model"] = embedding_model
    if db_path is not None:
        cli_overrides["db_path"] = str(db_path)
        
    indexer_config_dict = config_manager.merge_cli_overrides("indexer", cli_overrides)
    
    # Ensure db_path is set
    if "db_path" not in indexer_config_dict:
        resolved_db_path = config_manager.resolve_db_path(None, indexer_config_dict)
        indexer_config_dict["db_path"] = str(resolved_db_path)

    # Handle database_path alias (some code may use database_path instead of db_path)
    if db_path:
        indexer_config_dict["database_path"] = str(db_path)
    elif "database_path" not in indexer_config_dict:
        indexer_config_dict["database_path"] = indexer_config_dict.get("db_path", str(config_manager.resolve_db_path(None, indexer_config_dict)))

    return dict(indexer_config_dict)


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
    japanese_encodings: Optional[List[str]] = None,
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
) -> None:
    """Index documents for search.

    This command indexes documents in the specified directories, making them searchable.
    When run without --force, it performs incremental indexing by default, only processing
    new or modified files.
    """
    # Create base command for common functionality
    base_command = BaseCommand(ctx)
    
    # Get configuration manager from context
    config_manager = base_command.get_config_manager()

    # Create configurations using helper functions
    crawler_config_dict = _create_crawler_config(
        config_manager,
        recursive,
        max_depth,
        include_patterns,
        exclude_patterns,
        japanese_encodings,
    )

    indexer_config_dict = _create_indexer_config(
        config_manager,
        chunk_size,
        chunk_overlap,
        embedding_model,
        db_path,
    )

    # Display database path
    base_command.print_database_path(indexer_config_dict['db_path'])

    # Create configuration objects
    # Combine crawler and indexer configs for the IndexerConfig
    combined_config = {"crawler": crawler_config_dict, "indexer": indexer_config_dict}
    indexer_config = IndexerConfig(config_dict=combined_config)

    # Use hierarchical logger for indexing operation
    with base_command.logger.live_display():
        # Initialize indexer with nested loading operation
        init_op = base_command.logger.start_operation("Initializing Oboyu indexer...")
        model_name = indexer_config_dict.get("embedding_model", "cl-nagoya/ruri-v3-30m")
        load_op = base_command.logger.start_operation(f"Loading embedding model ({model_name})...")

        # Create indexer (loads model and sets up database)
        indexer = Indexer(config=indexer_config)

        base_command.logger.complete_operation(load_op)
        base_command.logger.complete_operation(init_op)

        # Track totals
        total_chunks = 0
        total_files = 0
        start_time = time.time()

        # Process each directory
        for directory in directories:
            # Start directory scanning operation
            scan_op_id = base_command.logger.start_operation(f"Scanning directory {directory}...", expandable=False)

            # Create progress callback using helper function
            indexer_progress_callback = create_indexer_progress_callback(base_command.logger, scan_op_id)
            
            # Determine change detection strategy
            detection_strategy = change_detection or indexer_config.change_detection_strategy
            if verify_integrity:
                detection_strategy = "hash"  # Force hash-based detection for integrity verification
            
            # Determine cleanup behavior
            should_cleanup = cleanup_deleted if cleanup_deleted is not None else indexer_config.cleanup_deleted_files

            # Index directory with enhanced options
            chunks_indexed, files_processed = indexer.index_directory(
                directory,
                incremental=not force,
                change_detection_strategy=detection_strategy,
                cleanup_deleted=should_cleanup,
                progress_callback=indexer_progress_callback
            )

            # Add summary
            summary_op = base_command.logger.start_operation(f"Indexed {chunks_indexed} chunks from {files_processed} documents")
            base_command.logger.complete_operation(summary_op)

            # Update totals
            total_chunks += chunks_indexed
            total_files += files_processed

        # Clean up resources
        indexer.close()

    # Show summary after live display
    elapsed_time = time.time() - start_time
    base_command.console.print(f"\nIndexed {total_files} files ({total_chunks} chunks) in {elapsed_time:.1f}s")
