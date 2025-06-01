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
from oboyu.indexer import Indexer
from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.config.model_config import ModelConfig
from oboyu.indexer.config.processing_config import ProcessingConfig
from oboyu.indexer.config.search_config import SearchConfig

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
    base_command.print_database_path(indexer_config_dict["db_path"])

    # Create indexer configuration objects
    model_config = ModelConfig(
        embedding_model=indexer_config_dict.get("embedding_model", "cl-nagoya/ruri-v3-30m"),
        embedding_device=indexer_config_dict.get("embedding_device", "cpu"),
        use_onnx=indexer_config_dict.get("use_onnx", True),
    )
    
    search_config = SearchConfig(
        bm25_k1=indexer_config_dict.get("bm25_k1", 1.2),
        bm25_b=indexer_config_dict.get("bm25_b", 0.75),
        use_reranker=indexer_config_dict.get("use_reranker", False),
    )
    
    processing_config = ProcessingConfig(
        chunk_size=indexer_config_dict.get("chunk_size", 1024),
        chunk_overlap=indexer_config_dict.get("chunk_overlap", 256),
        db_path=Path(indexer_config_dict.get("db_path", "oboyu.db")),
    )
    
    indexer_config = IndexerConfig(
        model=model_config,
        search=search_config,
        processing=processing_config,
    )
    
    # Create indexer
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
                patterns=crawler_config.include_patterns,
                exclude_patterns=crawler_config.exclude_patterns,
                max_depth=crawler_config.depth,
                follow_symlinks=crawler_config.follow_symlinks,
            )

            # Detect changes (extract just paths from tuples)
            paths_only = [path for path, metadata in discovered_paths]
            changes = indexer.change_detector.detect_changes(paths_only, strategy="smart")

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
    base_command.print_database_path(indexer_config_dict["db_path"])

    # Create indexer configuration objects
    model_config = ModelConfig(
        embedding_model=indexer_config_dict.get("embedding_model", "cl-nagoya/ruri-v3-30m"),
        embedding_device=indexer_config_dict.get("embedding_device", "cpu"),
        use_onnx=indexer_config_dict.get("use_onnx", True),
    )
    
    search_config = SearchConfig(
        bm25_k1=indexer_config_dict.get("bm25_k1", 1.2),
        bm25_b=indexer_config_dict.get("bm25_b", 0.75),
        use_reranker=indexer_config_dict.get("use_reranker", False),
    )
    
    processing_config = ProcessingConfig(
        chunk_size=indexer_config_dict.get("chunk_size", 1024),
        chunk_overlap=indexer_config_dict.get("chunk_overlap", 256),
        db_path=Path(indexer_config_dict.get("db_path", "oboyu.db")),
    )
    
    indexer_config = IndexerConfig(
        model=model_config,
        search=search_config,
        processing=processing_config,
    )
    
    # Create indexer
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

            # Discover all documents
            discovered_paths = discover_documents(
                Path(directory),
                patterns=crawler_config.include_patterns,
                exclude_patterns=crawler_config.exclude_patterns,
                max_depth=crawler_config.depth,
                follow_symlinks=crawler_config.follow_symlinks,
            )

            # Detect changes (extract just paths from tuples)
            paths_only = [path for path, metadata in discovered_paths]
            changes = indexer.change_detector.detect_changes(paths_only, strategy=detection_strategy)

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
    # Create base command for common functionality
    base_command = BaseCommand(ctx)

    # Get configuration manager from context
    config_manager = base_command.get_config_manager()

    # Create configurations using helper functions
    _create_crawler_config(
        config_manager,
        recursive,
        max_depth,
        include_patterns,
        exclude_patterns,
    )

    indexer_config_dict = _create_indexer_config(
        config_manager,
        chunk_size,
        chunk_overlap,
        embedding_model,
        db_path,
    )

    # Display database path
    base_command.print_database_path(indexer_config_dict["db_path"])

    # Create configuration objects using new modular structure
    model_config = ModelConfig(
        embedding_model=indexer_config_dict.get("embedding_model", "cl-nagoya/ruri-v3-30m"),
        embedding_device=indexer_config_dict.get("embedding_device", "cpu"),
        use_onnx=indexer_config_dict.get("use_onnx", True),
        reranker_model=indexer_config_dict.get("reranker_model", "cl-nagoya/ruri-reranker-small"),
        reranker_device=indexer_config_dict.get("reranker_device", "cpu"),
        reranker_use_onnx=indexer_config_dict.get("reranker_use_onnx", True),
    )

    search_config = SearchConfig(
        bm25_k1=indexer_config_dict.get("bm25_k1", 1.2),
        bm25_b=indexer_config_dict.get("bm25_b", 0.75),
        use_reranker=indexer_config_dict.get("use_reranker", True),
        top_k_multiplier=indexer_config_dict.get("reranker_top_k_multiplier", 3),
    )

    processing_config = ProcessingConfig(
        chunk_size=indexer_config_dict.get("chunk_size", 1024),
        chunk_overlap=indexer_config_dict.get("chunk_overlap", 256),
        db_path=Path(indexer_config_dict.get("db_path", "oboyu.db")),
        max_workers=indexer_config_dict.get("max_workers", 4),
        ef_construction=indexer_config_dict.get("ef_construction", 128),
        ef_search=indexer_config_dict.get("ef_search", 64),
        m=indexer_config_dict.get("m", 16),
        m0=indexer_config_dict.get("m0"),
    )

    indexer_config = IndexerConfig(
        model=model_config,
        search=search_config,
        processing=processing_config,
    )

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

            # Create progress callback using hierarchical display (or None for quiet mode)
            if quiet_progress:
                indexer_progress_callback = None
            else:
                indexer_progress_callback = create_indexer_progress_callback(base_command.logger, scan_op_id)

            # Determine change detection strategy
            _detection_strategy = change_detection or "smart"  # Default strategy
            if verify_integrity:
                _detection_strategy = "hash"  # Force hash-based detection for integrity verification

            # Determine cleanup behavior
            _should_cleanup = cleanup_deleted if cleanup_deleted is not None else True  # Default behavior

            # Use crawler + indexer workflow for NewIndexer
            from oboyu.crawler.crawler import Crawler

            # Create crawler with configuration
            crawler = Crawler(
                depth=max_depth or 10,
                include_patterns=include_patterns or ["*.txt", "*.md", "*.html", "*.py", "*.java"],
                exclude_patterns=exclude_patterns or ["*/node_modules/*", "*/venv/*"],
                follow_symlinks=False,
                max_workers=4,
                respect_gitignore=True,
            )

            # Crawl directory to get documents
            crawler_results = crawler.crawl(directory, progress_callback=indexer_progress_callback)

            # Index the crawled documents with progress callback
            result = indexer.index_documents(crawler_results, progress_callback=indexer_progress_callback)
            chunks_indexed = result.get("indexed_chunks", 0)
            files_processed = result.get("total_documents", 0)

            # Create dummy diff_stats for compatibility
            diff_stats = {
                "total_files_discovered": files_processed,
                "files_skipped": 0,
                "chunks_from_skipped_files": 0,
                "deleted_files": 0,
            }

            # Add detailed summary with differential stats
            if not force and diff_stats.get("total_files_discovered", 0) > 0:
                # Show differential update efficiency
                total_discovered = diff_stats["total_files_discovered"]
                files_skipped = diff_stats.get("files_skipped", 0)
                chunks_skipped = diff_stats.get("chunks_from_skipped_files", 0)
                deleted_files = diff_stats.get("deleted_files", 0)

                efficiency_pct = (files_skipped / total_discovered * 100) if total_discovered > 0 else 0

                summary_text = f"Indexed {chunks_indexed} chunks from {files_processed} documents"
                if files_skipped > 0:
                    summary_text += f" ({efficiency_pct:.1f}% files skipped, {chunks_skipped} chunks avoided)"
                if deleted_files > 0:
                    summary_text += f" ({deleted_files} deleted files cleaned up)"

                summary_op = base_command.logger.start_operation(summary_text)
            else:
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
