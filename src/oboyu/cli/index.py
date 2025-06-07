"""Index command implementation for Oboyu CLI.

This module provides the command-line interface for indexing documents.
"""

from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

from oboyu.cli.base import BaseCommand
from oboyu.cli.common_options import ConfigOption
from oboyu.cli.index_config import create_crawler_config
from oboyu.cli.services.indexing_service import IndexingService

app = typer.Typer(
    help="Index documents for search",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    context_settings={
        "allow_interspersed_args": True,
    },
)

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





@app.callback(invoke_without_command=True)
def index(
    ctx: typer.Context,
    directories: DirectoryOption,
    config: ConfigOption = None,
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

    config_manager = base_command.get_config_manager()
    indexing_service = IndexingService(
        config_manager,
        base_command.services.indexer_factory,
        base_command.services.console_manager
    )
    
    # Get database path and display it
    resolved_db_path = indexing_service.get_database_path(db_path)
    base_command.print_database_path(resolved_db_path)
    
    # Create progress callback if not quiet
    progress_callback = None
    if not quiet_progress:
        def progress_callback_func(message: str) -> None:
            op_id = base_command.logger.start_operation(message, expandable=False)
            base_command.logger.complete_operation(op_id)
        progress_callback = progress_callback_func
    
    # Execute indexing with progress tracking
    with base_command.logger.live_display():
        init_op = base_command.logger.start_operation("Initializing Oboyu indexer...")
        
        result = indexing_service.execute_indexing(
            directories=directories,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            db_path=db_path,
            change_detection=change_detection,
            cleanup_deleted=cleanup_deleted,
            verify_integrity=verify_integrity,
            force=force,
            max_depth=max_depth,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback,
        )
        
        base_command.logger.complete_operation(init_op)
    
    base_command.console.print(f"\nIndexed {result.total_files} files ({result.total_chunks} chunks) in {result.elapsed_time:.1f}s")
