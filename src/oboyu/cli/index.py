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
) -> None:
    """Index documents for search.

    This command indexes documents in the specified directories, making them searchable.
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

            # Index directory
            chunks_indexed, files_processed = indexer.index_directory(directory, incremental=not force, progress_callback=indexer_progress_callback)

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
