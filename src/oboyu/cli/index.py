"""Index command implementation for Oboyu CLI.

This module provides the command-line interface for indexing documents.
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import typer
from typing_extensions import Annotated

from oboyu.cli.formatters import console
from oboyu.cli.hierarchical_logger import HierarchicalLogger, create_hierarchical_logger
from oboyu.common.paths import DEFAULT_DB_PATH
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

# Define command options
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

RecursiveOption = Annotated[
    Optional[bool],
    typer.Option(
        "--recursive/--no-recursive",
        "-r/-nr",
        help="Process directories recursively",
    ),
]

IncludePatternsOption = Annotated[
    Optional[List[str]],
    typer.Option(
        "--include-patterns",
        "-i",
        help="File patterns to include (e.g., '*.txt,*.md')",
    ),
]

ExcludePatternsOption = Annotated[
    Optional[List[str]],
    typer.Option(
        "--exclude-patterns",
        "-e",
        help="File patterns to exclude (e.g., '*/node_modules/*')",
    ),
]

MaxDepthOption = Annotated[
    Optional[int],
    typer.Option(
        "--max-depth",
        "-d",
        help="Maximum recursion depth",
        min=1,
    ),
]

ForceOption = Annotated[
    bool,
    typer.Option(
        "--force",
        "-f",
        help="Force re-index of all documents",
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

ChunkSizeOption = Annotated[
    Optional[int],
    typer.Option(
        "--chunk-size",
        help="Chunk size in characters",
        min=64,
    ),
]

ChunkOverlapOption = Annotated[
    Optional[int],
    typer.Option(
        "--chunk-overlap",
        help="Chunk overlap in characters",
        min=0,
    ),
]

EmbeddingModelOption = Annotated[
    Optional[str],
    typer.Option(
        "--embedding-model",
        help="Embedding model to use",
    ),
]

DatabasePathOption = Annotated[
    Optional[Path],
    typer.Option(
        "--db-path",
        help="Path to database file",
        file_okay=True,
        dir_okay=False,
        writable=True,
    ),
]


@manage_app.command(name="clear")
def clear(
    ctx: typer.Context,
    db_path: DatabasePathOption = None,
    force: ForceOption = False,
) -> None:
    """Clear all data from the index database.

    This command removes all indexed documents and their embeddings from the database
    while preserving the database schema and structure.
    """
    # Get global options from context
    config_data = ctx.obj.get("config_data", {}) if ctx.obj else {}

    # Create indexer configuration
    indexer_config_dict = config_data.get("indexer", {})

    # Handle database path explicitly, with clear precedence
    if db_path is not None:
        indexer_config_dict["db_path"] = str(db_path)
        console.print(f"Using database: {db_path}")
    elif "db_path" in indexer_config_dict:
        console.print(f"Using database: {indexer_config_dict['db_path']}")
    else:
        # Use the default path from central definition
        indexer_config_dict["db_path"] = str(DEFAULT_DB_PATH)
        console.print(f"Using database: {DEFAULT_DB_PATH}")

    # Create configuration object
    indexer_config = IndexerConfig(config_dict={"indexer": indexer_config_dict})

    # Confirm before clearing if not forced
    if not force:
        console.print("Warning: This will remove all indexed documents and search data.")
        confirm = typer.confirm("Are you sure you want to continue?")
        if not confirm:
            console.print("Operation cancelled.")
            return

    # Use hierarchical logger for clear operation
    logger = create_hierarchical_logger(console)
    
    with logger.live_display():
        # Initialize indexer
        with logger.operation("Initializing Oboyu indexer..."):
            model_name = indexer_config_dict.get("embedding_model", "cl-nagoya/ruri-v3-30m")
            load_op = logger.start_operation(
                f"Loading embedding model ({model_name})..."
            )
            indexer = Indexer(config=indexer_config)
            logger.complete_operation(load_op)
            logger.update_operation(load_op, f"Loading embedding model ({model_name})... ✓ Done")
        
        # Clear the index
        with logger.operation("Clearing index database..."):
            clear_op = logger.start_operation("Removing all indexed data...")
            indexer.clear_index()
            logger.complete_operation(clear_op)
            logger.update_operation(clear_op, "Removing all indexed data... ✓ Done")
        
        # Clean up resources
        indexer.close()
    
    console.print("\nIndex database cleared successfully!")


def _create_crawler_config(
    config_data: dict[str, Any],
    recursive: Optional[bool],
    max_depth: Optional[int],
    include_patterns: Optional[List[str]],
    exclude_patterns: Optional[List[str]],
    japanese_encodings: Optional[List[str]],
) -> dict[str, Any]:
    """Create crawler configuration from config data and command-line options."""
    crawler_config_dict = config_data.get("crawler", {})

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


def _create_progress_callback(logger: HierarchicalLogger, scan_op_id: str, current_ops: dict[str, Optional[str]]) -> Callable[[str, int, int], None]:
    """Create a progress callback for the indexer."""
    files_found = 0
    last_stage = None
    
    def indexer_progress_callback(stage: str, current: int, total: int) -> None:
        nonlocal files_found, last_stage
        
        if stage == "crawling":
            if total > 0 and files_found == 0:
                files_found = total
                logger.update_operation(
                    scan_op_id,
                    "Scanning directory...",
                    details=f"Found {total} files"
                )
                logger.complete_operation(scan_op_id)
                logger.update_operation(
                    scan_op_id,
                    f"Found {total} files (ctrl+r to expand)"
                )
                
                # Start processing documents operation
                current_ops["process"] = logger.start_operation("Processing documents...")
        
        elif stage == "processing":
            # Update single line for file processing progress
            if current <= total:
                if not current_ops["read"]:
                    current_ops["read"] = logger.start_operation(f"Reading files... {current}/{total}")
                else:
                    logger.update_operation(
                        current_ops["read"],
                        f"Reading files... {current}/{total}"
                    )
                
                # Mark complete on last file
                if current == total:
                    if current_ops["read"]:
                        logger.complete_operation(current_ops["read"])
                        logger.update_operation(
                            current_ops["read"],
                            f"Reading files... ✓ {total} files processed"
                        )
                        current_ops["read"] = None
        
        elif stage == "embedding":
            # Start embedding generation on first batch
            if current == 1 and last_stage != "embedding":
                if current_ops["process"]:
                    logger.complete_operation(current_ops["process"])
                    current_ops["process"] = None
                current_ops["embed"] = logger.start_operation("Generating embeddings...")
            
            # Update batch progress
            batch_op_id = current_ops.get("batch")
            if not batch_op_id:
                batch_op_id = logger.start_operation(f"Processing batch {current}/{total}...")
                current_ops["batch"] = batch_op_id
            else:
                logger.update_operation(batch_op_id, f"Processing batch {current}/{total}...")
            
            # Mark complete on last batch
            if current == total:
                if batch_op_id:
                    logger.complete_operation(batch_op_id)
                    logger.update_operation(batch_op_id, f"Processing batch {total}/{total}... ✓ Done")
                    current_ops["batch"] = None
        
        elif stage == "storing" or stage == "storing_embeddings":
            # Handle storing progress
            if current == 1 and last_stage not in ["storing", "storing_embeddings"]:
                # Start storing operation on first chunk
                if current_ops.get("embed"):
                    logger.complete_operation(current_ops["embed"])
                    current_ops["embed"] = None
                store_op_id = logger.start_operation("Storing in database...")
                current_ops["store"] = store_op_id
        
        last_stage = stage
    
    return indexer_progress_callback


def _create_indexer_config(
    config_data: dict[str, Any],
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    embedding_model: Optional[str],
    db_path: Optional[Path],
) -> dict[str, Any]:
    """Create indexer configuration from config data and command-line options."""
    indexer_config_dict = config_data.get("indexer", {})

    # Override with command-line options
    if chunk_size is not None:
        indexer_config_dict["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        indexer_config_dict["chunk_overlap"] = chunk_overlap
    if embedding_model is not None:
        indexer_config_dict["embedding_model"] = embedding_model

    # Handle database path
    if db_path:
        indexer_config_dict["database_path"] = str(db_path)
    elif "database_path" not in indexer_config_dict:
        indexer_config_dict["database_path"] = str(DEFAULT_DB_PATH)

    return dict(indexer_config_dict)


@app.callback(invoke_without_command=True)
def index(
    ctx: typer.Context,
    directories: DirectoryOption,
    recursive: RecursiveOption = True,
    include_patterns: IncludePatternsOption = None,
    exclude_patterns: ExcludePatternsOption = None,
    max_depth: MaxDepthOption = None,
    force: ForceOption = False,
    encoding_detection: EncodingDetectionOption = True,
    japanese_encodings: JapaneseEncodingsOption = None,
    chunk_size: ChunkSizeOption = None,
    chunk_overlap: ChunkOverlapOption = None,
    embedding_model: EmbeddingModelOption = None,
    db_path: DatabasePathOption = None,
) -> None:
    """Index documents for search.

    This command indexes documents in the specified directories, making them searchable.
    """
    # Get global options from context
    config_data = ctx.obj.get("config_data", {}) if ctx.obj else {}

    # Create configurations using helper functions
    crawler_config_dict = _create_crawler_config(
        config_data,
        recursive,
        max_depth,
        include_patterns,
        exclude_patterns,
        japanese_encodings,
    )
    
    indexer_config_dict = _create_indexer_config(
        config_data,
        chunk_size,
        chunk_overlap,
        embedding_model,
        db_path,
    )

    # Handle database path explicitly if needed (already handled in helper)
    if db_path is not None:
        indexer_config_dict["db_path"] = str(db_path)
        console.print(f"Using database: {db_path}")
    elif "db_path" in indexer_config_dict:
        console.print(f"Using database: {indexer_config_dict['db_path']}")
    else:
        # Use the default path from central definition
        indexer_config_dict["db_path"] = str(DEFAULT_DB_PATH)
        console.print(f"Using database: {DEFAULT_DB_PATH}")

    # Create configuration objects
    # Combine crawler and indexer configs for the IndexerConfig
    combined_config = {
        "crawler": crawler_config_dict,
        "indexer": indexer_config_dict
    }
    indexer_config = IndexerConfig(config_dict=combined_config)

    # Use hierarchical logger for indexing operation
    logger = create_hierarchical_logger(console)
    
    with logger.live_display():
        # Initialize indexer
        with logger.operation("Initializing Oboyu indexer..."):
            model_name = indexer_config_dict.get("embedding_model", "cl-nagoya/ruri-v3-30m")
            load_op = logger.start_operation(
                f"Loading embedding model ({model_name})..."
            )
            
            # Create indexer (loads model and sets up database)
            start_time = time.time()
            indexer = Indexer(config=indexer_config)
            duration = time.time() - start_time
            
            logger.complete_operation(load_op)
            logger.update_operation(load_op, f"Loading embedding model ({model_name})... ✓ Done ({duration:.1f}s)")

        # Track totals
        total_chunks = 0
        total_files = 0
        start_time = time.time()

        # Process each directory
        for directory in directories:
            # Start directory scanning operation
            scan_op_id = logger.start_operation(
                f"Scanning directory {directory}...",
                expandable=True
            )
            
            # Track current operations for updates
            current_ops: Dict[str, Optional[str]] = {
                "process": None,
                "embed": None,
                "batch": None,
                "read": None,
                "store": None
            }
            
            # Create progress callback using helper function
            indexer_progress_callback = _create_progress_callback(logger, scan_op_id, current_ops)
            
            # Index directory
            chunks_indexed, files_processed = indexer.index_directory(
                directory,
                incremental=not force,
                progress_callback=indexer_progress_callback
            )
            
            # Complete any remaining operations
            for op_id in current_ops.values():
                if op_id:
                    try:
                        logger.complete_operation(op_id)
                    except Exception:  # noqa: S110
                        pass  # Operation might already be completed
            
            # Add summary
            summary_op = logger.start_operation(
                f"Indexed {chunks_indexed} chunks from {files_processed} documents"
            )
            logger.complete_operation(summary_op)
            
            # Update totals
            total_chunks += chunks_indexed
            total_files += files_processed

        # Clean up resources
        indexer.close()
    
    # Show summary after live display
    elapsed_time = time.time() - start_time
    console.print(f"\nIndexed {total_files} files ({total_chunks} chunks) in {elapsed_time:.1f}s")

