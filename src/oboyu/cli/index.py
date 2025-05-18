"""Index command implementation for Oboyu CLI.

This module provides the command-line interface for indexing documents.
"""

import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from oboyu.cli.paths import DEFAULT_DB_PATH
from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.indexer import Indexer

# Create Typer app
app = typer.Typer(help="Index documents for search and manage the index")

# Create manage subcommand app
manage_app = typer.Typer(help="Manage the index database")
app.add_typer(manage_app, name="manage", help="Manage the index database")

# Create console for rich output
console = Console()

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
        console.print(f"Using explicitly specified database path: [cyan]{db_path}[/cyan]")
    elif "db_path" in indexer_config_dict:
        console.print(f"Using configured database path: [cyan]{indexer_config_dict['db_path']}[/cyan]")
    else:
        # Use the default path from central definition
        indexer_config_dict["db_path"] = str(DEFAULT_DB_PATH)
        console.print(f"Using default database path: [cyan]{DEFAULT_DB_PATH}[/cyan]")

    # Create configuration object
    indexer_config = IndexerConfig(config_dict={"indexer": indexer_config_dict})

    # Confirm before clearing if not forced
    if not force:
        console.print("[bold yellow]Warning:[/bold yellow] This will remove all indexed documents and search data.")
        confirm = typer.confirm("Are you sure you want to continue?")
        if not confirm:
            console.print("Operation cancelled.")
            return

    # Import progress indicator if not already imported
    try:
        # Check if create_indeterminate_progress is already imported
        create_indeterminate_progress
    except NameError:
        from oboyu.cli.formatters import create_indeterminate_progress

    # Show progress during indexer initialization
    with create_indeterminate_progress("Initializing...") as init_progress:
        init_task = init_progress.add_task("Loading embedding model and setting up database...", total=None)

        # Create indexer (this loads the model and sets up the database)
        indexer = Indexer(config=indexer_config)

        # Mark initialization as complete
        init_progress.update(init_task, description="[green]✓[/green] Initialization complete")

    # Clear the index with progress indicator
    console.print("Clearing index database...")
    with create_indeterminate_progress("Clearing...") as clear_progress:
        clear_task = clear_progress.add_task("Removing indexed data...", total=None)

        # Clear the index
        indexer.clear_index()

        # Mark clearing as complete
        clear_progress.update(clear_task, description="[green]✓[/green] Database cleared")

    console.print("[bold green]Index database cleared successfully![/bold green]")


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

    # Create crawler configuration
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

    # Create indexer configuration
    indexer_config_dict = config_data.get("indexer", {})

    # Override with command-line options
    if chunk_size is not None:
        indexer_config_dict["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        indexer_config_dict["chunk_overlap"] = chunk_overlap
    if embedding_model is not None:
        indexer_config_dict["embedding_model"] = embedding_model

    # Handle database path explicitly, with clear precedence:
    # 1. Command-line option (highest priority)
    # 2. Config file value
    # 3. Default from central path definition (lowest priority)
    if db_path is not None:
        indexer_config_dict["db_path"] = str(db_path)
        console.print(f"Using explicitly specified database path: [cyan]{db_path}[/cyan]")
    elif "db_path" in indexer_config_dict:
        console.print(f"Using configured database path: [cyan]{indexer_config_dict['db_path']}[/cyan]")
    else:
        # Use the default path from central definition
        indexer_config_dict["db_path"] = str(DEFAULT_DB_PATH)
        console.print(f"Using default database path: [cyan]{DEFAULT_DB_PATH}[/cyan]")

    # Create configuration objects
    # We don't need to use the crawler_config directly; it's used by the indexer internally
    # when we call index_directory method
    indexer_config = IndexerConfig(config_dict={"indexer": indexer_config_dict})

    # Provide immediate feedback
    console.print(f"Starting indexing of {', '.join(str(d) for d in directories)}...")

    # Import progress indicator
    from oboyu.cli.formatters import create_indeterminate_progress

    # Show progress during indexer initialization (model loading and database setup)
    with create_indeterminate_progress("Initializing...") as init_progress:
        init_task = init_progress.add_task("Loading embedding model and setting up database...", total=None)

        # Create indexer (this loads the model and sets up the database)
        indexer = Indexer(config=indexer_config)

        # Mark initialization as complete
        init_progress.update(init_task, description="[green]✓[/green] Initialization complete")

    console.print("Scanning files...")

    # Process each directory
    total_chunks = 0
    total_files = 0
    start_time = time.time()

    # Reuse the progress indicator already imported

    # First, count the total number of files to be processed
    console.print("Counting files to index...")

    # This requires a two-pass approach:
    # 1. First, we'll estimate the number of files to be processed
    # 2. Then we'll process them with a progress bar

    # Progress tracking is handled by the Progress class

    with create_indeterminate_progress("Scanning directories...", show_count=True) as scan_progress:
        scan_task = scan_progress.add_task("Scanning for documents...", total=None)

        for directory in directories:
            # Update progress
            scan_progress.update(
                scan_task,
                description=f"Scanning [cyan]{directory}[/cyan]...",
                completed=total_files
            )

    # Now do the actual indexing with an accurate progress bar
    console.print("")
    console.print("Indexing documents...")

    with create_indeterminate_progress("Indexing documents...", show_count=True) as progress:
        task = progress.add_task("Indexing documents...", total=None)

        for directory in directories:
            # Output information
            console.print(f"\nProcessing directory: [cyan]{directory}[/cyan]")
            progress.update(task, description=f"Indexing [cyan]{directory}[/cyan]...")

            # Index directory - now returns tuple of (chunks, files)
            chunks_indexed, files_processed = indexer.index_directory(directory, incremental=not force)
            total_chunks += chunks_indexed
            total_files += files_processed

            # Update progress with count
            progress.update(
                task,
                completed=total_files,
                description=f"Processed {files_processed} files ({total_files} total), {chunks_indexed} chunks"
            )

    # Show summary
    elapsed_time = time.time() - start_time
    console.print("[bold green]Indexing complete![/bold green]")
    console.print(f"Processed [bold]{total_files}[/bold] files, indexed [bold]{total_chunks}[/bold] chunks")
    console.print(f"Time elapsed: [bold]{elapsed_time:.2f}[/bold] seconds")
    console.print(f"Processing rate: [bold]{total_chunks / max(1, elapsed_time):.1f}[/bold] chunks/sec")
    console.print(f"Database: [cyan]{indexer_config.db_path}[/cyan]")
    console.print(f"Average chunks per file: [bold]{total_chunks / max(1, total_files):.1f}[/bold]")

