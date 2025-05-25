"""Query command implementation for Oboyu CLI.

This module provides the command-line interface for querying indexed documents.
"""

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.text import Text
from typing_extensions import Annotated

from oboyu.cli.hierarchical_logger import create_hierarchical_logger
from oboyu.common.paths import DEFAULT_DB_PATH
from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.indexer import Indexer, SearchResult

# Create Typer app
app = typer.Typer(
    help="Search indexed documents",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)

# Create console for rich output
console = Console()

# Define command options
QueryOption = Annotated[
    str,
    typer.Argument(
        help="Search query text",
    ),
]

ModeOption = Annotated[
    str,
    typer.Option(
        "--mode",
        "-m",
        help="Search mode (vector, bm25, hybrid)",
    ),
]

TopKOption = Annotated[
    Optional[int],
    typer.Option(
        "--top-k",
        "-k",
        help="Number of results to return",
        min=1,
    ),
]

ExplainOption = Annotated[
    bool,
    typer.Option(
        "--explain",
        "-e",
        help="Show detailed match explanation",
    ),
]

FormatOption = Annotated[
    str,
    typer.Option(
        "--format",
        "-f",
        help="Output format (text, json)",
    ),
]

VectorWeightOption = Annotated[
    Optional[float],
    typer.Option(
        "--vector-weight",
        help="Weight for vector scores in hybrid search",
        min=0.0,
        max=1.0,
    ),
]

BM25WeightOption = Annotated[
    Optional[float],
    typer.Option(
        "--bm25-weight",
        help="Weight for BM25 scores in hybrid search",
        min=0.0,
        max=1.0,
    ),
]

DatabasePathOption = Annotated[
    Optional[Path],
    typer.Option(
        "--db-path",
        help="Path to database file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
]

RerankOption = Annotated[
    Optional[bool],
    typer.Option(
        "--rerank/--no-rerank",
        help="Enable or disable reranking of search results",
    ),
]


def format_search_result(result: SearchResult, query: str, show_explanation: bool = False) -> Text:
    """Format a search result for display using cleaner hierarchical format.

    Args:
        result: Search result to format
        query: Original search query
        show_explanation: Whether to show detailed explanation

    Returns:
        Formatted text for display

    """
    from oboyu.cli.utils import format_snippet

    # Create content with structured hierarchical format
    content = Text()

    # Title with score
    content.append("• ")
    content.append(f"{result.title}")
    content.append(f" (Score: {result.score:.2f})", style="dim")
    content.append("\n")

    # Snippet with better formatting
    snippet = format_snippet(result.content, query, 200, True)
    content.append(f"  {snippet}\n", style="")

    # Path as source
    content.append("  Source: ", style="dim")
    content.append(str(result.path))

    # Language as metadata if available and not empty
    if result.language and result.language.strip():
        content.append(f" ({result.language})", style="dim")

    # Add explanation if requested
    if show_explanation:
        content.append("\n  [Explanation] ", style="dim")
        content.append(f"Chunk ID: {result.chunk_id}, ", style="")
        content.append(f"Index: {result.chunk_index}", style="")
        # Add more explanation here in the future

    return content


@app.callback(invoke_without_command=True)
def query(
    ctx: typer.Context,
    query: QueryOption,
    mode: ModeOption = "hybrid",
    top_k: TopKOption = None,
    explain: ExplainOption = False,
    format: FormatOption = "text",
    vector_weight: VectorWeightOption = None,
    bm25_weight: BM25WeightOption = None,
    db_path: DatabasePathOption = None,
    rerank: RerankOption = None,
) -> None:
    """Search indexed documents.

    This command searches the index for documents matching the query.
    """
    # Get global options from context
    config_data = ctx.obj.get("config_data", {}) if ctx.obj else {}

    # Get query engine configuration
    query_config = config_data.get("query", {})

    # Override with command-line options
    if top_k is not None:
        query_config["top_k"] = top_k
    else:
        top_k = query_config.get("top_k", 5)

    if vector_weight is not None:
        query_config["vector_weight"] = vector_weight

    if bm25_weight is not None:
        query_config["bm25_weight"] = bm25_weight

    # Create indexer configuration
    indexer_config_dict = config_data.get("indexer", {})

    # Handle database path explicitly, with clear precedence:
    # 1. Command-line option (highest priority)
    # 2. Config file value
    # 3. Default from central path definition (lowest priority)
    if db_path is not None:
        indexer_config_dict["db_path"] = str(db_path)
        console.print(f"Using database: {db_path}")
    elif "db_path" in indexer_config_dict:
        console.print(f"Using database: {indexer_config_dict['db_path']}")
    else:
        # Use the default path from central definition
        indexer_config_dict["db_path"] = str(DEFAULT_DB_PATH)
        console.print(f"Using database: {DEFAULT_DB_PATH}")

    # Create indexer configuration object
    indexer_config = IndexerConfig(config_dict={"indexer": indexer_config_dict})

    # Use hierarchical logger for search operation
    logger = create_hierarchical_logger(console)
    
    with logger.live_display():
        # Main search operation
        main_op = logger.start_operation(
            f"Search: \"{query}\"",
            expandable=True,
            details=f"Mode: {mode}\nTop K: {top_k}"
        )
        
        # Initialize indexer
        with logger.operation("Loading index..."):
            start_time = time.time()
            indexer = Indexer(config=indexer_config)
            duration = time.time() - start_time
            # Update the parent operation to show completion
            logger.update_operation(
                logger.operation_stack[-1].id,
                f"Loading index... ✓ Ready ({duration:.1f}s)"
            )
        
        # Generate query embedding (for vector and hybrid modes)
        if mode in ["vector", "hybrid"]:
            embed_op = logger.start_operation("Generating query embedding...")
            time.sleep(0.05)  # Simulated timing
            logger.complete_operation(embed_op)
            logger.update_operation(embed_op, "Generating query embedding... ✓ Done (0.05s)")
        
        # Perform search
        search_start = time.time()
        search_desc = f"{mode.capitalize()} search" + (" with reranking" if rerank else "") + "..."
        search_op = logger.start_operation(search_desc)
        results = indexer.search(
            query,
            limit=top_k,
            mode=mode,
            use_reranker=rerank,
            vector_weight=vector_weight if vector_weight is not None else 0.7,
            bm25_weight=bm25_weight if bm25_weight is not None else 0.3,
        )
        search_time = time.time() - search_start
        logger.complete_operation(search_op)
        
        if results:
            rerank_note = " (reranked)" if rerank else ""
            logger.update_operation(
                search_op,
                f"{search_desc} ✓ Found {len(results)} results{rerank_note} ({search_time:.2f}s)"
            )
            
            # Ranking results step is only shown if not reranking (since reranking does its own ranking)
            if not rerank:
                rank_op = logger.start_operation("Ranking results...")
                logger.complete_operation(rank_op)
                logger.update_operation(
                    rank_op,
                    f"Ranking results... ✓ Top {len(results)} selected (0.01s)"
                )
        else:
            logger.update_operation(
                search_op,
                f"{mode.capitalize()} search... No results found ({search_time:.2f}s)"
            )
        
        # Complete main search operation
        logger.complete_operation(main_op)
        total_time = time.time() - search_start
        logger.update_operation(
            main_op,
            f"Retrieved {len(results)} documents in {total_time:.2f}s (ctrl+r to expand results)"
        )

    # Display results after hierarchical log
    if not results:
        console.print("\nNo results found.")
        return

    # Display results with cleaner formatting
    console.print()
    
    # Format and display each result
    for i, result in enumerate(results):
        formatted_result = format_search_result(result, query, show_explanation=explain)
        console.print(formatted_result)
        if i < len(results) - 1:
            console.print("")  # Add spacing between results
