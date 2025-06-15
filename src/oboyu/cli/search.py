"""Unified search command implementation for Oboyu CLI.

This module provides a unified command-line interface for searching indexed documents
with GraphRAG enhancement enabled by default. This replaces both the traditional
query command and the separate GraphRAG search functionality.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from sentence_transformers import SentenceTransformer
from typing_extensions import Annotated

# Disable tokenizer parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from oboyu.adapters.graphrag import OboyuGraphRAGService
from oboyu.adapters.kg_repositories import DuckDBKGRepository
from oboyu.adapters.property_graph import DuckPGQPropertyGraphService
from oboyu.cli.base import BaseCommand
from oboyu.cli.commands.query import QueryCommand
from oboyu.common.config import ConfigManager
from oboyu.common.types import SearchResult

# Create Typer app
app = typer.Typer(
    help="Search indexed documents with GraphRAG enhancement",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)

# Create console for rich output
console = Console()
logger = logging.getLogger(__name__)


@app.callback(invoke_without_command=True)
def search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Search query text")],
    mode: Annotated[str, typer.Option("--mode", help="Search mode: vector, bm25, or hybrid")] = "hybrid",
    top_k: Annotated[Optional[int], typer.Option("--top-k", help="Maximum number of results")] = None,
    no_graph: Annotated[bool, typer.Option("--no-graph", help="Disable GraphRAG enhancement")] = False,
    expand: Annotated[bool, typer.Option("--expand", help="Show query expansion details")] = False,
    explain: Annotated[bool, typer.Option("--explain", help="Show processing details")] = False,
    no_rerank: Annotated[bool, typer.Option("--no-rerank", help="Disable reranking")] = False,
    format: Annotated[str, typer.Option("--format", help="Output format: text or json")] = "text",
    rrf_k: Annotated[Optional[int], typer.Option("--rrf-k", help="RRF constant for hybrid search")] = None,
    db_path: Annotated[Optional[Path], typer.Option("--db-path", "-p", help="Path to database file")] = None,
) -> None:
    """Search indexed documents with GraphRAG enhancement enabled by default.

    This unified search command combines traditional semantic search with GraphRAG
    functionality for enhanced results. GraphRAG is enabled by default unless
    explicitly disabled with --no-graph.

    Examples:
        oboyu search "machine learning algorithms"
        oboyu search "python pandas" --no-graph
        oboyu search "data science" --expand --explain
        oboyu search "neural networks" --mode vector --top-k 5

    """
    # Create base command for common functionality
    base_command = BaseCommand(ctx)

    # Get configuration manager
    config_manager = base_command.get_config_manager()

    try:
        if no_graph:
            # Use traditional search only
            _execute_traditional_search(base_command, config_manager, query, mode, top_k, rrf_k, db_path, not no_rerank, explain, format)
        else:
            # Use GraphRAG-enhanced search (default)
            asyncio.run(_execute_graphrag_search(base_command, config_manager, query, mode, top_k, expand, explain, not no_rerank, format, db_path))

    except Exception as e:
        base_command.console.print(f"❌ Search failed: {e}", style="red")
        logging.error(f"Search error: {e}")
        raise typer.Exit(1)


def _execute_traditional_search(
    base_command: BaseCommand,
    config_manager: ConfigManager,
    query: str,
    mode: str,
    top_k: Optional[int],
    rrf_k: Optional[int],
    db_path: Optional[Path],
    rerank: bool,
    explain: bool,
    format: str,
) -> None:
    """Execute traditional search without GraphRAG."""
    query_service = QueryCommand(config_manager)

    # Execute search
    result = query_service.execute_query_with_context(
        query=query,
        mode=mode,
        top_k=top_k,
        rrf_k=rrf_k,
        db_path=db_path,
        rerank=rerank,
    )

    # Display results
    search_type = f"{mode} search"
    if rerank:
        search_type += " with reranking"

    _display_results(base_command.console, result.results, result.elapsed_time, search_type, explain, format, result.reranker_used)


async def _execute_graphrag_search(
    base_command: BaseCommand,
    config_manager: ConfigManager,
    query: str,
    mode: str,
    top_k: Optional[int],
    expand: bool,
    explain: bool,
    rerank: bool,
    format: str,
    db_path: Optional[Path],
) -> None:
    """Execute GraphRAG-enhanced search."""
    config_data = config_manager.get_section("indexer")

    # Initialize GraphRAG service
    base_command.console.print("🚀 Initializing GraphRAG search...")
    graphrag_service = await _get_graphrag_service(base_command, config_data)

    try:
        if expand:
            # Show query expansion details
            base_command.console.print(f"🔍 Expanding query: '{query}'")
            expansion_result = await graphrag_service.expand_query_with_entities(
                query=query,
                max_entities=10,
                entity_similarity_threshold=0.7,
                expand_depth=1,
            )

            _display_expansion_results(base_command.console, expansion_result)

            if not expansion_result["expanded_entities"] and not explain:
                # Fall back to traditional search if no entities found
                base_command.console.print("⚠️ No graph entities found, using traditional search")
                _execute_traditional_search(base_command, config_manager, query, mode, top_k, None, db_path, rerank, explain, format)
                return

        if explain:
            # Show detailed processing explanation
            await _show_query_explanation(base_command, graphrag_service, query)

        # Perform GraphRAG search
        base_command.console.print(f"🔍 Performing GraphRAG search: '{query}'")
        results = await graphrag_service.semantic_search_with_graph_context(
            query=query,
            max_results=top_k or 10,
            use_graph_expansion=True,
            rerank_with_graph=rerank,
        )

        if not results:
            base_command.console.print("❌ No results found with GraphRAG, trying traditional search...")
            _execute_traditional_search(base_command, config_manager, query, mode, top_k, None, db_path, rerank, explain, format)
            return

        # Convert GraphRAG results to SearchResult format
        search_results = []
        for result in results:
            search_result = SearchResult(
                score=result["relevance_score"],
                path=str(Path(result.get("file_path", "unknown"))),
                title=result.get("title", ""),
                content=result["content"],
                chunk_index=result.get("chunk_index", 0),
            )
            search_results.append(search_result)

        # Display results
        search_type = f"GraphRAG {mode} search"
        if rerank:
            search_type += " with graph reranking"

        # Calculate elapsed time (GraphRAG doesn't return timing)
        elapsed_time = 0.0  # Placeholder since GraphRAG doesn't provide timing

        _display_results(base_command.console, search_results, elapsed_time, search_type, explain, format, rerank)

    finally:
        # Clean up resources
        try:
            if hasattr(graphrag_service, "_indexer"):
                graphrag_service._indexer.close()
        except Exception as cleanup_error:
            logger.debug(f"Error during cleanup: {cleanup_error}")


async def _get_graphrag_service(base_command: BaseCommand, config: dict) -> OboyuGraphRAGService:
    """Get configured GraphRAG service with auto-initialization."""
    # Get database connection through indexer
    indexer_config = base_command.create_indexer_config()
    indexer = base_command.create_indexer(indexer_config, show_progress=False, show_model_loading=False)

    # Ensure database is initialized
    if not indexer.database_service._is_initialized:
        indexer.database_service.initialize()

    connection = indexer.database_service.db_manager.get_connection()

    # Initialize services
    kg_repository = DuckDBKGRepository(connection)
    property_graph_service = DuckPGQPropertyGraphService(connection)

    # Load embedding model
    try:
        embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        embedding_model = SentenceTransformer(embedding_model_name)
    except Exception as e:
        raise typer.BadParameter(f"Failed to load embedding model: {e}")

    # Create GraphRAG service
    graphrag_service = OboyuGraphRAGService(
        kg_repository=kg_repository,
        property_graph_service=property_graph_service,
        embedding_model=embedding_model,
        database_connection=connection,
    )

    # Store the indexer reference for cleanup later
    graphrag_service._indexer = indexer

    return graphrag_service


def _display_expansion_results(console: Console, expansion_result: dict) -> None:
    """Display query expansion results."""
    console.print("[green]✅ Query expansion complete[/green]")
    console.print(f"Original query: {expansion_result['original_query']}")
    console.print(f"Extracted candidates: {len(expansion_result['extracted_candidates'])}")
    console.print(f"Matched entities: {expansion_result['matched_entities']}")

    # Display expanded entities
    expanded_entities = expansion_result["expanded_entities"]
    if expanded_entities:
        from rich.table import Table

        entity_table = Table(title="Expanded Entities")
        entity_table.add_column("Name", style="yellow")
        entity_table.add_column("Type", style="green")
        entity_table.add_column("Relevance", style="cyan")
        entity_table.add_column("Confidence", style="blue")

        for item in expanded_entities:
            entity = item["entity"]
            relevance = item["relevance_score"]
            entity_table.add_row(entity.name, entity.entity_type, f"{relevance:.3f}", f"{entity.confidence:.3f}")

        console.print(entity_table)

    # Display relations if any
    relations = expansion_result["relations"]
    if relations:
        console.print(f"\n[cyan]Found {len(relations)} related relations[/cyan]")


async def _show_query_explanation(base_command: BaseCommand, graphrag_service: OboyuGraphRAGService, query: str) -> None:
    """Show detailed query processing explanation."""
    base_command.console.print(f"🔍 Analyzing query: '{query}'")

    # Expand query to understand processing
    expansion_result = await graphrag_service.expand_query_with_entities(
        query=query,
        max_entities=5,
        expand_depth=1,
    )

    expanded_entities = [item["entity"] for item in expansion_result["expanded_entities"]]

    # Get contextual chunks
    contextual_chunks = await graphrag_service.get_contextual_chunks(
        entities=expanded_entities,
        relations=expansion_result["relations"],
        max_chunks=5,
    )

    # Generate explanation
    explanation = await graphrag_service.generate_query_explanation(
        original_query=query,
        expanded_entities=expanded_entities,
        selected_chunks=contextual_chunks,
    )

    # Display explanation
    base_command.console.print("[green]✅ Query processing explanation:[/green]")
    base_command.console.print(explanation)

    # Show detailed breakdown
    base_command.console.print("\n[cyan]Detailed breakdown:[/cyan]")
    base_command.console.print(f"• Extracted candidates: {expansion_result['extracted_candidates']}")
    base_command.console.print(f"• Found entities: {len(expanded_entities)}")
    base_command.console.print(f"• Found relations: {len(expansion_result['relations'])}")
    base_command.console.print(f"• Contextual chunks: {len(contextual_chunks)}")


def _display_results(
    console: Console,
    results: list[SearchResult],
    elapsed_time: float,
    search_type: str,
    explain: bool,
    format: str,
    reranker_used: bool = False,
) -> None:
    """Display search results."""
    if not results:
        if format == "json":
            # Output empty JSON structure for no results
            json_output = {"results": [], "count": 0, "search_type": search_type, "duration": elapsed_time}
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
        else:
            console.print("❌ No results found.")
        return

    if format == "json":
        # Convert results to JSON format
        json_results = []
        for result in results:
            # Create snippet (first 200 chars)
            content = result.content[:200].replace("\n", " ").strip()
            if len(result.content) > 200:
                content += "..."

            json_result = {
                "score": result.score,
                "file_path": str(result.path),
                "title": result.title or "",
                "snippet": content,
                "language": getattr(result, "language", "en"),
            }

            # Add chunk index if explain mode is enabled
            if explain:
                json_result["chunk_index"] = result.chunk_index

            json_results.append(json_result)

        # Create final JSON output structure
        json_output = {
            "results": json_results,
            "count": len(results),
            "search_type": search_type,
            "duration": elapsed_time,
        }

        # Output JSON using print to avoid Rich formatting
        print(json.dumps(json_output, indent=2, ensure_ascii=False))
    else:
        # Original text format
        console.print(f"\n🎯 Found [bold green]{len(results)}[/bold green] results ([dim]{search_type}, {elapsed_time:.3f}s[/dim])\n")

        # Display results
        for i, result in enumerate(results, 1):
            # Score color coding
            score = result.score
            if score >= 0.8:
                score_color = "bright_green"
            elif score >= 0.6:
                score_color = "green"
            elif score >= 0.4:
                score_color = "yellow"
            else:
                score_color = "red"

            # Display result
            console.print(f"[bold blue]{i:2d}.[/bold blue] [{score_color}]{score:.3f}[/{score_color}] [dim]{result.path}[/dim]")

            if result.title:
                console.print(f"    [bold]{result.title}[/bold]")

            # Content preview
            content = result.content[:200].replace("\n", " ").strip()
            if len(result.content) > 200:
                content += "..."
            console.print(f"    {content}")

            if explain:
                console.print(f"    [dim]Chunk index: {result.chunk_index}[/dim]")

            console.print()  # Empty line between results
