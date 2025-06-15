"""Entity deduplication command implementation for Oboyu CLI.

This module provides the command-line interface for deduplicating entities
in the knowledge graph. Moved from oboyu kg deduplicate to top-level command.
"""

import asyncio
import logging
from typing import Optional

import typer
from rich.table import Table
from sentence_transformers import SentenceTransformer
from typing_extensions import Annotated

from oboyu.adapters.entity_deduplication import EDCDeduplicationService
from oboyu.adapters.kg_extraction import LLMKGExtractionService
from oboyu.adapters.kg_repositories import DuckDBKGRepository
from oboyu.application.kg import KnowledgeGraphService
from oboyu.cli.base import BaseCommand

app = typer.Typer(help="Deduplicate entities in the knowledge graph")
logger = logging.getLogger(__name__)


class DeduplicateCommand(BaseCommand):
    """Entity deduplication command."""

    def __init__(self, ctx: typer.Context) -> None:
        """Initialize deduplicate command."""
        super().__init__(ctx)

    async def _get_kg_service(self, config: dict) -> KnowledgeGraphService:
        """Get configured Knowledge Graph service with auto-initialization."""
        # Auto-enable KG on first use - no configuration required
        self.console.print("🚀 Initializing Knowledge Graph functionality...")

        # Set default KG model path if not configured
        kg_model_path = config.get("kg_model_path", "SakanaAI/TinySwallow-1.5B-Instruct-GGUF")

        self.console.print(f"📦 Using model: {kg_model_path}")

        # Get database connection through indexer
        indexer_config = self.create_indexer_config()
        indexer = self.create_indexer(indexer_config, show_progress=False, show_model_loading=False)

        try:
            # Ensure database is initialized
            if not indexer.database_service._is_initialized:
                indexer.database_service.initialize()

            connection = indexer.database_service.db_manager.get_connection()
        except Exception as e:
            # Clean up indexer on error
            try:
                indexer.close()
            except Exception as cleanup_error:
                logger.debug(f"Error during indexer cleanup: {cleanup_error}")
            raise e

        # Initialize KG repository
        kg_repository = DuckDBKGRepository(connection)

        # Initialize extraction service
        self.console.print("🤖 Loading Knowledge Graph extraction model...")
        try:
            extraction_service = LLMKGExtractionService(
                model_path=kg_model_path,
                temperature=config.get("kg_temperature", 0.1),
                max_tokens=config.get("kg_max_tokens", 512),
            )
        except Exception as e:
            self.console.print(f"[red]❌ Failed to load extraction model: {e}[/red]")
            raise typer.Exit(1)

        # Initialize deduplication service (required for this command)
        self.console.print("🔗 Loading entity deduplication service...")
        try:
            # Use existing embedding model for deduplication
            embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
            embedding_model = SentenceTransformer(embedding_model_name)
            deduplication_service = EDCDeduplicationService(
                embedding_model=embedding_model,
                llm_service=extraction_service,
            )
            self.console.print("✅ Entity deduplication service loaded!")
        except Exception as e:
            self.console.print(f"[red]❌ Entity deduplication service failed to load: {e}[/red]")
            raise typer.Exit(1)

        # Create KG service
        self.console.print("⚙️  Initializing Knowledge Graph service...")
        kg_service = KnowledgeGraphService(
            kg_repository=kg_repository,
            extraction_service=extraction_service,
            deduplication_service=deduplication_service,
            confidence_threshold=config.get("kg_confidence_threshold", 0.7),
            enable_deduplication=True,
        )

        self.console.print("[green]🎉 Knowledge Graph system ready![/green]")

        # Store the indexer reference for cleanup later
        kg_service._indexer = indexer

        return kg_service


@app.callback(invoke_without_command=True)
def deduplicate(
    ctx: typer.Context,
    entity_type: Annotated[Optional[str], typer.Option("--type", help="Entity type to deduplicate (all if not specified)")] = None,
    similarity_threshold: Annotated[float, typer.Option("--similarity", help="Vector similarity threshold")] = 0.85,
    verification_threshold: Annotated[float, typer.Option("--verification", help="LLM verification threshold")] = 0.8,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Processing batch size")] = 100,
    show_duplicates: Annotated[bool, typer.Option("--show-duplicates", help="Show potential duplicate entities for analysis")] = False,
    entity_name: Annotated[
        Optional[str], typer.Option("--entity-name", help="Specific entity name to find duplicates for (when --show-duplicates is used)")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Maximum number of duplicate results to show")] = 10,
) -> None:
    """Deduplicate entities in the knowledge graph.

    This command identifies and merges duplicate entities in the knowledge graph
    using vector similarity and LLM verification.

    Examples:
        oboyu deduplicate
        oboyu deduplicate --type PERSON --similarity 0.9
        oboyu deduplicate --show-duplicates --entity-name "Apple Inc"

    """

    async def _deduplicate() -> None:
        command = DeduplicateCommand(ctx)
        try:
            config_manager = command.get_config_manager()
            config_data = config_manager.get_section("indexer")
            kg_service = await command._get_kg_service(config_data)

            if not kg_service.deduplication_service:
                command.console.print("[red]❌ Entity deduplication service not available[/red]")
                raise typer.Exit(1)

            if show_duplicates:
                # Show potential duplicates for analysis
                if not entity_name:
                    command.console.print("[red]❌ --entity-name is required when using --show-duplicates[/red]")
                    raise typer.Exit(1)

                await _show_duplicates(command, kg_service, entity_name, entity_type, similarity_threshold, limit)
            else:
                # Perform actual deduplication
                await _perform_deduplication(command, kg_service, entity_type, similarity_threshold, verification_threshold, batch_size)

        except Exception as e:
            command.console.print(f"[red]❌ Entity deduplication failed: {e}[/red]")
            logger.error(f"Entity deduplication failed: {e}")
            raise typer.Exit(1)
        finally:
            # Clean up resources
            try:
                if "kg_service" in locals() and hasattr(kg_service, "_indexer"):
                    kg_service._indexer.close()
            except Exception as cleanup_error:
                logger.debug(f"Error during cleanup: {cleanup_error}")

    asyncio.run(_deduplicate())


async def _perform_deduplication(
    command: DeduplicateCommand,
    kg_service: KnowledgeGraphService,
    entity_type: Optional[str],
    similarity_threshold: float,
    verification_threshold: float,
    batch_size: int,
) -> None:
    """Perform entity deduplication."""
    command.console.print("🔄 Starting entity deduplication...")

    # Perform deduplication
    stats = await kg_service.deduplicate_all_entities(
        entity_type=entity_type,
        similarity_threshold=similarity_threshold,
        verification_threshold=verification_threshold,
        batch_size=batch_size,
    )

    # Display results
    table = Table(title="Entity Deduplication Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Original Entities", str(stats["original_count"]))
    table.add_row("Deduplicated Entities", str(stats["deduplicated_count"]))
    table.add_row("Entities Merged", str(stats["merged_count"]))

    if stats["original_count"] > 0:
        reduction_percent = (stats["merged_count"] / stats["original_count"]) * 100
        table.add_row("Reduction", f"{reduction_percent:.1f}%")

    command.console.print(table)


async def _show_duplicates(
    command: DeduplicateCommand,
    kg_service: KnowledgeGraphService,
    entity_name: str,
    entity_type: Optional[str],
    similarity_threshold: float,
    limit: int,
) -> None:
    """Show potential duplicate entities for analysis."""
    command.console.print(f"🔍 Searching for duplicates of '{entity_name}'...")

    # Find potential duplicates
    duplicates = await kg_service.find_duplicate_entities(
        entity_name=entity_name,
        entity_type=entity_type,
        similarity_threshold=similarity_threshold,
    )

    if not duplicates:
        command.console.print("[green]✅ No potential duplicates found[/green]")
        return

    # Display results
    table = Table(title=f"Potential Duplicates for '{entity_name}'")
    table.add_column("Entity ID", style="cyan")
    table.add_column("Name", style="yellow")
    table.add_column("Type", style="green")
    table.add_column("Similarity", style="blue")

    for entity_id, name, similarity in duplicates[:limit]:
        # Get entity type for display
        entity_info = await kg_service.kg_repository.get_entity_by_id(entity_id)
        entity_type_display = entity_info.entity_type if entity_info else "Unknown"
        table.add_row(entity_id[:12] + "...", name, entity_type_display, f"{similarity:.3f}")

    command.console.print(table)
    command.console.print(f"Found {len(duplicates)} potential duplicates")


if __name__ == "__main__":
    app()
