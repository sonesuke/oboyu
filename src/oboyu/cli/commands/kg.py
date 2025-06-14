"""Knowledge Graph CLI commands.

This module provides command-line interface for building and managing
the Property Graph Index functionality.
"""

import asyncio
import logging
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sentence_transformers import SentenceTransformer

from oboyu.adapters.entity_deduplication import EDCDeduplicationService
from oboyu.adapters.kg_extraction import ELYZAKGExtractionService
from oboyu.adapters.kg_repositories import DuckDBKGRepository
from oboyu.application.kg import KnowledgeGraphService
from oboyu.cli.base import BaseCommand
from oboyu.config.schema import IndexerConfigSchema

app = typer.Typer(help="Knowledge Graph operations")
logger = logging.getLogger(__name__)


class KGCommand(BaseCommand):
    """Base class for Knowledge Graph commands."""

    def __init__(self) -> None:
        """Initialize KG command."""
        super().__init__()
        self.console = Console()

    async def _get_kg_service(self, config: IndexerConfigSchema) -> KnowledgeGraphService:
        """Get configured Knowledge Graph service."""
        if not config.kg_enabled:
            raise typer.BadParameter("Knowledge Graph is not enabled. Set kg_enabled=true in config.")

        if not config.kg_model_path:
            raise typer.BadParameter("KG model path not configured. Set kg_model_path in config.")

        # Get database connection
        database_manager = await self.get_database_manager()
        connection = database_manager.get_connection()

        # Initialize KG repository
        kg_repository = DuckDBKGRepository(connection)

        # Initialize extraction service
        extraction_service = ELYZAKGExtractionService(
            model_path=config.kg_model_path,
            temperature=config.kg_temperature,
            max_tokens=config.kg_max_tokens,
        )

        # Initialize deduplication service (optional)
        deduplication_service = None
        try:
            # Use existing embedding model for deduplication
            embedding_model = SentenceTransformer(config.embedding_model)
            deduplication_service = EDCDeduplicationService(
                embedding_model=embedding_model,
                llm_service=extraction_service,
            )
        except Exception as e:
            logger.warning(f"Could not initialize deduplication service: {e}")

        # Create KG service
        kg_service = KnowledgeGraphService(
            kg_repository=kg_repository,
            extraction_service=extraction_service,
            deduplication_service=deduplication_service,
            confidence_threshold=config.kg_confidence_threshold,
            enable_deduplication=deduplication_service is not None,
        )

        return kg_service


@app.command()
def build(
    full: bool = typer.Option(False, "--full", help="Rebuild entire knowledge graph"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Processing batch size"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit number of chunks to process"),
) -> None:
    """Build knowledge graph from existing chunks."""

    async def _build() -> None:
        command = KGCommand()
        try:
            config = await command.get_config()
            kg_service = await command._get_kg_service(config.indexer)

            # Validate extraction service
            command.console.print("🔍 Validating extraction service...")
            if not await kg_service.validate_extraction_service():
                command.console.print("[red]❌ Extraction service validation failed[/red]")
                raise typer.Exit(1)

            command.console.print("[green]✅ Extraction service validated[/green]")

            # Get chunks to process
            if full:
                command.console.print("🔄 Full rebuild: Getting all chunks...")
                # Get all chunks from database
                database_manager = await command.get_database_manager()
                connection = database_manager.get_connection()

                query = "SELECT id, content FROM chunks ORDER BY created_at"
                if limit:
                    query += f" LIMIT {limit}"

                results = connection.execute(query).fetchall()
                chunks_to_process = [(result[1], result[0]) for result in results]
            else:
                command.console.print("📊 Delta update: Getting unprocessed chunks...")
                unprocessed_chunk_ids = await kg_service.get_unprocessed_chunks(limit)

                if not unprocessed_chunk_ids:
                    command.console.print("[green]✅ All chunks are already processed[/green]")
                    return

                # Get chunk content
                database_manager = await command.get_database_manager()
                connection = database_manager.get_connection()

                placeholders = ",".join(["?" for _ in unprocessed_chunk_ids])
                query = f"SELECT id, content FROM chunks WHERE id IN ({placeholders})"
                results = connection.execute(query, unprocessed_chunk_ids).fetchall()
                chunks_to_process = [(result[1], result[0]) for result in results]

            if not chunks_to_process:
                command.console.print("[yellow]⚠️  No chunks found to process[/yellow]")
                return

            command.console.print(f"📈 Processing {len(chunks_to_process)} chunks...")

            # Configure batch size
            actual_batch_size = batch_size or config.indexer.kg_batch_size

            # Build knowledge graph with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=command.console,
            ) as progress:
                task = progress.add_task("Building knowledge graph...", total=len(chunks_to_process))

                # Process and update progress
                processing_results = []
                for i in range(0, len(chunks_to_process), actual_batch_size):
                    batch = chunks_to_process[i : i + actual_batch_size]
                    batch_results = await kg_service.build_knowledge_graph(
                        batch,
                        entity_types=config.indexer.kg_entity_types,
                        relation_types=config.indexer.kg_relation_types,
                        batch_size=actual_batch_size,
                    )
                    processing_results.extend(batch_results)
                    progress.update(task, advance=len(batch))

            # Display results
            completed = sum(1 for r in processing_results if r.status == "completed")
            errors = sum(1 for r in processing_results if r.status == "error")
            total_entities = sum(r.entity_count for r in processing_results)
            total_relations = sum(r.relation_count for r in processing_results)

            # Create results table
            table = Table(title="Knowledge Graph Build Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Chunks Processed", str(len(chunks_to_process)))
            table.add_row("Successful", str(completed))
            table.add_row("Errors", str(errors))
            table.add_row("Entities Extracted", str(total_entities))
            table.add_row("Relations Extracted", str(total_relations))

            command.console.print(table)

            if errors > 0:
                command.console.print(f"[yellow]⚠️  {errors} chunks failed to process[/yellow]")

            # Get final stats
            stats = await kg_service.get_knowledge_graph_stats()
            command.console.print("📊 Knowledge Graph Stats:")
            command.console.print(f"  Total Entities: {stats['entity_count']}")
            command.console.print(f"  Total Relations: {stats['relation_count']}")

        except Exception as e:
            command.console.print(f"[red]❌ Knowledge graph build failed: {e}[/red]")
            logger.error(f"KG build failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_build())


@app.command()
def stats() -> None:
    """Show knowledge graph statistics."""

    async def _stats() -> None:
        command = KGCommand()
        try:
            config = await command.get_config()
            kg_service = await command._get_kg_service(config.indexer)

            stats = await kg_service.get_knowledge_graph_stats()

            table = Table(title="Knowledge Graph Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Entities", str(stats["entity_count"]))
            table.add_row("Total Relations", str(stats["relation_count"]))
            table.add_row("Processing Version", stats["processing_version"])

            command.console.print(table)

        except Exception as e:
            command.console.print(f"[red]❌ Failed to get statistics: {e}[/red]")
            logger.error(f"Failed to get KG stats: {e}")
            raise typer.Exit(1)

    asyncio.run(_stats())


@app.command()
def validate() -> None:
    """Validate knowledge graph extraction service."""

    async def _validate() -> None:
        command = KGCommand()
        try:
            config = await command.get_config()
            kg_service = await command._get_kg_service(config.indexer)

            command.console.print("🔍 Validating extraction service...")

            if await kg_service.validate_extraction_service():
                command.console.print("[green]✅ Extraction service is working correctly[/green]")
            else:
                command.console.print("[red]❌ Extraction service validation failed[/red]")
                raise typer.Exit(1)

        except Exception as e:
            command.console.print(f"[red]❌ Validation failed: {e}[/red]")
            logger.error(f"KG validation failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_validate())


@app.command()
def deduplicate(
    entity_type: Optional[str] = typer.Option(None, "--type", help="Entity type to deduplicate (all if not specified)"),
    similarity_threshold: float = typer.Option(0.85, "--similarity", help="Vector similarity threshold"),
    verification_threshold: float = typer.Option(0.8, "--verification", help="LLM verification threshold"),
    batch_size: int = typer.Option(100, "--batch-size", help="Processing batch size"),
) -> None:
    """Deduplicate entities in the knowledge graph."""

    async def _deduplicate() -> None:
        command = KGCommand()
        try:
            config = await command.get_config()
            kg_service = await command._get_kg_service(config.indexer)

            if not kg_service.deduplication_service:
                command.console.print("[red]❌ Entity deduplication service not available[/red]")
                raise typer.Exit(1)

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

        except Exception as e:
            command.console.print(f"[red]❌ Entity deduplication failed: {e}[/red]")
            logger.error(f"Entity deduplication failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_deduplicate())


@app.command()
def find_duplicates(
    entity_name: str = typer.Argument(..., help="Entity name to search for duplicates"),
    entity_type: Optional[str] = typer.Option(None, "--type", help="Entity type filter"),
    similarity_threshold: float = typer.Option(0.85, "--similarity", help="Minimum similarity threshold"),
    limit: int = typer.Option(10, "--limit", help="Maximum number of results"),
) -> None:
    """Find potential duplicate entities for a given name."""

    async def _find_duplicates() -> None:
        command = KGCommand()
        try:
            config = await command.get_config()
            kg_service = await command._get_kg_service(config.indexer)

            if not kg_service.deduplication_service:
                command.console.print("[red]❌ Entity deduplication service not available[/red]")
                raise typer.Exit(1)

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
            table.add_column("Similarity", style="green")

            for entity_id, name, similarity in duplicates[:limit]:
                table.add_row(entity_id[:12] + "...", name, f"{similarity:.3f}")

            command.console.print(table)
            command.console.print(f"Found {len(duplicates)} potential duplicates")

        except Exception as e:
            command.console.print(f"[red]❌ Duplicate search failed: {e}[/red]")
            logger.error(f"Duplicate search failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_find_duplicates())


if __name__ == "__main__":
    app()
