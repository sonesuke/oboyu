"""Build Knowledge Graph command implementation for Oboyu CLI.

This module provides the command-line interface for building knowledge graphs
from indexed documents. Moved from oboyu kg build to top-level command.
"""

import asyncio
import logging
from typing import Optional

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sentence_transformers import SentenceTransformer
from typing_extensions import Annotated

from oboyu.adapters.entity_deduplication import EDCDeduplicationService
from oboyu.adapters.kg_extraction import LLMKGExtractionService
from oboyu.adapters.kg_repositories import DuckDBKGRepository
from oboyu.application.kg import KnowledgeGraphService
from oboyu.cli.base import BaseCommand

app = typer.Typer(help="Build knowledge graph from indexed documents")
logger = logging.getLogger(__name__)


class BuildKGCommand(BaseCommand):
    """Build Knowledge Graph command."""

    def __init__(self, ctx: typer.Context) -> None:
        """Initialize build KG command."""
        super().__init__(ctx)

    async def _get_kg_service(self, config: dict) -> KnowledgeGraphService:
        """Get configured Knowledge Graph service with auto-initialization."""
        # Auto-enable KG on first use - no configuration required
        self.console.print("🚀 Initializing Knowledge Graph functionality...")

        # Set default KG model path if not configured
        kg_model_path = config.get("kg_model_path", "SakanaAI/TinySwallow-1.5B-Instruct-GGUF")

        self.console.print(f"📦 Using model: {kg_model_path}")
        self.console.print("💡 This will download the model on first use - please be patient!")

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

        # Initialize extraction service with progress indication
        self.console.print("🤖 Loading Knowledge Graph extraction model...")
        try:
            extraction_service = LLMKGExtractionService(
                model_path=kg_model_path,
                temperature=config.get("kg_temperature", 0.1),
                max_tokens=config.get("kg_max_tokens", 512),
            )
            self.console.print("✅ Extraction model loaded successfully!")
        except Exception as e:
            self.console.print(f"[red]❌ Failed to load extraction model: {e}[/red]")
            self.console.print("💡 Make sure you have sufficient disk space and internet connection for model download")
            raise typer.Exit(1)

        # Initialize deduplication service (optional)
        self.console.print("🔗 Loading entity deduplication service...")
        deduplication_service = None
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
            self.console.print(f"[yellow]⚠️  Entity deduplication unavailable: {e}[/yellow]")
            self.console.print("💡 KG will work without deduplication, but may have duplicate entities")
            logger.warning(f"Could not initialize deduplication service: {e}")

        # Create KG service
        self.console.print("⚙️  Initializing Knowledge Graph service...")
        kg_service = KnowledgeGraphService(
            kg_repository=kg_repository,
            extraction_service=extraction_service,
            deduplication_service=deduplication_service,
            confidence_threshold=config.get("kg_confidence_threshold", 0.7),
            enable_deduplication=deduplication_service is not None,
        )

        self.console.print("[green]🎉 Knowledge Graph system ready![/green]")

        # Store the indexer reference for cleanup later
        kg_service._indexer = indexer

        return kg_service


@app.callback(invoke_without_command=True)
def build_kg(
    ctx: typer.Context,
    full: Annotated[bool, typer.Option("--full", help="Rebuild entire knowledge graph")] = False,
    batch_size: Annotated[Optional[int], typer.Option("--batch-size", help="Processing batch size")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", help="Limit number of chunks to process")] = None,
) -> None:
    """Build knowledge graph from existing chunks.

    This command extracts entities and relations from your indexed documents
    to build a knowledge graph for enhanced search capabilities.

    Examples:
        oboyu build-kg
        oboyu build-kg --full
        oboyu build-kg --batch-size 20 --limit 100

    """

    async def _build() -> None:
        command = BuildKGCommand(ctx)

        # First-time user guidance
        command.console.print("\n[bold cyan]Building Knowledge Graph from indexed documents[/bold cyan]")
        command.console.print("💡 This command will:")
        command.console.print("   • Extract entities and relations from your documents")
        command.console.print("   • Build a knowledge graph for enhanced search")
        command.console.print("   • Download required AI models (first time only)")
        command.console.print()

        try:
            config_manager = command.get_config_manager()
            config_data = config_manager.get_section("indexer")
            kg_service = await command._get_kg_service(config_data)

            # Get chunks to process
            if full:
                command.console.print("🔄 Full rebuild: Getting all chunks...")
                # Get all chunks from database
                indexer_config = command.create_indexer_config()
                indexer = command.create_indexer(indexer_config, show_progress=False, show_model_loading=False)

                # Ensure database is initialized
                if not indexer.database_service._is_initialized:
                    indexer.database_service.initialize()

                connection = indexer.database_service.db_manager.get_connection()

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
                indexer_config = command.create_indexer_config()
                indexer = command.create_indexer(indexer_config, show_progress=False, show_model_loading=False)

                # Ensure database is initialized
                if not indexer.database_service._is_initialized:
                    indexer.database_service.initialize()

                connection = indexer.database_service.db_manager.get_connection()

                placeholders = ",".join(["?" for _ in unprocessed_chunk_ids])
                query = f"SELECT id, content FROM chunks WHERE id IN ({placeholders})"
                results = connection.execute(query, unprocessed_chunk_ids).fetchall()
                chunks_to_process = [(result[1], result[0]) for result in results]

            if not chunks_to_process:
                command.console.print("[yellow]⚠️  No chunks found to process[/yellow]")
                return

            command.console.print(f"📈 Processing {len(chunks_to_process)} chunks...")

            # Configure batch size
            actual_batch_size = batch_size or config_data.get("kg_batch_size", 10)

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
                        entity_types=config_data.get("kg_entity_types", []),
                        relation_types=config_data.get("kg_relation_types", []),
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
        finally:
            # Clean up resources
            try:
                if "kg_service" in locals() and hasattr(kg_service, "_indexer"):
                    kg_service._indexer.close()
            except Exception as cleanup_error:
                logger.debug(f"Error during cleanup: {cleanup_error}")

    asyncio.run(_build())


if __name__ == "__main__":
    app()
