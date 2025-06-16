"""Entity embedding management command for Oboyu CLI.

This module provides commands to manage pre-computed entity embeddings
for enhanced EDC (Extract-Define-Canonicalize) performance.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from oboyu.adapters.kg_repositories import DuckDBKGRepository
from oboyu.application.services.entity_embedding_service import EntityEmbeddingService
from oboyu.cli.base import BaseCommand
from oboyu.cli.common_options import ConfigOption
from oboyu.indexer.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="embeddings",
    help="Manage entity embeddings for enhanced EDC performance",
    add_completion=False,
)


class EntityEmbeddingCommand(BaseCommand):
    """Base command for entity embedding operations."""

    def __init__(self, config: dict, db_path: Optional[Path] = None):
        """Initialize embedding command.
        
        Args:
            config: Configuration dictionary
            db_path: Optional database path override
        """
        super().__init__(config, db_path)
        self.entity_embedding_service: Optional[EntityEmbeddingService] = None

    async def _setup_services(self) -> None:
        """Set up required services."""
        await super()._setup_services()
        
        # Initialize embedding service
        embedding_config = self.config.get("kg", {}).get("embedding", {})
        embedding_model = embedding_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        batch_size = embedding_config.get("batch_size", 100)
        
        try:
            embedding_service = EmbeddingService(
                model_name=embedding_model,
                batch_size=batch_size,
                query_prefix="",  # No prefix for entity embeddings
                use_cache=True,
            )
            
            self.entity_embedding_service = EntityEmbeddingService(
                kg_repository=self.kg_repository,
                embedding_service=embedding_service,
                embedding_model=embedding_model,
                batch_size=batch_size,
            )
            
            logger.info(f"Initialized EntityEmbeddingService with model: {embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise typer.Exit(1)


@app.command("build")
def build_embeddings(
    ctx: typer.Context,
    config: ConfigOption = None,
    db_path: Annotated[
        Optional[Path],
        typer.Option(
            "--db-path",
            "-p",
            help="Path to the database file",
            exists=False,
            resolve_path=True,
        ),
    ] = None,
    entity_type: Annotated[
        Optional[str],
        typer.Option(
            "--type",
            "-t",
            help="Build embeddings only for entities of specific type",
        ),
    ] = None,
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild",
            help="Rebuild all embeddings (ignores existing ones)",
        ),
    ] = False,
    update_stale: Annotated[
        bool,
        typer.Option(
            "--update-stale",
            help="Update only stale embeddings",
        ),
    ] = False,
    max_age_days: Annotated[
        int,
        typer.Option(
            "--max-age-days",
            help="Maximum age in days for embeddings (used with --update-stale)",
        ),
    ] = 30,
) -> None:
    """Build or update entity embeddings for enhanced EDC performance."""
    
    async def run_build() -> None:
        config_data = ctx.obj["config_data"]
        command = EntityEmbeddingCommand(config_data, db_path)
        
        try:
            await command._setup_services()
            
            if not command.entity_embedding_service:
                console.print("[red]Failed to initialize embedding service[/red]")
                raise typer.Exit(1)
            
            if update_stale:
                console.print(f"[yellow]Updating stale embeddings (older than {max_age_days} days)...[/yellow]")
                updated_count = await command.entity_embedding_service.update_stale_embeddings(max_age_days)
                console.print(f"[green]Updated {updated_count} stale embeddings[/green]")
                
            elif rebuild:
                console.print("[yellow]Rebuilding all embeddings...[/yellow]")
                rebuilt_count = await command.entity_embedding_service.rebuild_all_embeddings(entity_type)
                console.print(f"[green]Rebuilt {rebuilt_count} embeddings[/green]")
                
            else:
                # Build embeddings for entities that don't have them
                console.print("[yellow]Building embeddings for entities without them...[/yellow]")
                
                if entity_type:
                    entities = await command.kg_repository.find_entities_by_type(entity_type)
                    console.print(f"Found {len(entities)} entities of type '{entity_type}'")
                else:
                    entities = await command.kg_repository.get_all_entities()
                    console.print(f"Found {len(entities)} total entities")
                
                if entities:
                    computed_count = await command.entity_embedding_service.batch_compute_embeddings(
                        entities, skip_existing=True
                    )
                    console.print(f"[green]Computed {computed_count} new embeddings[/green]")
                else:
                    console.print("[yellow]No entities found to process[/yellow]")
            
        except Exception as e:
            logger.error(f"Embedding build failed: {e}")
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
        finally:
            await command.cleanup()

    asyncio.run(run_build())


@app.command("stats")
def embedding_stats(
    ctx: typer.Context,
    config: ConfigOption = None,
    db_path: Annotated[
        Optional[Path],
        typer.Option(
            "--db-path",
            "-p",
            help="Path to the database file",
            exists=False,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Show entity embedding statistics."""
    
    async def run_stats() -> None:
        config_data = ctx.obj["config_data"]
        command = EntityEmbeddingCommand(config_data, db_path)
        
        try:
            await command._setup_services()
            
            if not command.entity_embedding_service:
                console.print("[red]Failed to initialize embedding service[/red]")
                raise typer.Exit(1)
            
            # Get embedding statistics
            stats = await command.entity_embedding_service.get_embedding_statistics()
            
            # Create statistics table
            table = Table(title="Entity Embedding Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            # Add basic statistics
            table.add_row("Total Entities", str(stats.get("total_entities", "N/A")))
            table.add_row("Entities with Embeddings", str(stats.get("entities_with_embeddings", "N/A")))
            table.add_row("Entities without Embeddings", str(stats.get("entities_without_embeddings", "N/A")))
            table.add_row("Coverage Percentage", f"{stats.get('embedding_coverage_percent', 0):.1f}%")
            table.add_row("Current Model", str(stats.get("current_model", "N/A")))
            table.add_row("Embedding Dimensions", str(stats.get("embedding_dimensions", "N/A")))
            
            # Add model-specific statistics
            for key, value in stats.items():
                if key.startswith("model_"):
                    model_name = key.replace("model_", "")
                    table.add_row(f"Model '{model_name}' Count", str(value))
            
            console.print(table)
            
            # Show error if any
            if "error" in stats:
                console.print(f"[red]Warning: {stats['error']}[/red]")
            
        except Exception as e:
            logger.error(f"Failed to get embedding statistics: {e}")
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
        finally:
            await command.cleanup()

    asyncio.run(run_stats())


@app.command("clear")
def clear_cache(
    ctx: typer.Context,
    config: ConfigOption = None,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Confirm cache clearing without prompting",
        ),
    ] = False,
) -> None:
    """Clear embedding cache files."""
    
    if not confirm:
        if not typer.confirm("Are you sure you want to clear the embedding cache?"):
            console.print("[yellow]Cache clearing cancelled[/yellow]")
            raise typer.Exit()
    
    try:
        # Initialize embedding service to access cache
        config_data = ctx.obj["config_data"]
        embedding_config = config_data.get("kg", {}).get("embedding", {})
        embedding_model = embedding_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        
        embedding_service = EmbeddingService(
            model_name=embedding_model,
            use_cache=True,
        )
        
        # Clear the cache
        embedding_service.clear_cache()
        console.print("[green]Embedding cache cleared successfully[/green]")
        
    except Exception as e:
        logger.error(f"Failed to clear embedding cache: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()