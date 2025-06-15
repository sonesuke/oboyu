"""Knowledge Graph Property Graph CLI commands.

This module provides command-line interface for DuckPGQ property graph
operations and advanced graph analytics.
"""

import asyncio
import logging
from typing import Optional

import typer
from rich.table import Table

from oboyu.adapters.property_graph import DuckPGQPropertyGraphService
from oboyu.cli.base import BaseCommand

app = typer.Typer(help="Property Graph operations")
logger = logging.getLogger(__name__)


class PropertyGraphCommand(BaseCommand):
    """Base class for Property Graph commands."""

    def __init__(self, ctx: typer.Context) -> None:
        """Initialize property graph command."""
        super().__init__(ctx)

    async def _get_property_graph_service(self, config: dict) -> DuckPGQPropertyGraphService:
        """Get configured Property Graph service with auto-initialization."""
        # Auto-enable Property Graph on first use - no configuration required
        self.console.print("🚀 Initializing Property Graph functionality...")

        # Get database connection through indexer
        indexer_config = self.create_indexer_config()
        indexer = self.create_indexer(indexer_config, show_progress=False, show_model_loading=False)

        # Ensure database is initialized
        if not indexer.database_service._is_initialized:
            indexer.database_service.initialize()

        connection = indexer.database_service.db_manager.get_connection()

        # Initialize property graph service
        self.console.print("⚙️  Setting up DuckPGQ property graph engine...")
        pg_service = DuckPGQPropertyGraphService(connection)

        self.console.print("[green]🎉 Property Graph system ready![/green]")

        # Store the indexer reference for cleanup later
        pg_service._indexer = indexer

        return pg_service


@app.command()
def init(ctx: typer.Context) -> None:
    """Initialize DuckPGQ property graph for advanced graph queries."""

    async def _init() -> None:
        command = PropertyGraphCommand(ctx)
        try:
            config_manager = command.get_config_manager()
            config_data = config_manager.get_section("indexer")
            pg_service = await command._get_property_graph_service(config_data)

            command.console.print("🔧 Initializing DuckPGQ property graph...")

            if await pg_service.initialize_property_graph():
                command.console.print("[green]✅ Property graph initialized successfully[/green]")

                # Get basic stats
                stats = await pg_service.get_graph_statistics()
                command.console.print(f"📊 Graph contains {stats['entity_count']} entities and {stats['relation_count']} relations")
            else:
                command.console.print("[red]❌ Property graph initialization failed[/red]")
                raise typer.Exit(1)

        except Exception as e:
            command.console.print(f"[red]❌ Property graph initialization failed: {e}[/red]")
            logger.error(f"Property graph init failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_init())


@app.command()
def stats(ctx: typer.Context) -> None:
    """Show property graph statistics and metrics."""

    async def _stats() -> None:
        command = PropertyGraphCommand(ctx)
        try:
            config_manager = command.get_config_manager()
            config_data = config_manager.get_section("indexer")
            pg_service = await command._get_property_graph_service(config_data)

            if not await pg_service.is_property_graph_available():
                command.console.print("[yellow]⚠️ Property graph not available. Run 'graph init' first.[/yellow]")
                raise typer.Exit(1)

            command.console.print("📊 Calculating graph statistics...")
            stats = await pg_service.get_graph_statistics()

            # Create main statistics table
            table = Table(title="Property Graph Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Entities", str(stats["entity_count"]))
            table.add_row("Total Relations", str(stats["relation_count"]))
            table.add_row("Graph Density", f"{stats['density']:.4f}")
            table.add_row("Average Degree", f"{stats['average_degree']:.2f}")

            command.console.print(table)

            # Entity types distribution
            if "entity_types" in stats and stats["entity_types"]:
                entity_table = Table(title="Entity Types Distribution")
                entity_table.add_column("Entity Type", style="yellow")
                entity_table.add_column("Count", style="green")

                for entity_type, count in stats["entity_types"].items():
                    entity_table.add_row(entity_type, str(count))

                command.console.print(entity_table)

            # Relation types distribution
            if "relation_types" in stats and stats["relation_types"]:
                relation_table = Table(title="Relation Types Distribution")
                relation_table.add_column("Relation Type", style="yellow")
                relation_table.add_column("Count", style="green")

                for relation_type, count in stats["relation_types"].items():
                    relation_table.add_row(relation_type, str(count))

                command.console.print(relation_table)

        except Exception as e:
            command.console.print(f"[red]❌ Failed to get graph statistics: {e}[/red]")
            logger.error(f"Graph stats failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_stats())


@app.command()
def find_path(
    ctx: typer.Context,
    source: str = typer.Argument(..., help="Source entity ID or name"),
    target: str = typer.Argument(..., help="Target entity ID or name"),
    max_hops: int = typer.Option(6, "--max-hops", help="Maximum number of hops"),
) -> None:
    """Find shortest path between two entities in the knowledge graph."""

    async def _find_path() -> None:
        command = PropertyGraphCommand(ctx)
        try:
            config_manager = command.get_config_manager()
            config_data = config_manager.get_section("indexer")
            pg_service = await command._get_property_graph_service(config_data)

            if not await pg_service.is_property_graph_available():
                command.console.print("[yellow]⚠️ Property graph not available. Run 'graph init' first.[/yellow]")
                raise typer.Exit(1)

            command.console.print(f"🔍 Finding shortest path from '{source}' to '{target}'...")

            # TODO: Add entity name to ID resolution if needed
            path = await pg_service.find_shortest_path(source, target, max_hops)

            if not path:
                command.console.print("[yellow]⚠️ No path found between the specified entities[/yellow]")
                return

            command.console.print(f"[green]✅ Found path with {len(path)} steps:[/green]")

            # Display path
            table = Table(title=f"Shortest Path: {source} → {target}")
            table.add_column("Step", style="cyan")
            table.add_column("Entity", style="yellow")
            table.add_column("Relation", style="green")
            table.add_column("Confidence", style="blue")

            for i, (entity, relation) in enumerate(path, 1):
                table.add_row(str(i), f"{entity.name} ({entity.entity_type})", f"{relation.relation_type}", f"{relation.confidence:.3f}")

            command.console.print(table)

        except Exception as e:
            command.console.print(f"[red]❌ Path finding failed: {e}[/red]")
            logger.error(f"Path finding failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_find_path())


@app.command()
def centrality(
    ctx: typer.Context,
    entity_type: Optional[str] = typer.Option(None, "--type", help="Filter by entity type"),
    centrality_type: str = typer.Option("degree", "--centrality", help="Centrality type (degree, betweenness)"),
    limit: int = typer.Option(20, "--limit", help="Number of results to show"),
) -> None:
    """Calculate and display entity centrality scores."""

    async def _centrality() -> None:
        command = PropertyGraphCommand(ctx)
        try:
            config_manager = command.get_config_manager()
            config_data = config_manager.get_section("indexer")
            pg_service = await command._get_property_graph_service(config_data)

            if not await pg_service.is_property_graph_available():
                command.console.print("[yellow]⚠️ Property graph not available. Run 'graph init' first.[/yellow]")
                raise typer.Exit(1)

            command.console.print(f"📊 Calculating {centrality_type} centrality scores...")

            scores = await pg_service.get_entity_centrality_scores(entity_type=entity_type, limit=limit, centrality_type=centrality_type)

            if not scores:
                command.console.print("[yellow]⚠️ No entities found for centrality calculation[/yellow]")
                return

            # Display results
            title = f"{centrality_type.title()} Centrality Scores"
            if entity_type:
                title += f" ({entity_type})"

            table = Table(title=title)
            table.add_column("Rank", style="cyan")
            table.add_column("Entity ID", style="blue")
            table.add_column("Entity Name", style="yellow")
            table.add_column("Score", style="green")

            for i, (entity_id, entity_name, score) in enumerate(scores, 1):
                table.add_row(str(i), entity_id[:12] + "...", entity_name, f"{score:.2f}")

            command.console.print(table)

        except Exception as e:
            command.console.print(f"[red]❌ Centrality calculation failed: {e}[/red]")
            logger.error(f"Centrality calculation failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_centrality())


@app.command()
def subgraph(
    ctx: typer.Context,
    entity_id: str = typer.Argument(..., help="Central entity ID"),
    depth: int = typer.Option(2, "--depth", help="Subgraph depth"),
) -> None:
    """Extract and display subgraph around an entity."""

    async def _subgraph() -> None:
        command = PropertyGraphCommand(ctx)
        try:
            config_manager = command.get_config_manager()
            config_data = config_manager.get_section("indexer")
            pg_service = await command._get_property_graph_service(config_data)

            if not await pg_service.is_property_graph_available():
                command.console.print("[yellow]⚠️ Property graph not available. Run 'graph init' first.[/yellow]")
                raise typer.Exit(1)

            command.console.print(f"🔍 Extracting subgraph around entity '{entity_id}' (depth: {depth})...")

            subgraph_data = await pg_service.find_entity_subgraph(entity_id, depth)

            entities = subgraph_data["entities"]
            relations = subgraph_data["relations"]

            if not entities and not relations:
                command.console.print("[yellow]⚠️ No subgraph found around the specified entity[/yellow]")
                return

            # Display subgraph summary
            command.console.print("[green]✅ Subgraph extracted:[/green]")
            command.console.print(f"  • Entities: {len(entities)}")
            command.console.print(f"  • Relations: {len(relations)}")
            command.console.print(f"  • Depth: {depth}")

            # Display entities
            if entities:
                entity_table = Table(title="Entities in Subgraph")
                entity_table.add_column("ID", style="cyan")
                entity_table.add_column("Name", style="yellow")
                entity_table.add_column("Type", style="green")
                entity_table.add_column("Confidence", style="blue")

                for entity in entities[:10]:  # Limit display
                    entity_table.add_row(entity.id[:12] + "...", entity.name, entity.entity_type, f"{entity.confidence:.3f}")

                if len(entities) > 10:
                    entity_table.add_row("...", f"({len(entities) - 10} more)", "", "")

                command.console.print(entity_table)

        except Exception as e:
            command.console.print(f"[red]❌ Subgraph extraction failed: {e}[/red]")
            logger.error(f"Subgraph extraction failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_subgraph())


if __name__ == "__main__":
    app()
