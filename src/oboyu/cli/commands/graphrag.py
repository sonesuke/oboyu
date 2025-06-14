"""GraphRAG CLI commands.

This module provides command-line interface for GraphRAG (Graph Retrieval
Augmented Generation) operations and enhanced semantic search.
"""

import asyncio
import logging

import typer
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from oboyu.adapters.graphrag import OboyuGraphRAGService
from oboyu.adapters.kg_repositories import DuckDBKGRepository
from oboyu.adapters.property_graph import DuckPGQPropertyGraphService
from oboyu.cli.base import BaseCommand
from oboyu.config.schema import IndexerConfigSchema

app = typer.Typer(help="GraphRAG operations")
logger = logging.getLogger(__name__)


class GraphRAGCommand(BaseCommand):
    """Base class for GraphRAG commands."""

    def __init__(self) -> None:
        """Initialize GraphRAG command."""
        super().__init__()
        self.console = Console()

    async def _get_graphrag_service(self, config: IndexerConfigSchema) -> OboyuGraphRAGService:
        """Get configured GraphRAG service."""
        if not config.kg_enabled:
            raise typer.BadParameter("Knowledge Graph is not enabled. Set kg_enabled=true in config.")

        # Get database connection
        database_manager = await self.get_database_manager()
        connection = database_manager.get_connection()

        # Initialize services
        kg_repository = DuckDBKGRepository(connection)
        property_graph_service = DuckPGQPropertyGraphService(connection)

        # Load embedding model
        try:
            embedding_model = SentenceTransformer(config.embedding_model)
        except Exception as e:
            raise typer.BadParameter(f"Failed to load embedding model: {e}")

        # Create GraphRAG service
        graphrag_service = OboyuGraphRAGService(
            kg_repository=kg_repository,
            property_graph_service=property_graph_service,
            embedding_model=embedding_model,
            database_connection=connection,
        )

        return graphrag_service


@app.command()
def expand_query(
    query: str = typer.Argument(..., help="Query to expand with knowledge graph entities"),
    max_entities: int = typer.Option(10, "--max-entities", help="Maximum entities to include"),
    similarity_threshold: float = typer.Option(0.7, "--similarity", help="Entity similarity threshold"),
    depth: int = typer.Option(1, "--depth", help="Entity expansion depth"),
) -> None:
    """Expand a query with relevant entities from the knowledge graph."""

    async def _expand_query() -> None:
        command = GraphRAGCommand()
        try:
            config = await command.get_config()
            graphrag_service = await command._get_graphrag_service(config.indexer)

            command.console.print(f"🔍 Expanding query: '{query}'")

            # Expand query with entities
            expansion_result = await graphrag_service.expand_query_with_entities(
                query=query,
                max_entities=max_entities,
                entity_similarity_threshold=similarity_threshold,
                expand_depth=depth,
            )

            # Display expansion results
            command.console.print("[green]✅ Query expansion complete[/green]")
            command.console.print(f"Original query: {expansion_result['original_query']}")
            command.console.print(f"Extracted candidates: {len(expansion_result['extracted_candidates'])}")
            command.console.print(f"Matched entities: {expansion_result['matched_entities']}")

            # Display expanded entities
            expanded_entities = expansion_result["expanded_entities"]
            if expanded_entities:
                entity_table = Table(title="Expanded Entities")
                entity_table.add_column("Name", style="yellow")
                entity_table.add_column("Type", style="green")
                entity_table.add_column("Relevance", style="cyan")
                entity_table.add_column("Confidence", style="blue")

                for item in expanded_entities:
                    entity = item["entity"]
                    relevance = item["relevance_score"]
                    entity_table.add_row(entity.name, entity.entity_type, f"{relevance:.3f}", f"{entity.confidence:.3f}")

                command.console.print(entity_table)

            # Display relations if any
            relations = expansion_result["relations"]
            if relations:
                command.console.print(f"\n[cyan]Found {len(relations)} related relations[/cyan]")

        except Exception as e:
            command.console.print(f"[red]❌ Query expansion failed: {e}[/red]")
            logger.error(f"Query expansion failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_expand_query())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int = typer.Option(10, "--max-results", help="Maximum number of results"),
    use_graph: bool = typer.Option(True, "--use-graph", help="Use graph expansion"),
    rerank: bool = typer.Option(True, "--rerank", help="Rerank with graph centrality"),
) -> None:
    """Perform GraphRAG-enhanced semantic search."""

    async def _search() -> None:
        command = GraphRAGCommand()
        try:
            config = await command.get_config()
            graphrag_service = await command._get_graphrag_service(config.indexer)

            command.console.print(f"🔍 Performing GraphRAG search: '{query}'")

            # Perform GraphRAG search
            results = await graphrag_service.semantic_search_with_graph_context(
                query=query,
                max_results=max_results,
                use_graph_expansion=use_graph,
                rerank_with_graph=rerank,
            )

            if not results:
                command.console.print("[yellow]⚠️ No results found[/yellow]")
                return

            command.console.print(f"[green]✅ Found {len(results)} results[/green]")

            # Display results
            for i, result in enumerate(results, 1):
                command.console.print(f"\n[cyan]Result {i}:[/cyan]")
                command.console.print(f"Chunk ID: {result['chunk_id']}")
                command.console.print(f"Relevance Score: {result['relevance_score']:.3f}")
                command.console.print(f"Search Type: {result['search_type']}")

                # Show content preview
                content_preview = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                command.console.print(f"Content: {content_preview}")

                # Show graph entities if present
                graph_entities = result.get("graph_entities", [])
                if graph_entities:
                    entity_names = [entity["name"] for entity in graph_entities[:3]]
                    command.console.print(f"Graph Entities: {', '.join(entity_names)}")

                # Show graph relations if present
                graph_relations = result.get("graph_relations", [])
                if graph_relations:
                    relation_types = [rel["type"] for rel in graph_relations[:3]]
                    command.console.print(f"Graph Relations: {', '.join(relation_types)}")

        except Exception as e:
            command.console.print(f"[red]❌ GraphRAG search failed: {e}[/red]")
            logger.error(f"GraphRAG search failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_search())


@app.command()
def explain_query(
    query: str = typer.Argument(..., help="Query to explain"),
    max_entities: int = typer.Option(5, "--max-entities", help="Maximum entities for explanation"),
) -> None:
    """Generate explanation of how a query would be processed by GraphRAG."""

    async def _explain_query() -> None:
        command = GraphRAGCommand()
        try:
            config = await command.get_config()
            graphrag_service = await command._get_graphrag_service(config.indexer)

            command.console.print(f"🔍 Analyzing query: '{query}'")

            # Expand query to understand processing
            expansion_result = await graphrag_service.expand_query_with_entities(
                query=query,
                max_entities=max_entities,
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
            command.console.print("[green]✅ Query processing explanation:[/green]")
            command.console.print(explanation)

            # Show detailed breakdown
            command.console.print("\n[cyan]Detailed breakdown:[/cyan]")
            command.console.print(f"• Extracted candidates: {expansion_result['extracted_candidates']}")
            command.console.print(f"• Found entities: {len(expanded_entities)}")
            command.console.print(f"• Found relations: {len(expansion_result['relations'])}")
            command.console.print(f"• Contextual chunks: {len(contextual_chunks)}")

        except Exception as e:
            command.console.print(f"[red]❌ Query explanation failed: {e}[/red]")
            logger.error(f"Query explanation failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_explain_query())


@app.command()
def entity_summaries(
    entity_names: str = typer.Argument(..., help="Comma-separated entity names"),
    include_relations: bool = typer.Option(True, "--include-relations", help="Include relation information"),
    max_length: int = typer.Option(200, "--max-length", help="Maximum summary length"),
) -> None:
    """Generate natural language summaries for entities."""

    async def _entity_summaries() -> None:
        command = GraphRAGCommand()
        try:
            config = await command.get_config()
            graphrag_service = await command._get_graphrag_service(config.indexer)

            names = [name.strip() for name in entity_names.split(",")]
            command.console.print(f"🔍 Generating summaries for: {', '.join(names)}")

            # Find entities by name
            all_entities = []
            for name in names:
                entities = await graphrag_service.kg_repository.search_entities_by_name(name, limit=1)
                if entities:
                    all_entities.append(entities[0])
                else:
                    command.console.print(f"[yellow]⚠️ Entity '{name}' not found[/yellow]")

            if not all_entities:
                command.console.print("[yellow]⚠️ No entities found[/yellow]")
                return

            # Generate summaries
            summaries = await graphrag_service.generate_entity_summaries(
                entities=all_entities,
                include_relations=include_relations,
                max_summary_length=max_length,
            )

            # Display summaries
            command.console.print(f"[green]✅ Generated {len(summaries)} summaries:[/green]")

            for entity in all_entities:
                if entity.id in summaries:
                    command.console.print(f"\n[cyan]{entity.name} ({entity.entity_type}):[/cyan]")
                    command.console.print(summaries[entity.id])

        except Exception as e:
            command.console.print(f"[red]❌ Summary generation failed: {e}[/red]")
            logger.error(f"Summary generation failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_entity_summaries())


@app.command()
def find_clusters(
    seed_entities: str = typer.Argument(..., help="Comma-separated seed entity names"),
    threshold: float = typer.Option(0.8, "--threshold", help="Clustering similarity threshold"),
    max_cluster_size: int = typer.Option(15, "--max-size", help="Maximum cluster size"),
) -> None:
    """Find clusters of related entities."""

    async def _find_clusters() -> None:
        command = GraphRAGCommand()
        try:
            config = await command.get_config()
            graphrag_service = await command._get_graphrag_service(config.indexer)

            names = [name.strip() for name in seed_entities.split(",")]
            command.console.print(f"🔍 Finding clusters for: {', '.join(names)}")

            # Find seed entities
            seed_entity_objects = []
            for name in names:
                entities = await graphrag_service.kg_repository.search_entities_by_name(name, limit=1)
                if entities:
                    seed_entity_objects.append(entities[0])

            if not seed_entity_objects:
                command.console.print("[yellow]⚠️ No seed entities found[/yellow]")
                return

            # Find clusters
            clusters = await graphrag_service.find_entity_clusters(
                query_entities=seed_entity_objects,
                clustering_threshold=threshold,
                max_cluster_size=max_cluster_size,
            )

            if not clusters:
                command.console.print("[yellow]⚠️ No clusters found[/yellow]")
                return

            # Display clusters
            command.console.print(f"[green]✅ Found {len(clusters)} clusters:[/green]")

            for i, cluster in enumerate(clusters, 1):
                command.console.print(f"\n[cyan]Cluster {i} ({len(cluster)} entities):[/cyan]")

                cluster_table = Table()
                cluster_table.add_column("Name", style="yellow")
                cluster_table.add_column("Type", style="green")
                cluster_table.add_column("Confidence", style="blue")

                for entity in cluster:
                    cluster_table.add_row(entity.name, entity.entity_type, f"{entity.confidence:.3f}")

                command.console.print(cluster_table)

        except Exception as e:
            command.console.print(f"[red]❌ Cluster finding failed: {e}[/red]")
            logger.error(f"Cluster finding failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_find_clusters())


if __name__ == "__main__":
    app()
