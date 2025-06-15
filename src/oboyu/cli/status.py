"""Enhanced status command implementation for Oboyu CLI.

This module provides the command-line interface for showing unified
index and knowledge graph status information.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich.table import Table
from typing_extensions import Annotated

from oboyu.adapters.kg_repositories import DuckDBKGRepository
from oboyu.cli.base import BaseCommand
from oboyu.cli.common_options import ConfigOption
from oboyu.cli.services.indexing_service import IndexingService

DirectoryOption = Annotated[
    Optional[List[Path]],
    typer.Argument(
        help="Directories to check status for (optional - shows overall stats if not provided)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
]

logger = logging.getLogger(__name__)


def status(
    ctx: typer.Context,
    directories: DirectoryOption = None,
    config: ConfigOption = None,
    db_path: Annotated[
        Optional[Path],
        typer.Option(
            "--db-path",
            "-p",
            help="Path to the database file (default: from config)",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    detailed: Annotated[
        bool,
        typer.Option("--detailed", "-d", help="Show detailed file-by-file status"),
    ] = False,
    show_kg: Annotated[
        bool,
        typer.Option("--kg", help="Show knowledge graph statistics"),
    ] = True,
) -> None:
    """Show unified index and knowledge graph status.

    This command shows indexing status for directories (if provided) and
    overall statistics including knowledge graph information.

    Examples:
        oboyu status                    # Show overall stats
        oboyu status ~/documents        # Show status for specific directory
        oboyu status --detailed ~/docs  # Show detailed file-by-file status
        oboyu status --no-kg           # Show only index stats, skip KG

    """

    async def _status() -> None:
        base_command = BaseCommand(ctx)
        config_manager = base_command.get_config_manager()
        indexing_service = IndexingService(config_manager, base_command.services.indexer_factory, base_command.services.console_manager)

        # Get database path and display it
        resolved_db_path = indexing_service.get_database_path(db_path)
        base_command.print_database_path(resolved_db_path)

        # Show overall database statistics first
        await _show_database_stats(base_command, show_kg)

        # If directories are provided, show directory-specific status
        if directories:
            base_command.console.print("\n[bold cyan]Directory Status[/bold cyan]")

            # Get status for all directories
            status_results = indexing_service.get_status(directories, db_path)

            for status_result in status_results:
                base_command.console.print(f"\n[bold]Status for {status_result.directory}:[/bold]")
                base_command.console.print(f"  New files: {status_result.new_files}")
                base_command.console.print(f"  Modified files: {status_result.modified_files}")
                base_command.console.print(f"  Deleted files: {status_result.deleted_files}")
                base_command.console.print(f"  Total indexed: {status_result.total_indexed}")

                if detailed:
                    # Get detailed diff for this directory
                    diff_results = indexing_service.get_diff([status_result.directory], db_path)
                    if diff_results:
                        diff_result = diff_results[0]

                        if diff_result.new_files:
                            base_command.console.print("\n  [green]New files:[/green]")
                            for f in diff_result.new_files[:10]:  # Show first 10
                                base_command.console.print(f"    + {f}")
                            if len(diff_result.new_files) > 10:
                                base_command.console.print(f"    ... and {len(diff_result.new_files) - 10} more")

                        if diff_result.modified_files:
                            base_command.console.print("\n  [yellow]Modified files:[/yellow]")
                            for f in diff_result.modified_files[:10]:  # Show first 10
                                base_command.console.print(f"    ~ {f}")
                            if len(diff_result.modified_files) > 10:
                                base_command.console.print(f"    ... and {len(diff_result.modified_files) - 10} more")

                        if diff_result.deleted_files:
                            base_command.console.print("\n  [red]Deleted files:[/red]")
                            for f in diff_result.deleted_files[:10]:  # Show first 10
                                base_command.console.print(f"    - {f}")
                            if len(diff_result.deleted_files) > 10:
                                base_command.console.print(f"    ... and {len(diff_result.deleted_files) - 10} more")

    asyncio.run(_status())


async def _show_database_stats(base_command: BaseCommand, show_kg: bool) -> None:
    """Show unified database statistics including index and KG stats."""
    try:
        # Get database connection
        indexer_config = base_command.create_indexer_config()
        indexer = base_command.create_indexer(indexer_config, show_progress=False, show_model_loading=False)

        if not indexer.database_service._is_initialized:
            indexer.database_service.initialize()

        connection = indexer.database_service.db_manager.get_connection()

        # Create overall statistics table
        table = Table(title="Oboyu Database Statistics")
        table.add_column("Category", style="cyan", width=20)
        table.add_column("Metric", style="yellow", width=25)
        table.add_column("Value", style="green", width=15)

        # Index statistics
        try:
            chunks_result = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
            if chunks_result:
                chunks_count = chunks_result[0]
                table.add_row("Index", "Total Chunks", str(chunks_count))

            # Get file count if files table exists
            try:
                files_result = connection.execute("SELECT COUNT(DISTINCT file_path) FROM chunks").fetchone()
                if files_result:
                    files_count = files_result[0]
                    table.add_row("", "Unique Files", str(files_count))
            except Exception:
                logger.debug("Could not get file count from chunks table")

            # Get database size (PostgreSQL function - skip for DuckDB)
            try:
                db_size = connection.execute("SELECT pg_size_pretty(pg_database_size(current_database()))").fetchone()
                if db_size:
                    table.add_row("", "Database Size", str(db_size[0]))
            except Exception:
                # DuckDB doesn't have pg_size_pretty function
                logger.debug("Database size query not supported")

        except Exception as e:
            table.add_row("Index", "Status", "Error")
            logger.debug(f"Error getting index stats: {e}")

        # Knowledge Graph statistics
        if show_kg:
            try:
                # Initialize KG repository for potential future use
                _ = DuckDBKGRepository(connection)

                # Check if KG tables exist
                tables_result = connection.execute("SELECT table_name FROM information_schema.tables WHERE table_name IN ('entities', 'relations')").fetchall()

                if len(tables_result) >= 2:
                    # Get entity count
                    try:
                        entity_result = connection.execute("SELECT COUNT(*) FROM entities").fetchone()
                        if entity_result:
                            entity_count = entity_result[0]
                            table.add_row("Knowledge Graph", "Total Entities", str(entity_count))
                    except Exception:
                        table.add_row("Knowledge Graph", "Entities", "N/A")

                    # Get relation count
                    try:
                        relation_result = connection.execute("SELECT COUNT(*) FROM relations").fetchone()
                        if relation_result:
                            relation_count = relation_result[0]
                            table.add_row("", "Total Relations", str(relation_count))
                    except Exception:
                        table.add_row("", "Relations", "N/A")

                    # Get entity types
                    try:
                        entity_types = connection.execute(
                            "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type ORDER BY COUNT(*) DESC LIMIT 5"
                        ).fetchall()

                        if entity_types:
                            table.add_row("", "Top Entity Types", "")
                            for entity_type, count in entity_types:
                                table.add_row("", f"  {entity_type}", str(count))
                    except Exception:
                        logger.debug("Could not get entity type statistics")

                else:
                    table.add_row("Knowledge Graph", "Status", "Not Built")
                    table.add_row("", "Hint", "Run 'oboyu build-kg'")

            except Exception as e:
                table.add_row("Knowledge Graph", "Status", "Error")
                logger.debug(f"Error getting KG stats: {e}")

        base_command.console.print(table)

        # Clean up
        indexer.close()

    except Exception as e:
        base_command.console.print(f"[red]❌ Failed to get database statistics: {e}[/red]")
        logger.error(f"Database stats error: {e}")
