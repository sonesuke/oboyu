"""CSV enrichment command implementation for Oboyu CLI.

This module provides a command-line interface for enriching CSV data using
Oboyu's semantic search and GraphRAG capabilities. It takes CSV files and
JSON schema as input to automatically populate additional columns with
relevant information from the indexed knowledge base.
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

# Create console for rich output
console = Console()
logger = logging.getLogger(__name__)


def enrich(
    ctx: typer.Context,
    csv_file: Annotated[Path, typer.Argument(help="Input CSV file to enrich")],
    schema_file: Annotated[Path, typer.Argument(help="JSON schema file defining enrichment configuration")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output CSV file path")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Processing batch size")] = 10,
    max_results: Annotated[int, typer.Option("--max-results", help="Maximum search results per query")] = 5,
    confidence: Annotated[float, typer.Option("--confidence", help="Minimum confidence threshold")] = 0.5,
    no_graph: Annotated[bool, typer.Option("--no-graph", help="Disable GraphRAG enhancement")] = False,
    db_path: Annotated[Optional[Path], typer.Option("--db-path", "-p", help="Path to database file")] = None,
) -> None:
    """Enrich CSV data using semantic search and GraphRAG capabilities.

    This command takes a CSV file and a JSON schema configuration to automatically
    populate additional columns with relevant information from the indexed knowledge base.

    The enrichment process leverages Oboyu's hybrid search (vector + BM25) and GraphRAG
    functionality to find relevant information and extract specific data points.

    Examples:
        oboyu enrich companies.csv schema.json
        oboyu enrich companies.csv schema.json --output enriched.csv
        oboyu enrich data.csv config.json --batch-size 5 --confidence 0.7
        oboyu enrich data.csv config.json --no-graph --max-results 3

    """
    # Create base command for common functionality
    base_command = BaseCommand(ctx)

    try:
        # Validate input files
        if not csv_file.exists():
            raise typer.BadParameter(f"CSV file not found: {csv_file}")

        if not schema_file.exists():
            raise typer.BadParameter(f"Schema file not found: {schema_file}")

        # Set default output path if not specified
        if output is None:
            output = csv_file.parent / f"{csv_file.stem}_enriched.csv"

        # Load and validate schema
        with open(schema_file, "r", encoding="utf-8") as f:
            schema_dict = json.load(f)

        # Validate schema structure using Pydantic
        from oboyu.config.enrichment_schema import validate_enrichment_config

        schema = validate_enrichment_config(schema_dict)

        # Execute enrichment
        asyncio.run(
            _execute_enrichment(
                base_command=base_command,
                csv_file=csv_file,
                output_file=output,
                schema=schema.model_dump(),  # Convert Pydantic model to dict
                batch_size=batch_size,
                max_results=max_results,
                confidence=confidence,
                use_graph=not no_graph,
                db_path=db_path,
            )
        )

        base_command.console.print(f"✅ Enrichment completed successfully! Output saved to: {output}")

    except Exception as e:
        base_command.console.print(f"❌ Enrichment failed: {e}", style="red")
        logger.error(f"Enrichment error: {e}")
        raise typer.Exit(1)


async def _execute_enrichment(
    base_command: BaseCommand,
    csv_file: Path,
    output_file: Path,
    schema: dict,
    batch_size: int,
    max_results: int,
    confidence: float,
    use_graph: bool,
    db_path: Optional[Path],
) -> None:
    """Execute the CSV enrichment process."""
    # Import pandas here to avoid startup overhead
    try:
        import pandas as pd
    except ImportError:
        raise typer.BadParameter("pandas is required for CSV processing. Install with: uv add pandas")

    # Load CSV data
    base_command.console.print(f"📁 Loading CSV file: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        base_command.console.print(f"📊 Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        raise typer.BadParameter(f"Failed to load CSV file: {e}")

    # Validate CSV columns match schema
    input_columns = set(schema["input_schema"]["columns"].keys())
    csv_columns = set(df.columns)
    missing_columns = input_columns - csv_columns

    if missing_columns:
        raise typer.BadParameter(f"CSV file missing required columns: {missing_columns}")

    # Initialize GraphRAG service if enabled
    graphrag_service = None
    if use_graph:
        base_command.console.print("🚀 Initializing GraphRAG service...")
        graphrag_service = await _get_graphrag_service(base_command)

    # Initialize enrichment service
    from oboyu.application.enrichment import EnrichmentService

    enrichment_service = EnrichmentService(
        graphrag_service=graphrag_service,
        console=base_command.console,
        max_results=max_results,
        confidence_threshold=confidence,
    )

    # Process enrichment
    base_command.console.print("🔍 Starting enrichment process...")

    try:
        enriched_df = await enrichment_service.enrich_dataframe(
            df=df,
            schema=schema,
            batch_size=batch_size,
        )

        # Save enriched data
        base_command.console.print(f"💾 Saving enriched data to: {output_file}")
        enriched_df.to_csv(output_file, index=False)

        # Show summary
        new_columns = set(enriched_df.columns) - set(df.columns)
        base_command.console.print(f"📈 Added {len(new_columns)} new columns: {new_columns}")

    finally:
        # Clean up resources
        if graphrag_service and hasattr(graphrag_service, "_indexer"):
            try:
                graphrag_service._indexer.close()
            except Exception as cleanup_error:
                logger.debug(f"Error during cleanup: {cleanup_error}")


async def _get_graphrag_service(base_command: BaseCommand) -> OboyuGraphRAGService:
    """Get configured GraphRAG service with auto-initialization."""
    # Get configuration manager
    config_manager = base_command.get_config_manager()
    config_data = config_manager.get_section("indexer")

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
        embedding_model_name = config_data.get("embedding_model", "all-MiniLM-L6-v2")
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
