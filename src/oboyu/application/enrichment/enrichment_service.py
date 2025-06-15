"""Core enrichment service for CSV data processing.

This module provides the main EnrichmentService class that orchestrates
the enrichment of CSV data using various extraction strategies and
GraphRAG capabilities.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    import pandas as pd

from .extraction_strategies import (
    EntityExtractionStrategy,
    GraphRelationsStrategy,
    SearchContentStrategy,
)
from .protocols import GraphRAGService

logger = logging.getLogger(__name__)


class EnrichmentService:
    """Service for enriching CSV data using semantic search and GraphRAG."""

    def __init__(
        self,
        graphrag_service: Optional[GraphRAGService] = None,
        console: Optional[Console] = None,
        max_results: int = 5,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize the enrichment service.

        Args:
            graphrag_service: GraphRAG service instance for enhanced search
            console: Rich console for progress display
            max_results: Maximum search results per query
            confidence_threshold: Minimum confidence threshold for results

        """
        self.graphrag_service = graphrag_service
        self.console = console or Console()
        self.max_results = max_results
        self.confidence_threshold = confidence_threshold

        # Initialize extraction strategies
        self.strategies = {
            "search_content": SearchContentStrategy(graphrag_service, max_results, confidence_threshold),
            "entity_extraction": EntityExtractionStrategy(graphrag_service, max_results, confidence_threshold),
            "graph_relations": GraphRelationsStrategy(graphrag_service, max_results, confidence_threshold),
        }

    async def enrich_dataframe(
        self,
        df: "pd.DataFrame",
        schema: Dict[str, Any],
        batch_size: int = 10,
    ) -> "pd.DataFrame":
        """Enrich a pandas DataFrame according to the provided schema.

        Args:
            df: Input pandas DataFrame
            schema: Enrichment schema configuration
            batch_size: Number of rows to process in each batch

        Returns:
            Enriched pandas DataFrame with additional columns

        """
        enrichment_schema = schema["enrichment_schema"]

        # Initialize new columns in the dataframe
        for col_name, col_config in enrichment_schema["columns"].items():
            df[col_name] = None

        # Process rows in batches
        total_rows = len(df)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Enriching {total_rows} rows...", total=total_rows)

            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                batch_df = df.iloc[batch_start:batch_end]

                # Process batch
                await self._process_batch(batch_df, schema, batch_start)

                # Update progress
                progress.update(task, advance=batch_end - batch_start)

        return df

    async def _process_batch(
        self,
        batch_df: "pd.DataFrame",
        schema: Dict[str, Any],
        batch_start_idx: int,
    ) -> None:
        """Process a batch of rows for enrichment.

        Args:
            batch_df: Batch of rows to process
            schema: Enrichment schema configuration
            batch_start_idx: Starting index of the batch in the full dataframe

        """
        enrichment_schema = schema["enrichment_schema"]

        # Create tasks for parallel processing of columns
        column_tasks = []

        for col_name, col_config in enrichment_schema["columns"].items():
            task = self._enrich_column(batch_df, col_name, col_config, batch_start_idx)
            column_tasks.append(task)

        # Execute all column enrichment tasks in parallel
        await asyncio.gather(*column_tasks, return_exceptions=True)

    async def _enrich_column(
        self,
        batch_df: "pd.DataFrame",
        col_name: str,
        col_config: Dict[str, Any],
        batch_start_idx: int,
    ) -> None:
        """Enrich a specific column for a batch of rows.

        Args:
            batch_df: Batch of rows to process
            col_name: Name of the column to enrich
            col_config: Column configuration from schema
            batch_start_idx: Starting index of the batch in the full dataframe

        """
        strategy_name = col_config["source_strategy"]
        strategy = self.strategies.get(strategy_name)

        if not strategy:
            logger.warning(f"Unknown strategy '{strategy_name}' for column '{col_name}'")
            return

        # Process each row in the batch
        for idx, row in batch_df.iterrows():
            try:
                # Generate query from template
                query = self._format_query_template(col_config["query_template"], row)

                # Extract value using the appropriate strategy
                value = await strategy.extract_value(query, col_config, row)

                # Set the value in the dataframe using loc for proper typing
                batch_df.loc[idx, col_name] = value

            except Exception as e:
                logger.error(f"Error enriching column '{col_name}' for row {idx}: {e}")
                batch_df.loc[idx, col_name] = None

    def _format_query_template(self, template: str, row: "pd.Series[Any]") -> str:
        """Format a query template with values from the current row.

        Args:
            template: Query template with placeholder variables
            row: Current row data

        Returns:
            Formatted query string

        """
        try:
            # Simple template formatting using row values
            # Example: "{company_name} 概要" -> "トヨタ自動車 概要"
            formatted_query = template

            # Replace placeholders with actual values
            for col_name in row.index:
                placeholder = f"{{{col_name}}}"
                if placeholder in template:
                    value = str(row[col_name]) if row[col_name] is not None else ""
                    formatted_query = formatted_query.replace(placeholder, value)

            return formatted_query

        except Exception as e:
            logger.error(f"Error formatting query template '{template}': {e}")
            return template

    async def get_enrichment_summary(self, df: "pd.DataFrame", schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the enrichment results.

        Args:
            df: Enriched DataFrame
            schema: Enrichment schema configuration

        Returns:
            Dictionary containing enrichment summary statistics

        """
        summary: Dict[str, Any] = {
            "total_rows": len(df),
            "enriched_columns": {},
            "overall_completion": 0.0,
        }

        enrichment_columns = list(schema["enrichment_schema"]["columns"].keys())
        total_cells = len(df) * len(enrichment_columns)
        filled_cells = 0

        for col_name in enrichment_columns:
            if col_name in df.columns:
                non_null_count = df[col_name].notna().sum()
                completion_rate = non_null_count / len(df) if len(df) > 0 else 0.0

                enriched_columns = summary["enriched_columns"]
                enriched_columns[col_name] = {
                    "filled_rows": int(non_null_count),
                    "completion_rate": completion_rate,
                }

                filled_cells += non_null_count

        summary["overall_completion"] = filled_cells / total_cells if total_cells > 0 else 0.0

        return summary
