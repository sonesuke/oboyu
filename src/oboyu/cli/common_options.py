"""Common CLI options shared across multiple Oboyu commands.

This module provides standardized option definitions to ensure
consistency across the CLI commands.
"""

from pathlib import Path
from typing import Any, List, Optional

import typer
from typing_extensions import Annotated

# Database path option used by multiple commands
DatabasePathOption = Annotated[
    Optional[Path],
    typer.Option(
        "--db-path",
        help="Path to database file",
        envvar="OBOYU_DB_PATH",
    ),
]

# Verbose output option
VerboseOption = Annotated[
    bool,
    typer.Option(
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
]

# Force operation option
ForceOption = Annotated[
    bool,
    typer.Option(
        "--force",
        "-f",
        help="Force operation without confirmation",
    ),
]

# Debug mode option
DebugOption = Annotated[
    bool,
    typer.Option(
        "--debug",
        "-d",
        help="Enable debug mode with additional logging",
    ),
]

# Configuration file option
ConfigOption = Annotated[
    Optional[Path],
    typer.Option(
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
]

# Interactive mode option
InteractiveOption = Annotated[
    bool,
    typer.Option(
        "--interactive",
        "-i",
        help="Start interactive mode for continuous operations",
    ),
]

# Top-K results option for search commands
TopKOption = Annotated[
    Optional[int],
    typer.Option(
        "--top-k",
        "-k",
        help="Number of results to return",
        min=1,
    ),
]

# Search mode option
SearchModeOption = Annotated[
    str,
    typer.Option(
        "--mode",
        "-m",
        help="Search mode (vector, bm25, hybrid)",
    ),
]

# Include patterns option for file processing
IncludePatternsOption = Annotated[
    Optional[List[str]],
    typer.Option(
        "--include-patterns",
        "-i",
        help="File patterns to include (e.g., '*.txt,*.md')",
    ),
]

# Exclude patterns option for file processing
ExcludePatternsOption = Annotated[
    Optional[List[str]],
    typer.Option(
        "--exclude-patterns",
        "-e",
        help="File patterns to exclude (e.g., '*/node_modules/*')",
    ),
]

# Recursive processing option
RecursiveOption = Annotated[
    Optional[bool],
    typer.Option(
        "--recursive/--no-recursive",
        "-r/-nr",
        help="Process directories recursively",
    ),
]

# Maximum depth option for recursive processing
MaxDepthOption = Annotated[
    Optional[int],
    typer.Option(
        "--max-depth",
        "-d",
        help="Maximum recursion depth",
        min=1,
    ),
]

# Chunk size option for text processing
ChunkSizeOption = Annotated[
    Optional[int],
    typer.Option(
        "--chunk-size",
        help="Chunk size in characters",
        min=64,
    ),
]

# Chunk overlap option for text processing
ChunkOverlapOption = Annotated[
    Optional[int],
    typer.Option(
        "--chunk-overlap",
        help="Chunk overlap in characters",
        min=0,
    ),
]

# Embedding model option
EmbeddingModelOption = Annotated[
    Optional[str],
    typer.Option(
        "--embedding-model",
        help="Embedding model to use",
    ),
]

# Vector weight option for hybrid search (DEPRECATED - RRF doesn't use weights)
VectorWeightOption = Annotated[
    Optional[float],
    typer.Option(
        "--vector-weight",
        help="DEPRECATED: Weight for vector scores in hybrid search. RRF algorithm is now used instead.",
        min=0.0,
        max=1.0,
    ),
]

# BM25 weight option for hybrid search (DEPRECATED - RRF doesn't use weights)
BM25WeightOption = Annotated[
    Optional[float],
    typer.Option(
        "--bm25-weight",
        help="DEPRECATED: Weight for BM25 scores in hybrid search. RRF algorithm is now used instead.",
        min=0.0,
        max=1.0,
    ),
]

# Reranking option for search commands
RerankOption = Annotated[
    Optional[bool],
    typer.Option(
        "--rerank/--no-rerank",
        help="Enable or disable reranking of search results",
    ),
]

# Explain option for detailed output
ExplainOption = Annotated[
    bool,
    typer.Option(
        "--explain",
        "-e",
        help="Show detailed match explanation",
    ),
]

# Format option for output formatting
FormatOption = Annotated[
    str,
    typer.Option(
        "--format",
        "-f",
        help="Output format (text, json)",
    ),
]


class CommonOptions:
    """Centralized option definitions and validation for CLI commands."""

    @staticmethod
    def database_path_option(help_text: str = "Path to database file") -> Any:  # noqa: ANN401
        """Create a standardized database path option.

        Args:
            help_text: Custom help text for the option

        Returns:
            Annotated type for database path option

        """
        return Annotated[
            Optional[Path],
            typer.Option(
                "--db-path",
                help=help_text,
                envvar="OBOYU_DB_PATH",
            ),
        ]

    @staticmethod
    def config_option(help_text: str = "Path to configuration file") -> Any:  # noqa: ANN401
        """Create a standardized config option.

        Args:
            help_text: Custom help text for the option

        Returns:
            Annotated type for config option

        """
        return Annotated[
            Optional[Path],
            typer.Option(
                "--config",
                "-c",
                help=help_text,
                exists=True,
                file_okay=True,
                dir_okay=False,
                readable=True,
            ),
        ]

    @staticmethod
    def verbose_option() -> type:
        """Create a standardized verbose option.

        Returns:
            Annotated type for verbose option

        """
        return VerboseOption

    @staticmethod
    def force_option() -> type:
        """Create a standardized force option.

        Returns:
            Annotated type for force option

        """
        return ForceOption
