"""Common CLI options shared across multiple Oboyu commands.

This module provides standardized option definitions to ensure
consistency across the CLI commands.
"""

from pathlib import Path
from typing import Optional

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
