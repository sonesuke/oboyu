"""CLI command implementations."""

from oboyu.cli.commands.health import HealthCommand
from oboyu.cli.commands.index import IndexCommand, IndexingService  # Legacy alias
from oboyu.cli.commands.query import QueryCommand, QueryService  # Legacy alias

__all__ = [
    "IndexCommand",
    "QueryCommand",
    "HealthCommand",
    # Legacy aliases for backward compatibility
    "IndexingService",
    "QueryService",
]
