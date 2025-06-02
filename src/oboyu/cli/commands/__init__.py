"""CLI command implementations."""

from oboyu.cli.commands.index import IndexCommand, IndexingService  # Legacy alias
from oboyu.cli.commands.query import QueryCommand, QueryService  # Legacy alias

__all__ = [
    "IndexCommand",
    "QueryCommand",
    # Legacy aliases for backward compatibility
    "IndexingService",
    "QueryService",
]
