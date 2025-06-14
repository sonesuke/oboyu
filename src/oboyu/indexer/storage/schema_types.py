"""Shared schema type definitions for Oboyu indexer.

This module provides common type definitions used across different schema modules
to avoid circular imports while maintaining type safety.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TableDefinition:
    """Definition of a database table."""

    name: str
    sql: str
    indexes: List[str]
    dependencies: List[str]  # Tables this table depends on (for foreign keys)


@dataclass
class SchemaVersion:
    """Schema version information."""

    version: str
    description: str
    migration_sql: List[str]
    rollback_sql: List[str]
