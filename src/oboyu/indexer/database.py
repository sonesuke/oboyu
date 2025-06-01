"""Legacy import compatibility for database module."""

# Re-export from the new location
from oboyu.indexer.storage.database_service import DatabaseService

# Legacy alias
Database = DatabaseService

__all__ = ["Database", "DatabaseService"]
