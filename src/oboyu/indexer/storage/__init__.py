"""Storage components."""

# Import the new database service
from oboyu.indexer.storage.database_service import DatabaseService

# Legacy alias for tests
Database = DatabaseService

__all__ = ["DatabaseService", "Database"]
