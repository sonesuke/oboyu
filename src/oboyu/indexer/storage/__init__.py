"""Storage components."""

# Import the new database service and components
from oboyu.indexer.storage.database_manager import DatabaseManager
from oboyu.indexer.storage.database_service import DatabaseService
from oboyu.indexer.storage.repositories import ChunkRepository, EmbeddingRepository, StatisticsRepository

# Legacy alias for tests
Database = DatabaseService

__all__ = [
    "DatabaseService",
    "Database",
    "DatabaseManager",
    "ChunkRepository",
    "EmbeddingRepository",
    "StatisticsRepository",
]
