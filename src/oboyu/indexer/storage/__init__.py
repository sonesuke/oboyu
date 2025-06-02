"""Storage components."""

# Import the new database service and components
# Import consolidated queries for backward compatibility
from oboyu.indexer.storage.consolidated_queries import (
    BM25Data,
    ChunkData,
    ChunkQueries,
    CollectionStatsData,
    DocumentStatsData,
    EmbeddingData,
    EmbeddingQueries,
    IndexQueries,
    QueryBuilder,
    SearchQueries,
    StatisticsQueries,
    UtilityQueries,
    VocabularyData,
)

# Import consolidated repositories (new structure)
from oboyu.indexer.storage.consolidated_repositories import (
    ChunkRepository,
    EmbeddingRepository,
    StatisticsRepository,
)
from oboyu.indexer.storage.database_manager import DatabaseManager
from oboyu.indexer.storage.database_service import DatabaseService

# Legacy imports for backward compatibility
try:
    from oboyu.indexer.storage.repositories import (
        ChunkRepository as LegacyChunkRepository,
        EmbeddingRepository as LegacyEmbeddingRepository,
        StatisticsRepository as LegacyStatisticsRepository,
    )
except ImportError:
    # If old repositories don't exist, use new ones
    LegacyChunkRepository = ChunkRepository
    LegacyEmbeddingRepository = EmbeddingRepository
    LegacyStatisticsRepository = StatisticsRepository

# Legacy alias for tests
Database = DatabaseService

__all__ = [
    "DatabaseService",
    "Database",
    "DatabaseManager",
    "ChunkRepository",
    "EmbeddingRepository",
    "StatisticsRepository",
    "QueryBuilder",
    "ChunkData",
    "EmbeddingData",
    "BM25Data",
    "VocabularyData",
    "DocumentStatsData",
    "CollectionStatsData",
    "ChunkQueries",
    "EmbeddingQueries",
    "IndexQueries",
    "SearchQueries",
    "StatisticsQueries",
    "UtilityQueries",
]
