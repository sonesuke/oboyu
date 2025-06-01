"""Unified query builder interface."""

from ..utils import DateTimeEncoder
from .chunk_queries import ChunkQueries
from .data_models import (
    BM25Data,
    ChunkData,
    CollectionStatsData,
    DocumentStatsData,
    EmbeddingData,
    VocabularyData,
)
from .embedding_queries import EmbeddingQueries


class QueryBuilder:
    """Unified query builder that delegates to specialized query classes."""

    # Chunk operations
    insert_chunk = ChunkQueries.insert_chunk
    upsert_chunk = ChunkQueries.upsert_chunk
    select_chunk_by_id = ChunkQueries.select_chunk_by_id
    select_chunks_by_path = ChunkQueries.select_chunks_by_path
    delete_chunks_by_path = ChunkQueries.delete_chunks_by_path
    chunk_from_row = ChunkQueries.chunk_from_row
    search_result_from_row = ChunkQueries.search_result_from_row
    from_chunk_to_chunk_data = ChunkQueries.from_chunk_to_chunk_data

    # Embedding operations
    insert_embedding = EmbeddingQueries.insert_embedding
    upsert_embedding = EmbeddingQueries.upsert_embedding
    search_by_vector = EmbeddingQueries.search_by_vector


# Export everything for backward compatibility
__all__ = [
    "QueryBuilder",
    "ChunkData",
    "EmbeddingData",
    "BM25Data",
    "VocabularyData",
    "DocumentStatsData",
    "CollectionStatsData",
    "DateTimeEncoder",
]
