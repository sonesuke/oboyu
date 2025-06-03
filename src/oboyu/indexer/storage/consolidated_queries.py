"""Consolidated database query builders for all storage operations."""
# pylint: disable=too-many-lines

import json
import uuid

# Data Models
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from oboyu.common.types import Chunk

from .utils import DateTimeEncoder


@dataclass
class ChunkData:
    """Type-safe data structure for chunk database operations."""

    id: str
    path: str
    title: str
    content: str
    chunk_index: int
    language: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingData:
    """Type-safe data structure for embedding database operations."""

    id: str
    chunk_id: str
    model: str
    vector: Union[List[float], NDArray[np.float32]]
    created_at: Optional[datetime] = None


@dataclass
class BM25Data:
    """Type-safe data structure for BM25 index operations."""

    term: str
    chunk_id: str
    term_frequency: int
    positions: Optional[List[int]] = None


@dataclass
class VocabularyData:
    """Type-safe data structure for vocabulary operations."""

    term: str
    document_frequency: int
    collection_frequency: int


@dataclass
class DocumentStatsData:
    """Type-safe data structure for document statistics."""

    chunk_id: str
    total_terms: int
    unique_terms: int
    avg_term_frequency: float


@dataclass
class CollectionStatsData:
    """Type-safe data structure for collection statistics."""

    total_documents: int
    total_terms: int
    avg_document_length: float
    last_updated: Optional[datetime] = None


class ChunkQueries:
    """Query builders for chunk database operations."""

    @staticmethod
    def insert_chunk(chunk_data: ChunkData) -> Tuple[str, List[Any]]:
        """Build query to insert a chunk."""
        metadata_json = json.dumps(chunk_data.metadata, cls=DateTimeEncoder) if chunk_data.metadata else None

        sql = """
            INSERT INTO chunks (id, path, title, content, chunk_index, language, created_at, modified_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = [
            chunk_data.id,
            chunk_data.path,
            chunk_data.title,
            chunk_data.content,
            chunk_data.chunk_index,
            chunk_data.language,
            chunk_data.created_at,
            chunk_data.modified_at,
            metadata_json,
        ]

        return sql.strip(), params

    @staticmethod
    def upsert_chunk(chunk_data: ChunkData) -> Tuple[str, List[Any]]:
        """Build query to insert or update a chunk."""
        metadata_json = json.dumps(chunk_data.metadata, cls=DateTimeEncoder) if chunk_data.metadata else None

        sql = """
            INSERT INTO chunks (id, path, title, content, chunk_index, language, created_at, modified_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                path = excluded.path,
                title = excluded.title,
                content = excluded.content,
                chunk_index = excluded.chunk_index,
                language = excluded.language,
                modified_at = excluded.modified_at,
                metadata = excluded.metadata
        """

        params = [
            chunk_data.id,
            chunk_data.path,
            chunk_data.title,
            chunk_data.content,
            chunk_data.chunk_index,
            chunk_data.language,
            chunk_data.created_at,
            chunk_data.modified_at,
            metadata_json,
        ]

        return sql.strip(), params

    @staticmethod
    def select_chunk_by_id(chunk_id: str) -> Tuple[str, List[Any]]:
        """Build query to select a chunk by ID."""
        sql = """
            SELECT id, path, title, content, chunk_index, language, created_at, modified_at, metadata
            FROM chunks
            WHERE id = ?
        """

        return sql.strip(), [chunk_id]

    @staticmethod
    def select_chunks_by_path(path: str) -> Tuple[str, List[Any]]:
        """Build query to select chunks by document path."""
        sql = """
            SELECT id, path, title, content, chunk_index, language, created_at, modified_at, metadata
            FROM chunks
            WHERE path = ?
            ORDER BY chunk_index
        """

        return sql.strip(), [path]

    @staticmethod
    def delete_chunks_by_path(path: str) -> Tuple[str, List[Any]]:
        """Build query to delete chunks by path."""
        sql = "DELETE FROM chunks WHERE path = ?"
        return sql, [path]

    @staticmethod
    def chunk_from_row(row: Tuple[Any, ...]) -> Dict[str, Any]:
        """Convert database row to chunk dictionary."""
        return {
            "id": str(row[0]),
            "path": str(row[1]),
            "title": str(row[2]),
            "content": str(row[3]),
            "chunk_index": int(row[4]),
            "language": str(row[5]) if row[5] else None,
            "created_at": row[6],
            "modified_at": row[7],
            "metadata": json.loads(row[8]) if row[8] else {},
        }

    @staticmethod
    def search_result_from_row(row: Tuple[Any, ...]) -> Dict[str, Any]:
        """Convert database search result row to result dictionary."""
        return {
            "chunk_id": str(row[0]),
            "path": str(row[1]),
            "title": str(row[2]),
            "content": str(row[3]),
            "chunk_index": int(row[4]),
            "language": str(row[5]) if row[5] else None,
            "metadata": json.loads(row[6]) if row[6] else {},
            "score": float(row[7]),
        }

    @staticmethod
    def from_chunk_to_chunk_data(chunk: Chunk) -> ChunkData:
        """Convert Chunk object to ChunkData."""
        return ChunkData(
            id=chunk.id,
            path=str(chunk.path),
            title=chunk.title,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            language=chunk.language,
            created_at=chunk.created_at,
            modified_at=chunk.modified_at,
            metadata=chunk.metadata,
        )


class EmbeddingQueries:
    """Query builders for embedding database operations."""

    @staticmethod
    def insert_embedding(embedding_data: EmbeddingData) -> Tuple[str, List[Any]]:
        """Build query to insert an embedding."""
        if isinstance(embedding_data.vector, np.ndarray):
            vector_list = embedding_data.vector.tolist()
        else:
            vector_list = embedding_data.vector

        sql = """
            INSERT INTO embeddings (id, chunk_id, model, vector, created_at)
            VALUES (?, ?, ?, ?, ?)
        """

        params = [embedding_data.id, embedding_data.chunk_id, embedding_data.model, vector_list, embedding_data.created_at]

        return sql.strip(), params

    @staticmethod
    def upsert_embedding(embedding_data: EmbeddingData) -> Tuple[str, List[Any]]:
        """Build query to insert or update an embedding."""
        if isinstance(embedding_data.vector, np.ndarray):
            vector_list = embedding_data.vector.tolist()
        else:
            vector_list = embedding_data.vector

        sql = """
            INSERT INTO embeddings (id, chunk_id, model, vector, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                chunk_id = excluded.chunk_id,
                model = excluded.model,
                vector = excluded.vector,
                created_at = excluded.created_at
        """

        params = [embedding_data.id, embedding_data.chunk_id, embedding_data.model, vector_list, embedding_data.created_at]

        return sql.strip(), params

    @staticmethod
    def search_by_vector(
        query_vector: Union[List[float], NDArray[np.float32]],
        limit: int = 10,
        language: Optional[str] = None,
        embedding_dimensions: int = 256
    ) -> Tuple[str, List[Any]]:
        """Build query for vector similarity search."""
        if isinstance(query_vector, np.ndarray):
            vector_list = query_vector.tolist()
        else:
            vector_list = query_vector

        # Generate unique temporary table name
        f"temp_query_vector_{uuid.uuid4().hex}"

        # Build query with temporary table approach
        sql = f"""
            WITH query_vector AS (
                SELECT ?::{f"FLOAT[{embedding_dimensions}]"} as vec
            )
            SELECT
                c.id as chunk_id,
                c.path,
                c.title,
                c.content,
                c.chunk_index,
                c.language,
                c.metadata,
                array_distance(e.vector, q.vec) as score
            FROM chunks c
            JOIN embeddings e ON c.id = e.chunk_id
            CROSS JOIN query_vector q
        """

        params = [vector_list]

        if language:
            sql += " WHERE c.language = ?"
            params.append(language)

        sql += f" ORDER BY score ASC LIMIT {limit}"

        return sql.strip(), params


class IndexQueries:
    """Query builders for index database operations."""

    @staticmethod
    def insert_vocabulary_term(vocab_data: VocabularyData) -> Tuple[str, List[Any]]:
        """Build query to insert vocabulary term."""
        sql = """
            INSERT INTO vocabulary (term, document_frequency, collection_frequency)
            VALUES (?, ?, ?)
        """

        params = [vocab_data.term, vocab_data.document_frequency, vocab_data.collection_frequency]

        return sql.strip(), params

    @staticmethod
    def upsert_vocabulary_term(vocab_data: VocabularyData) -> Tuple[str, List[Any]]:
        """Build query to insert or update vocabulary term."""
        sql = """
            INSERT INTO vocabulary (term, document_frequency, collection_frequency)
            VALUES (?, ?, ?)
            ON CONFLICT (term) DO UPDATE SET
                document_frequency = excluded.document_frequency,
                collection_frequency = excluded.collection_frequency
        """

        params = [vocab_data.term, vocab_data.document_frequency, vocab_data.collection_frequency]

        return sql.strip(), params

    @staticmethod
    def insert_inverted_index_entry(bm25_data: BM25Data) -> Tuple[str, List[Any]]:
        """Build query to insert inverted index entry."""
        sql = """
            INSERT INTO inverted_index (term, chunk_id, term_frequency, positions)
            VALUES (?, ?, ?, ?)
        """

        params = [bm25_data.term, bm25_data.chunk_id, bm25_data.term_frequency, bm25_data.positions]

        return sql.strip(), params

    @staticmethod
    def upsert_inverted_index_entry(bm25_data: BM25Data) -> Tuple[str, List[Any]]:
        """Build query to insert or update inverted index entry."""
        sql = """
            INSERT INTO inverted_index (term, chunk_id, term_frequency, positions)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (term, chunk_id) DO UPDATE SET
                term_frequency = excluded.term_frequency,
                positions = excluded.positions
        """

        params = [bm25_data.term, bm25_data.chunk_id, bm25_data.term_frequency, bm25_data.positions]

        return sql.strip(), params


class SearchQueries:
    """Query builders for search database operations."""

    @staticmethod
    def search_by_bm25(query_terms: List[str], limit: int = 10, language: Optional[str] = None) -> Tuple[str, List[Any]]:
        """Build query for BM25 search."""
        if not query_terms:
            return "SELECT NULL LIMIT 0", []

        terms_placeholder = ",".join(["?"] * len(query_terms))

        sql = f"""
            WITH query_terms AS (
                SELECT value AS term FROM (VALUES {",".join("(?)" for _ in query_terms)})
            ),
            collection_info AS (
                SELECT
                    total_documents,
                    avg_document_length
                FROM collection_stats
                WHERE id = 1
            ),
            chunk_scores AS (
                SELECT
                    ii.chunk_id,
                    SUM(
                        -- IDF component
                        LOG((ci.total_documents - v.document_frequency + 0.5) /
                            (v.document_frequency + 0.5) + 1.0) *
                        -- Normalized TF component (k1=1.2, b=0.75)
                        ((ii.term_frequency * 2.2) /
                         (ii.term_frequency + 1.2 *
                          (0.25 + 0.75 * (ds.total_terms / ci.avg_document_length))))
                    ) AS score
                FROM inverted_index ii
                JOIN vocabulary v ON ii.term = v.term
                JOIN document_stats ds ON ii.chunk_id = ds.chunk_id
                CROSS JOIN collection_info ci
                WHERE ii.term IN ({terms_placeholder})
                GROUP BY ii.chunk_id
                ORDER BY score DESC
                LIMIT ?
            )
            SELECT
                c.id as chunk_id,
                c.path,
                c.title,
                c.content,
                c.chunk_index,
                c.language,
                c.metadata,
                cs.score
            FROM chunk_scores cs
            JOIN chunks c ON cs.chunk_id = c.id
        """

        params = query_terms + query_terms + [limit]

        if language:
            sql += " WHERE c.language = ?"
            params.append(language)

        sql += " ORDER BY cs.score DESC"

        return sql.strip(), params


class StatisticsQueries:
    """Query builders for statistics database operations."""

    @staticmethod
    def upsert_document_stats(stats_data: DocumentStatsData) -> Tuple[str, List[Any]]:
        """Build query to insert or update document statistics."""
        sql = """
            INSERT INTO document_stats (chunk_id, total_terms, unique_terms, avg_term_frequency)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (chunk_id) DO UPDATE SET
                total_terms = excluded.total_terms,
                unique_terms = excluded.unique_terms,
                avg_term_frequency = excluded.avg_term_frequency
        """

        params = [stats_data.chunk_id, stats_data.total_terms, stats_data.unique_terms, stats_data.avg_term_frequency]

        return sql.strip(), params

    @staticmethod
    def upsert_collection_stats(stats_data: CollectionStatsData) -> Tuple[str, List[Any]]:
        """Build query to insert or update collection statistics."""
        sql = """
            INSERT INTO collection_stats
            (id, total_documents, total_terms, avg_document_length, last_updated)
            VALUES (1, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
            ON CONFLICT (id) DO UPDATE SET
                total_documents = excluded.total_documents,
                total_terms = excluded.total_terms,
                avg_document_length = excluded.avg_document_length,
                last_updated = excluded.last_updated
        """

        params = [stats_data.total_documents, stats_data.total_terms, stats_data.avg_document_length, stats_data.last_updated]

        return sql.strip(), params

    @staticmethod
    def get_database_statistics() -> Tuple[str, List[Any]]:
        """Build query to get database statistics."""
        sql = """
            SELECT
                (SELECT COUNT(*) FROM chunks) as chunk_count,
                (SELECT COUNT(DISTINCT path) FROM chunks) as document_count,
                (SELECT COUNT(*) FROM embeddings) as embedding_count,
                (SELECT COUNT(*) FROM vocabulary) as vocabulary_size,
                (SELECT COUNT(*) FROM inverted_index) as inverted_index_size,
                (SELECT model FROM embeddings LIMIT 1) as embedding_model,
                (SELECT MAX(modified_at) FROM chunks) as last_updated,
                (SELECT GROUP_CONCAT(DISTINCT language) FROM chunks WHERE language IS NOT NULL) as languages
        """

        return sql.strip(), []


class UtilityQueries:
    """Query builders for utility database operations."""

    @staticmethod
    def clear_all_data() -> List[Tuple[str, List[Any]]]:
        """Build queries to clear all data while preserving schema."""
        queries: List[Tuple[str, List[Any]]] = [
            ("DELETE FROM inverted_index", []),
            ("DELETE FROM vocabulary", []),
            ("DELETE FROM document_stats", []),
            ("DELETE FROM collection_stats", []),
            ("DELETE FROM embeddings", []),
            ("DELETE FROM chunks", []),
        ]

        return queries


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

    # Index operations
    insert_vocabulary_term = IndexQueries.insert_vocabulary_term
    upsert_vocabulary_term = IndexQueries.upsert_vocabulary_term
    insert_inverted_index_entry = IndexQueries.insert_inverted_index_entry
    upsert_inverted_index_entry = IndexQueries.upsert_inverted_index_entry

    # Search operations
    search_by_bm25 = SearchQueries.search_by_bm25

    # Statistics operations
    upsert_document_stats = StatisticsQueries.upsert_document_stats
    upsert_collection_stats = StatisticsQueries.upsert_collection_stats
    get_database_statistics = StatisticsQueries.get_database_statistics

    # Utility operations
    clear_all_data = UtilityQueries.clear_all_data


# Export everything for backward compatibility
__all__ = [
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
    "DateTimeEncoder",
]
