"""Statistics-related database query builders."""

from typing import Any, List, Tuple

from .data_models import CollectionStatsData, DocumentStatsData


class StatisticsQueries:
    """Query builders for statistics database operations."""

    @staticmethod
    def upsert_document_stats(stats_data: DocumentStatsData) -> Tuple[str, List[Any]]:
        """Build query to insert or update document statistics.

        Args:
            stats_data: Document statistics to upsert

        Returns:
            Tuple of (sql, parameters)

        """
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
        """Build query to insert or update collection statistics.

        Args:
            stats_data: Collection statistics to upsert

        Returns:
            Tuple of (sql, parameters)

        """
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
        """Build query to get database statistics.

        Returns:
            Tuple of (sql, parameters)

        """
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
