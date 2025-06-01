"""Index-related database query builders."""

from typing import Any, List, Tuple

from .data_models import BM25Data, VocabularyData


class IndexQueries:
    """Query builders for index database operations."""

    @staticmethod
    def insert_vocabulary_term(vocab_data: VocabularyData) -> Tuple[str, List[Any]]:
        """Build query to insert vocabulary term.

        Args:
            vocab_data: Vocabulary data to insert

        Returns:
            Tuple of (sql, parameters)

        """
        sql = """
            INSERT INTO vocabulary (term, document_frequency, collection_frequency)
            VALUES (?, ?, ?)
        """

        params = [vocab_data.term, vocab_data.document_frequency, vocab_data.collection_frequency]

        return sql.strip(), params

    @staticmethod
    def upsert_vocabulary_term(vocab_data: VocabularyData) -> Tuple[str, List[Any]]:
        """Build query to insert or update vocabulary term.

        Args:
            vocab_data: Vocabulary data to upsert

        Returns:
            Tuple of (sql, parameters)

        """
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
        """Build query to insert inverted index entry.

        Args:
            bm25_data: BM25 index data to insert

        Returns:
            Tuple of (sql, parameters)

        """
        sql = """
            INSERT INTO inverted_index (term, chunk_id, term_frequency, positions)
            VALUES (?, ?, ?, ?)
        """

        params = [bm25_data.term, bm25_data.chunk_id, bm25_data.term_frequency, bm25_data.positions]

        return sql.strip(), params

    @staticmethod
    def upsert_inverted_index_entry(bm25_data: BM25Data) -> Tuple[str, List[Any]]:
        """Build query to insert or update inverted index entry.

        Args:
            bm25_data: BM25 index data to upsert

        Returns:
            Tuple of (sql, parameters)

        """
        sql = """
            INSERT INTO inverted_index (term, chunk_id, term_frequency, positions)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (term, chunk_id) DO UPDATE SET
                term_frequency = excluded.term_frequency,
                positions = excluded.positions
        """

        params = [bm25_data.term, bm25_data.chunk_id, bm25_data.term_frequency, bm25_data.positions]

        return sql.strip(), params
