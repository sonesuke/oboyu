"""Utility database query builders."""

from typing import Any, List, Tuple


class UtilityQueries:
    """Query builders for utility database operations."""

    @staticmethod
    def clear_all_data() -> List[Tuple[str, List[Any]]]:
        """Build queries to clear all data while preserving schema.

        Returns:
            List of (sql, parameters) tuples for clearing data

        """
        # Order matters for foreign key constraints
        queries: List[Tuple[str, List[Any]]] = [
            ("DELETE FROM inverted_index", []),
            ("DELETE FROM vocabulary", []),
            ("DELETE FROM document_stats", []),
            ("DELETE FROM collection_stats", []),
            ("DELETE FROM embeddings", []),
            ("DELETE FROM chunks", []),
        ]

        return queries
