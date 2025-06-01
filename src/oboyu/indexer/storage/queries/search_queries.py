"""Search-related database query builders."""

from typing import Any, List, Optional, Tuple


class SearchQueries:
    """Query builders for search database operations."""

    @staticmethod
    def search_by_bm25(query_terms: List[str], limit: int = 10, language: Optional[str] = None) -> Tuple[str, List[Any]]:
        """Build query for BM25 search.

        Args:
            query_terms: List of search terms
            limit: Maximum number of results
            language: Optional language filter

        Returns:
            Tuple of (sql, parameters)

        """
        if not query_terms:
            # Return empty result query
            return "SELECT NULL LIMIT 0", []

        # Build terms placeholder for IN clause
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

        # Add language filter if specified
        if language:
            sql += " WHERE c.language = ?"
            params.append(language)

        sql += " ORDER BY cs.score DESC"

        return sql.strip(), params
