"""Embedding functionality mixin for DuckDB KG repository.

This module contains embedding-related methods for the DuckDB KG repository
to keep the main repository file under the line limit.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from oboyu.domain.models.knowledge_graph import Entity
from oboyu.ports.repositories.kg_repository import RepositoryError

# Base imports removed to avoid MRO conflicts

logger = logging.getLogger(__name__)


class EmbeddingRepositoryMixin:
    """Mixin providing embedding-related repository operations."""

    async def find_entities_with_stale_embeddings(self, cutoff_date: datetime, embedding_model: str) -> List[Entity]:
        """Find entities with embeddings older than cutoff date or different model."""
        try:
            results = self.connection.execute(
                """
                SELECT * FROM kg_entities
                WHERE embedding IS NULL
                   OR embedding_model IS NULL
                   OR embedding_model != ?
                   OR embedding_updated_at IS NULL
                   OR embedding_updated_at < ?
                ORDER BY created_at
                """,
                (embedding_model, cutoff_date.isoformat()),
            ).fetchall()

            if not results:
                return []

            columns = self._get_columns_safely(self.connection, "find_entities_with_stale_embeddings")
            entities = []
            for result in results:
                row_dict = dict(zip(columns, result))
                entities.append(self._entity_from_row(row_dict))

            return entities
        except Exception as e:
            logger.error(f"Failed to find entities with stale embeddings: {e}")
            raise RepositoryError(f"Failed to find entities with stale embeddings: {e}", "find_entities_with_stale_embeddings")

    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about entity embeddings."""
        try:
            # Total entities
            total_result = self.connection.execute("SELECT COUNT(*) FROM kg_entities").fetchone()
            total_entities = total_result[0] if total_result else 0

            # Entities with embeddings
            with_embeddings_result = self.connection.execute("SELECT COUNT(*) FROM kg_entities WHERE embedding IS NOT NULL").fetchone()
            with_embeddings = with_embeddings_result[0] if with_embeddings_result else 0

            # Entities without embeddings
            without_embeddings = total_entities - with_embeddings

            # Entities by model
            model_stats_result = self.connection.execute(
                """
                SELECT embedding_model, COUNT(*)
                FROM kg_entities
                WHERE embedding_model IS NOT NULL
                GROUP BY embedding_model
                """
            ).fetchall()

            model_stats = {}
            for model, count in model_stats_result:
                model_stats[f"model_{model}"] = count

            return {
                "total_entities": total_entities,
                "entities_with_embeddings": with_embeddings,
                "entities_without_embeddings": without_embeddings,
                "embedding_coverage_percent": round((with_embeddings / total_entities * 100) if total_entities > 0 else 0, 2),
                **model_stats,
            }
        except Exception as e:
            logger.error(f"Failed to get embedding statistics: {e}")
            raise RepositoryError(f"Failed to get embedding statistics: {e}", "get_embedding_statistics")
