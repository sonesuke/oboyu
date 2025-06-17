"""CRUD operations mixin for DuckDB KG repository.

This module contains basic CRUD operations for the DuckDB KG repository
to keep the main repository file under the line limit.
"""

import json
import logging
from typing import List

from oboyu.domain.models.knowledge_graph import Entity
from oboyu.ports.repositories.kg_repository import RepositoryError

# Base imports removed to avoid MRO conflicts

logger = logging.getLogger(__name__)


class CRUDRepositoryMixin:
    """Mixin providing basic CRUD operations."""

    async def update_entity(self, entity: Entity) -> None:
        """Update an existing entity."""
        try:
            self.connection.execute(
                """
                UPDATE kg_entities SET
                    name = ?, entity_type = ?, definition = ?, properties = ?,
                    chunk_id = ?, canonical_name = ?, merged_from = ?,
                    merge_confidence = ?, confidence = ?, embedding = ?,
                    embedding_model = ?, embedding_updated_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    entity.name,
                    entity.entity_type,
                    entity.definition,
                    json.dumps(entity.properties),
                    entity.chunk_id,
                    entity.canonical_name,
                    json.dumps(entity.merged_from),
                    entity.merge_confidence,
                    entity.confidence,
                    entity.embedding,
                    entity.embedding_model,
                    entity.embedding_updated_at.isoformat() if entity.embedding_updated_at else None,
                    entity.updated_at.isoformat(),
                    entity.id,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to update entity {entity.id}: {e}")
            raise RepositoryError(f"Failed to update entity: {e}", "update_entity")

    async def batch_update_entities(self, entities: List[Entity]) -> None:
        """Update multiple entities in batch."""
        if not entities:
            return

        try:
            data = [
                (
                    entity.name,
                    entity.entity_type,
                    entity.definition,
                    json.dumps(entity.properties),
                    entity.chunk_id,
                    entity.canonical_name,
                    json.dumps(entity.merged_from),
                    entity.merge_confidence,
                    entity.confidence,
                    entity.embedding,
                    entity.embedding_model,
                    entity.embedding_updated_at.isoformat() if entity.embedding_updated_at else None,
                    entity.updated_at.isoformat(),
                    entity.id,
                )
                for entity in entities
            ]

            self.connection.executemany(
                """
                UPDATE kg_entities SET
                    name = ?, entity_type = ?, definition = ?, properties = ?,
                    chunk_id = ?, canonical_name = ?, merged_from = ?,
                    merge_confidence = ?, confidence = ?, embedding = ?,
                    embedding_model = ?, embedding_updated_at = ?, updated_at = ?
                WHERE id = ?
                """,
                data,
            )
            logger.info(f"Updated {len(entities)} entities")
        except Exception as e:
            logger.error(f"Failed to batch update {len(entities)} entities: {e}")
            raise RepositoryError(f"Failed to batch update entities: {e}", "batch_update_entities")

    async def find_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Find all entities of a specific type."""
        try:
            results = self.connection.execute("SELECT * FROM kg_entities WHERE entity_type = ? ORDER BY confidence DESC", (entity_type,)).fetchall()

            if not results:
                return []

            columns = self._get_columns_safely(self.connection, "find_entities_by_type")
            entities = []
            for result in results:
                row_dict = dict(zip(columns, result))
                entities.append(self._entity_from_row(row_dict))

            return entities
        except Exception as e:
            logger.error(f"Failed to find entities by type {entity_type}: {e}")
            raise RepositoryError(f"Failed to find entities by type: {e}", "find_entities_by_type")

    async def get_all_entities(self) -> List[Entity]:
        """Get all entities in the knowledge graph."""
        try:
            results = self.connection.execute("SELECT * FROM kg_entities ORDER BY created_at").fetchall()

            if not results:
                return []

            columns = self._get_columns_safely(self.connection, "get_all_entities")
            entities = []
            for result in results:
                row_dict = dict(zip(columns, result))
                entities.append(self._entity_from_row(row_dict))

            return entities
        except Exception as e:
            logger.error(f"Failed to get all entities: {e}")
            raise RepositoryError(f"Failed to get all entities: {e}", "get_all_entities")
