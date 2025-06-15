"""DuckDB implementation of Knowledge Graph repository.

This module provides a concrete implementation of the KG repository interface
using DuckDB for persistence and querying of knowledge graph data.
"""

import json
import logging
from typing import Any, List, Optional

from duckdb import DuckDBPyConnection

from oboyu.domain.models.knowledge_graph import Entity, ProcessingStatus, Relation
from oboyu.ports.repositories.kg_repository import KGRepository, RepositoryError

from .base import DuckDBKGRepositoryBase

logger = logging.getLogger(__name__)


class DuckDBKGRepository(DuckDBKGRepositoryBase, KGRepository):
    """DuckDB-based implementation of knowledge graph repository."""

    def __init__(self, connection: DuckDBPyConnection) -> None:
        """Initialize repository with DuckDB connection.

        Args:
            connection: DuckDB database connection

        """
        self.connection = connection

    async def save_entity(self, entity: Entity) -> None:
        """Save an entity to the knowledge graph."""
        try:
            self.connection.execute(
                """
                INSERT INTO kg_entities
                (id, name, entity_type, definition, properties, chunk_id, canonical_name,
                 merged_from, merge_confidence, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    entity_type = EXCLUDED.entity_type,
                    definition = EXCLUDED.definition,
                    properties = EXCLUDED.properties,
                    chunk_id = EXCLUDED.chunk_id,
                    canonical_name = EXCLUDED.canonical_name,
                    merged_from = EXCLUDED.merged_from,
                    merge_confidence = EXCLUDED.merge_confidence,
                    confidence = EXCLUDED.confidence,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    entity.id,
                    entity.name,
                    entity.entity_type,
                    entity.definition,
                    json.dumps(entity.properties),
                    entity.chunk_id,
                    entity.canonical_name,
                    json.dumps(entity.merged_from),
                    entity.merge_confidence,
                    entity.confidence,
                    entity.created_at.isoformat(),
                    entity.updated_at.isoformat(),
                ),
            )
        except Exception as e:
            logger.error(f"Failed to save entity {entity.id}: {e}")
            raise RepositoryError(f"Failed to save entity: {e}", "save_entity")

    async def save_entities(self, entities: List[Entity]) -> None:
        """Save multiple entities to the knowledge graph."""
        if not entities:
            return

        try:
            data = [
                (
                    entity.id,
                    entity.name,
                    entity.entity_type,
                    entity.definition,
                    json.dumps(entity.properties),
                    entity.chunk_id,
                    entity.canonical_name,
                    json.dumps(entity.merged_from),
                    entity.merge_confidence,
                    entity.confidence,
                    entity.created_at.isoformat(),
                    entity.updated_at.isoformat(),
                )
                for entity in entities
            ]

            self.connection.executemany(
                """
                INSERT INTO kg_entities
                (id, name, entity_type, definition, properties, chunk_id, canonical_name,
                 merged_from, merge_confidence, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    entity_type = EXCLUDED.entity_type,
                    definition = EXCLUDED.definition,
                    properties = EXCLUDED.properties,
                    chunk_id = EXCLUDED.chunk_id,
                    canonical_name = EXCLUDED.canonical_name,
                    merged_from = EXCLUDED.merged_from,
                    merge_confidence = EXCLUDED.merge_confidence,
                    confidence = EXCLUDED.confidence,
                    updated_at = EXCLUDED.updated_at
                """,
                data,
            )
            logger.info(f"Saved {len(entities)} entities")
        except Exception as e:
            logger.error(f"Failed to save {len(entities)} entities: {e}")
            raise RepositoryError(f"Failed to save entities: {e}", "save_entities")

    async def save_relation(self, relation: Relation) -> None:
        """Save a relation to the knowledge graph."""
        try:
            self.connection.execute(
                """
                INSERT INTO kg_relations
                (id, source_id, target_id, relation_type, properties, chunk_id,
                 confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    source_id = EXCLUDED.source_id,
                    target_id = EXCLUDED.target_id,
                    relation_type = EXCLUDED.relation_type,
                    properties = EXCLUDED.properties,
                    chunk_id = EXCLUDED.chunk_id,
                    confidence = EXCLUDED.confidence,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    relation.id,
                    relation.source_id,
                    relation.target_id,
                    relation.relation_type,
                    json.dumps(relation.properties),
                    relation.chunk_id,
                    relation.confidence,
                    relation.created_at.isoformat(),
                    relation.updated_at.isoformat(),
                ),
            )
        except Exception as e:
            logger.error(f"Failed to save relation {relation.id}: {e}")
            raise RepositoryError(f"Failed to save relation: {e}", "save_relation")

    async def save_relations(self, relations: List[Relation]) -> None:
        """Save multiple relations to the knowledge graph."""
        if not relations:
            return

        try:
            data = [
                (
                    relation.id,
                    relation.source_id,
                    relation.target_id,
                    relation.relation_type,
                    json.dumps(relation.properties),
                    relation.chunk_id,
                    relation.confidence,
                    relation.created_at.isoformat(),
                    relation.updated_at.isoformat(),
                )
                for relation in relations
            ]

            self.connection.executemany(
                """
                INSERT INTO kg_relations
                (id, source_id, target_id, relation_type, properties, chunk_id,
                 confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    source_id = EXCLUDED.source_id,
                    target_id = EXCLUDED.target_id,
                    relation_type = EXCLUDED.relation_type,
                    properties = EXCLUDED.properties,
                    chunk_id = EXCLUDED.chunk_id,
                    confidence = EXCLUDED.confidence,
                    updated_at = EXCLUDED.updated_at
                """,
                data,
            )
            logger.info(f"Saved {len(relations)} relations")
        except Exception as e:
            logger.error(f"Failed to save {len(relations)} relations: {e}")
            raise RepositoryError(f"Failed to save relations: {e}", "save_relations")

    async def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by its ID."""
        try:
            result = self.connection.execute("SELECT * FROM kg_entities WHERE id = ?", (entity_id,)).fetchone()

            if result:
                # Convert result to dictionary
                columns = self._get_columns_safely(self.connection, "get_entity_by_id")
                row_dict = dict(zip(columns, result))
                return self._entity_from_row(row_dict)
            return None
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
            raise RepositoryError(f"Failed to get entity: {e}", "get_entity_by_id")

    async def get_entities_by_chunk_id(self, chunk_id: str) -> List[Entity]:
        """Retrieve all entities extracted from a specific chunk."""
        try:
            results = self.connection.execute("SELECT * FROM kg_entities WHERE chunk_id = ? ORDER BY created_at", (chunk_id,)).fetchall()

            if not results:
                return []

            columns = self._get_columns_safely(self.connection, "get_entities_by_chunk_id")
            entities = []
            for result in results:
                row_dict = dict(zip(columns, result))
                entities.append(self._entity_from_row(row_dict))

            return entities
        except Exception as e:
            logger.error(f"Failed to get entities for chunk {chunk_id}: {e}")
            raise RepositoryError(f"Failed to get entities by chunk: {e}", "get_entities_by_chunk_id")

    async def get_relations_by_chunk_id(self, chunk_id: str) -> List[Relation]:
        """Retrieve all relations extracted from a specific chunk."""
        try:
            results = self.connection.execute("SELECT * FROM kg_relations WHERE chunk_id = ? ORDER BY created_at", (chunk_id,)).fetchall()

            if not results:
                return []

            columns = self._get_columns_safely(self.connection, "get_relations_by_chunk_id")
            relations = []
            for result in results:
                row_dict = dict(zip(columns, result))
                relations.append(self._relation_from_row(row_dict))

            return relations
        except Exception as e:
            logger.error(f"Failed to get relations for chunk {chunk_id}: {e}")
            raise RepositoryError(f"Failed to get relations by chunk: {e}", "get_relations_by_chunk_id")

    async def get_entities_by_type(self, entity_type: str, limit: Optional[int] = None) -> List[Entity]:
        """Retrieve entities by type."""
        try:
            query = "SELECT * FROM kg_entities WHERE entity_type = ? ORDER BY confidence DESC"
            params: List[Any] = [entity_type]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            results = self.connection.execute(query, params).fetchall()

            if not results:
                return []

            columns = self._get_columns_safely(self.connection, "get_entities_by_type")
            entities = []
            for result in results:
                row_dict = dict(zip(columns, result))
                entities.append(self._entity_from_row(row_dict))

            return entities
        except Exception as e:
            logger.error(f"Failed to get entities by type {entity_type}: {e}")
            raise RepositoryError(f"Failed to get entities by type: {e}", "get_entities_by_type")

    async def get_relations_by_type(self, relation_type: str, limit: Optional[int] = None) -> List[Relation]:
        """Retrieve relations by type."""
        try:
            query = "SELECT * FROM kg_relations WHERE relation_type = ? ORDER BY confidence DESC"
            params: List[Any] = [relation_type]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            results = self.connection.execute(query, params).fetchall()

            if not results:
                return []

            columns = self._get_columns_safely(self.connection, "get_relations_by_type")
            relations = []
            for result in results:
                row_dict = dict(zip(columns, result))
                relations.append(self._relation_from_row(row_dict))

            return relations
        except Exception as e:
            logger.error(f"Failed to get relations by type {relation_type}: {e}")
            raise RepositoryError(f"Failed to get relations by type: {e}", "get_relations_by_type")

    async def get_entity_neighbors(self, entity_id: str, max_hops: int = 1) -> List[Entity]:
        """Get neighboring entities connected through relations."""
        try:
            if max_hops == 1:
                # Simple one-hop query
                results = self.connection.execute(
                    """
                    SELECT DISTINCT e.* FROM kg_entities e
                    JOIN kg_relations r ON (e.id = r.target_id AND r.source_id = ?)
                                        OR (e.id = r.source_id AND r.target_id = ?)
                    ORDER BY e.confidence DESC
                    """,
                    (entity_id, entity_id),
                ).fetchall()
            else:
                # Multi-hop query would require recursive CTE or graph traversal
                # For now, implement single hop
                results = self.connection.execute(
                    """
                    SELECT DISTINCT e.* FROM kg_entities e
                    JOIN kg_relations r ON (e.id = r.target_id AND r.source_id = ?)
                                        OR (e.id = r.source_id AND r.target_id = ?)
                    ORDER BY e.confidence DESC
                    """,
                    (entity_id, entity_id),
                ).fetchall()

            if not results:
                return []

            columns = self._get_columns_safely(self.connection, "get_entity_neighbors")
            entities = []
            for result in results:
                row_dict = dict(zip(columns, result))
                entities.append(self._entity_from_row(row_dict))

            return entities
        except Exception as e:
            logger.error(f"Failed to get neighbors for entity {entity_id}: {e}")
            raise RepositoryError(f"Failed to get entity neighbors: {e}", "get_entity_neighbors")

    async def search_entities_by_name(self, name_pattern: str, limit: Optional[int] = None) -> List[Entity]:
        """Search entities by name pattern."""
        try:
            query = "SELECT * FROM kg_entities WHERE name ILIKE ? ORDER BY confidence DESC"
            params: List[Any] = [f"%{name_pattern}%"]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            results = self.connection.execute(query, params).fetchall()

            if not results:
                return []

            columns = self._get_columns_safely(self.connection, "search_entities_by_name")
            entities = []
            for result in results:
                row_dict = dict(zip(columns, result))
                entities.append(self._entity_from_row(row_dict))

            return entities
        except Exception as e:
            logger.error(f"Failed to search entities by name {name_pattern}: {e}")
            raise RepositoryError(f"Failed to search entities by name: {e}", "search_entities_by_name")

    async def save_processing_status(self, status: ProcessingStatus) -> None:
        """Save processing status for a chunk."""
        try:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO kg_processing_status
                (chunk_id, processed_at, processing_version, entity_count, relation_count,
                 processing_time_ms, model_used, error_message, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    status.chunk_id,
                    status.processed_at.isoformat(),
                    status.processing_version,
                    status.entity_count,
                    status.relation_count,
                    status.processing_time_ms,
                    status.model_used,
                    status.error_message,
                    status.status,
                    status.created_at.isoformat(),
                    status.updated_at.isoformat(),
                ),
            )
        except Exception as e:
            logger.error(f"Failed to save processing status for chunk {status.chunk_id}: {e}")
            raise RepositoryError(f"Failed to save processing status: {e}", "save_processing_status")

    async def get_processing_status(self, chunk_id: str) -> Optional[ProcessingStatus]:
        """Get processing status for a chunk."""
        try:
            result = self.connection.execute("SELECT * FROM kg_processing_status WHERE chunk_id = ?", (chunk_id,)).fetchone()

            if result:
                columns = self._get_columns_safely(self.connection, "get_processing_status")
                row_dict = dict(zip(columns, result))
                return self._processing_status_from_row(row_dict)
            return None
        except Exception as e:
            logger.error(f"Failed to get processing status for chunk {chunk_id}: {e}")
            raise RepositoryError(f"Failed to get processing status: {e}", "get_processing_status")

    async def get_unprocessed_chunks(self, processing_version: str, limit: Optional[int] = None) -> List[str]:
        """Get chunk IDs that haven't been processed with the specified version."""
        try:
            query = """
                SELECT c.id FROM chunks c
                LEFT JOIN kg_processing_status kps ON c.id = kps.chunk_id
                    AND kps.processing_version = ?
                    AND kps.status = 'completed'
                WHERE kps.chunk_id IS NULL
                ORDER BY c.created_at
            """
            params: List[Any] = [processing_version]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            results = self.connection.execute(query, params).fetchall()
            return [str(result[0]) for result in results]
        except Exception as e:
            logger.error(f"Failed to get unprocessed chunks for version {processing_version}: {e}")
            raise RepositoryError(f"Failed to get unprocessed chunks: {e}", "get_unprocessed_chunks")

    async def delete_entities_by_chunk_id(self, chunk_id: str) -> int:
        """Delete all entities associated with a chunk."""
        try:
            result = self.connection.execute("DELETE FROM kg_entities WHERE chunk_id = ?", (chunk_id,))
            deleted_count = result.rowcount if hasattr(result, "rowcount") else 0
            logger.info(f"Deleted {deleted_count} entities for chunk {chunk_id}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete entities for chunk {chunk_id}: {e}")
            raise RepositoryError(f"Failed to delete entities: {e}", "delete_entities_by_chunk_id")

    async def delete_relations_by_chunk_id(self, chunk_id: str) -> int:
        """Delete all relations associated with a chunk."""
        try:
            result = self.connection.execute("DELETE FROM kg_relations WHERE chunk_id = ?", (chunk_id,))
            deleted_count = result.rowcount if hasattr(result, "rowcount") else 0
            logger.info(f"Deleted {deleted_count} relations for chunk {chunk_id}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete relations for chunk {chunk_id}: {e}")
            raise RepositoryError(f"Failed to delete relations: {e}", "delete_relations_by_chunk_id")

    async def get_entity_count(self) -> int:
        """Get total number of entities in the knowledge graph."""
        try:
            result = self.connection.execute("SELECT COUNT(*) FROM kg_entities").fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get entity count: {e}")
            raise RepositoryError(f"Failed to get entity count: {e}", "get_entity_count")

    async def get_relation_count(self) -> int:
        """Get total number of relations in the knowledge graph."""
        try:
            result = self.connection.execute("SELECT COUNT(*) FROM kg_relations").fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get relation count: {e}")
            raise RepositoryError(f"Failed to get relation count: {e}", "get_relation_count")
