"""Chunk retrieval utilities for GraphRAG.

This module provides utilities for retrieving and processing chunks
based on knowledge graph entities and relations.
"""

import json
import logging
from typing import Any, Dict, List

from oboyu.domain.models.knowledge_graph import Entity, Relation
from oboyu.ports.repositories.kg_repository import KGRepository
from oboyu.ports.services.graphrag_service import GraphRAGError

logger = logging.getLogger(__name__)


class ChunkRetrievalHelper:
    """Helper class for chunk retrieval operations."""

    def __init__(
        self,
        kg_repository: KGRepository,
        database_connection: Any,  # noqa: ANN401
    ) -> None:
        """Initialize chunk retrieval helper.

        Args:
            kg_repository: Knowledge graph repository
            database_connection: Database connection for chunk queries

        """
        self.kg_repository = kg_repository
        self.db_connection = database_connection

    async def get_contextual_chunks(
        self,
        entities: List[Entity],
        relations: List[Relation],
        max_chunks: int = 20,
        include_related: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get relevant chunks based on entities and relations."""
        try:
            if not entities:
                return []

            logger.info(f"Getting contextual chunks for {len(entities)} entities")

            # Collect chunk IDs from entities and relations
            chunk_ids = set()

            # Add chunks from entities
            for entity in entities:
                if entity.chunk_id:
                    chunk_ids.add(entity.chunk_id)

            # Add chunks from relations
            for relation in relations:
                if relation.chunk_id:
                    chunk_ids.add(relation.chunk_id)

            # If including related entities, expand chunk set
            if include_related:
                for entity in entities[:5]:  # Limit to prevent too many queries
                    try:
                        related_entities = await self.kg_repository.get_entity_neighbors(entity.id, max_hops=1)
                        for related_entity in related_entities[:3]:  # Limit related entities
                            if related_entity.chunk_id:
                                chunk_ids.add(related_entity.chunk_id)
                    except Exception as e:
                        logger.warning(f"Failed to get related entities for {entity.name}: {e}")

            # Retrieve chunk content from database
            chunks = []
            if chunk_ids:
                placeholders = ",".join(["?" for _ in chunk_ids])
                query = f"""
                    SELECT id, content, metadata, created_at
                    FROM chunks
                    WHERE id IN ({placeholders})
                    ORDER BY created_at DESC
                    LIMIT ?
                """

                params = list(chunk_ids) + [max_chunks]
                results = self.db_connection.execute(query, params).fetchall()

                # Convert to chunk dictionaries
                for row in results:
                    chunk_data = {
                        "chunk_id": row[0],
                        "content": row[1],
                        "metadata": json.loads(row[2]) if row[2] else {},
                        "created_at": row[3],
                        "related_entities": [],
                        "related_relations": [],
                    }

                    # Add entity and relation context
                    for entity in entities:
                        if entity.chunk_id == row[0]:
                            chunk_data["related_entities"].append(
                                {
                                    "id": entity.id,
                                    "name": entity.name,
                                    "type": entity.entity_type,
                                    "confidence": entity.confidence,
                                }
                            )

                    for relation in relations:
                        if relation.chunk_id == row[0]:
                            chunk_data["related_relations"].append(
                                {
                                    "id": relation.id,
                                    "type": relation.relation_type,
                                    "source_id": relation.source_id,
                                    "target_id": relation.target_id,
                                    "confidence": relation.confidence,
                                }
                            )

                    chunks.append(chunk_data)

            logger.info(f"Retrieved {len(chunks)} contextual chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to get contextual chunks: {e}")
            raise GraphRAGError(f"Contextual chunk retrieval failed: {e}", "get_contextual_chunks")
