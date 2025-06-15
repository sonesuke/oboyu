"""In-memory KGRepository implementation for fast testing.

This implementation provides a lightweight, fast KGRepository for contract testing
that doesn't require database setup or external dependencies.
"""

from typing import Dict, List, Optional

from oboyu.domain.models.knowledge_graph import Entity, ProcessingStatus, Relation
from oboyu.ports.repositories.kg_repository import KGRepository


class InMemoryKGRepository(KGRepository):
    """In-memory implementation of KGRepository for testing."""

    def __init__(self) -> None:
        """Initialize empty in-memory repository."""
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        self._processing_status: Dict[str, ProcessingStatus] = {}

    async def save_entity(self, entity: Entity) -> None:
        """Save entity in memory."""
        self._entities[entity.id] = entity

    async def save_entities(self, entities: List[Entity]) -> None:
        """Save multiple entities in memory."""
        for entity in entities:
            await self.save_entity(entity)

    async def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Retrieve entity by ID from memory."""
        return self._entities.get(entity_id)

    async def search_entities_by_name(self, name_pattern: str, limit: Optional[int] = None) -> List[Entity]:
        """Search entities by name pattern in memory."""
        results = []
        for entity in self._entities.values():
            if name_pattern.lower() in entity.name.lower():
                results.append(entity)
                if limit and len(results) >= limit:
                    break
        return results

    async def get_entities_by_type(self, entity_type: str, limit: Optional[int] = None) -> List[Entity]:
        """Get entities by type from memory."""
        results = []
        for entity in self._entities.values():
            if entity.entity_type == entity_type:
                results.append(entity)
                if limit and len(results) >= limit:
                    break
        return results

    async def get_entities_by_chunk_id(self, chunk_id: str) -> List[Entity]:
        """Get entities by chunk ID from memory."""
        return [entity for entity in self._entities.values() if entity.chunk_id == chunk_id]

    async def save_relation(self, relation: Relation) -> None:
        """Save relation in memory."""
        self._relations[relation.id] = relation

    async def save_relations(self, relations: List[Relation]) -> None:
        """Save multiple relations in memory."""
        for relation in relations:
            await self.save_relation(relation)

    async def get_relation_by_id(self, relation_id: str) -> Optional[Relation]:
        """Retrieve relation by ID from memory."""
        return self._relations.get(relation_id)

    async def get_relations_by_type(self, relation_type: str, limit: Optional[int] = None) -> List[Relation]:
        """Get relations by type from memory."""
        results = []
        for relation in self._relations.values():
            if relation.relation_type == relation_type:
                results.append(relation)
                if limit and len(results) >= limit:
                    break
        return results

    async def get_relations_by_chunk_id(self, chunk_id: str) -> List[Relation]:
        """Get relations by chunk ID from memory."""
        return [relation for relation in self._relations.values() if relation.chunk_id == chunk_id]

    async def get_entity_neighbors(self, entity_id: str, max_hops: int = 1) -> List[Entity]:
        """Get neighboring entities through relations in memory."""
        if max_hops <= 0:
            return []

        # Find all relations involving this entity
        neighbor_ids = set()
        for relation in self._relations.values():
            if relation.source_id == entity_id:
                neighbor_ids.add(relation.target_id)
            elif relation.target_id == entity_id:
                neighbor_ids.add(relation.source_id)

        # Get neighbor entities
        neighbors = []
        for neighbor_id in neighbor_ids:
            if neighbor_id in self._entities:
                neighbors.append(self._entities[neighbor_id])

        # For multi-hop, recursively get neighbors of neighbors
        if max_hops > 1:
            all_neighbors = set(neighbors)
            for neighbor in neighbors:
                deeper_neighbors = await self.get_entity_neighbors(neighbor.id, max_hops - 1)
                all_neighbors.update(deeper_neighbors)
            # Remove original entity from results
            all_neighbors.discard(self._entities.get(entity_id))
            neighbors = list(all_neighbors)

        return neighbors

    async def get_entity_count(self) -> int:
        """Get total entity count from memory."""
        return len(self._entities)

    async def get_relation_count(self) -> int:
        """Get total relation count from memory."""
        return len(self._relations)

    async def get_entity_type_stats(self) -> Dict[str, int]:
        """Get entity type statistics from memory."""
        stats: Dict[str, int] = {}
        for entity in self._entities.values():
            entity_type = entity.entity_type
            stats[entity_type] = stats.get(entity_type, 0) + 1
        return stats

    async def save_processing_status(self, status: ProcessingStatus) -> None:
        """Save processing status for a chunk."""
        self._processing_status[status.chunk_id] = status

    async def get_processing_status(self, chunk_id: str) -> Optional[ProcessingStatus]:
        """Get processing status for a chunk."""
        return self._processing_status.get(chunk_id)

    async def get_unprocessed_chunks(self, processing_version: str, limit: Optional[int] = None) -> List[str]:
        """Get chunk IDs that haven't been processed with the specified version."""
        unprocessed = []
        
        # Find all chunks that have entities or relations
        all_chunk_ids = set()
        for entity in self._entities.values():
            if entity.chunk_id:
                all_chunk_ids.add(entity.chunk_id)
        for relation in self._relations.values():
            if relation.chunk_id:
                all_chunk_ids.add(relation.chunk_id)
        
        # Filter out chunks that have been processed with this version
        for chunk_id in all_chunk_ids:
            status = self._processing_status.get(chunk_id)
            if status is None or status.processing_version != processing_version:
                unprocessed.append(chunk_id)
                if limit and len(unprocessed) >= limit:
                    break
        
        return unprocessed

    async def delete_entities_by_chunk_id(self, chunk_id: str) -> int:
        """Delete all entities associated with a chunk."""
        entities_to_delete = [
            entity_id for entity_id, entity in self._entities.items()
            if entity.chunk_id == chunk_id
        ]
        
        for entity_id in entities_to_delete:
            del self._entities[entity_id]
        
        return len(entities_to_delete)

    async def delete_relations_by_chunk_id(self, chunk_id: str) -> int:
        """Delete all relations associated with a chunk."""
        relations_to_delete = [
            relation_id for relation_id, relation in self._relations.items()
            if relation.chunk_id == chunk_id
        ]
        
        for relation_id in relations_to_delete:
            del self._relations[relation_id]
        
        return len(relations_to_delete)