"""Contract tests for KGRepository implementations.

This module defines the behavioral contracts that all KGRepository implementations
must satisfy, following the project's contract-based testing pattern.
"""

from datetime import datetime
from typing import List

import pytest

from oboyu.domain.models.knowledge_graph import Entity, Relation
from oboyu.ports.repositories.kg_repository import KGRepository


class KGRepositoryContract:
    """Contract defining expected behavior for KGRepository implementations."""

    @pytest.fixture
    def sample_entity(self) -> Entity:
        """Create a sample entity for testing."""
        return Entity(
            id="test-entity-1",
            name="テスト会社",
            entity_type="ORGANIZATION",
            chunk_id="test-chunk-1",
            confidence=0.9,
            definition="テスト用の会社エンティティ",
            properties={"industry": "technology", "location": "Japan"},
            created_at=datetime.fromisoformat("2024-01-15T10:30:00"),
            updated_at=datetime.fromisoformat("2024-01-15T10:30:00"),
        )

    @pytest.fixture
    def sample_relation(self, sample_entity: Entity) -> Relation:
        """Create a sample relation for testing."""
        return Relation(
            id="test-relation-1",
            source_id=sample_entity.id,
            target_id="test-entity-2",
            relation_type="EMPLOYS",
            chunk_id="test-chunk-1",
            confidence=0.8,
            properties={"start_date": "2024-01-01", "role": "engineer"},
            created_at=datetime.fromisoformat("2024-01-15T10:30:00"),
            updated_at=datetime.fromisoformat("2024-01-15T10:30:00"),
        )

    @pytest.fixture
    def sample_entities(self) -> List[Entity]:
        """Create multiple sample entities for testing."""
        return [
            Entity(
                id=f"test-entity-{i}",
                name=f"テストエンティティ{i}",
                entity_type="PERSON" if i % 2 == 0 else "ORGANIZATION",
                chunk_id=f"test-chunk-{i}",
                confidence=0.9 - (i * 0.1),
                definition=f"テスト用エンティティ{i}",
                created_at=datetime.fromisoformat("2024-01-15T10:30:00"),
                updated_at=datetime.fromisoformat("2024-01-15T10:30:00"),
            )
            for i in range(1, 6)
        ]

    @pytest.fixture
    def sample_relations(self, sample_entities: List[Entity]) -> List[Relation]:
        """Create multiple sample relations for testing."""
        relations = []
        for i in range(len(sample_entities) - 1):
            relation = Relation(
                id=f"test-relation-{i+1}",
                source_id=sample_entities[i].id,
                target_id=sample_entities[i+1].id,
                relation_type="KNOWS" if i % 2 == 0 else "WORKS_WITH",
                chunk_id=f"test-chunk-{i+1}",
                confidence=0.8 - (i * 0.1),
                created_at=datetime.fromisoformat("2024-01-15T10:30:00"),
                updated_at=datetime.fromisoformat("2024-01-15T10:30:00"),
            )
            relations.append(relation)
        return relations

    # Entity Operations Contract Tests

    async def test_store_and_retrieve_single_entity(
        self, repository: KGRepository, sample_entity: Entity
    ) -> None:
        """Contract: Store entity and retrieve it by ID."""
        # Store entity
        await repository.save_entity(sample_entity)

        # Retrieve entity by ID
        retrieved_entity = await repository.get_entity_by_id(sample_entity.id)

        assert retrieved_entity is not None
        assert retrieved_entity.id == sample_entity.id
        assert retrieved_entity.name == sample_entity.name
        assert retrieved_entity.entity_type == sample_entity.entity_type
        assert retrieved_entity.confidence == sample_entity.confidence

    async def test_store_multiple_entities(
        self, repository: KGRepository, sample_entities: List[Entity]
    ) -> None:
        """Contract: Store multiple entities in batch."""
        # Store all entities
        await repository.save_entities(sample_entities)

        # Verify all entities were stored
        for entity in sample_entities:
            retrieved = await repository.get_entity_by_id(entity.id)
            assert retrieved is not None
            assert retrieved.id == entity.id

    async def test_search_entities_by_name(
        self, repository: KGRepository, sample_entities: List[Entity]
    ) -> None:
        """Contract: Search entities by name pattern."""
        await repository.save_entities(sample_entities)

        # Search for entities with partial name match
        results = await repository.search_entities_by_name("テスト", limit=10)

        assert len(results) > 0
        for entity in results:
            assert "テスト" in entity.name

    async def test_get_entities_by_type(
        self, repository: KGRepository, sample_entities: List[Entity]
    ) -> None:
        """Contract: Filter entities by type."""
        await repository.save_entities(sample_entities)

        # Get only PERSON entities
        persons = await repository.get_entities_by_type("PERSON", limit=10)
        for entity in persons:
            assert entity.entity_type == "PERSON"

        # Get only ORGANIZATION entities
        orgs = await repository.get_entities_by_type("ORGANIZATION", limit=10)
        for entity in orgs:
            assert entity.entity_type == "ORGANIZATION"

    async def test_entity_not_found_returns_none(self, repository: KGRepository) -> None:
        """Contract: Return None for non-existent entity."""
        result = await repository.get_entity_by_id("non-existent-id")
        assert result is None

    # Relation Operations Contract Tests

    async def test_store_and_retrieve_single_relation(
        self, repository: KGRepository, sample_entity: Entity, sample_relation: Relation
    ) -> None:
        """Contract: Store relation and retrieve it by ID."""
        # Store entity first (relations depend on entities)
        await repository.save_entity(sample_entity)

        # Create target entity
        target_entity = Entity(
            id="test-entity-2",
            name="ターゲットエンティティ",
            entity_type="PERSON",
            chunk_id="test-chunk-2",
            confidence=0.8,
            created_at=datetime.fromisoformat("2024-01-15T10:30:00"),
            updated_at=datetime.fromisoformat("2024-01-15T10:30:00"),
        )
        await repository.save_entity(target_entity)

        # Store relation
        await repository.save_relation(sample_relation)

        # Retrieve relation by ID
        retrieved_relation = await repository.get_relation_by_id(sample_relation.id)

        assert retrieved_relation is not None
        assert retrieved_relation.id == sample_relation.id
        assert retrieved_relation.source_id == sample_relation.source_id
        assert retrieved_relation.target_id == sample_relation.target_id
        assert retrieved_relation.relation_type == sample_relation.relation_type

    async def test_store_multiple_relations(
        self, repository: KGRepository, sample_entities: List[Entity], sample_relations: List[Relation]
    ) -> None:
        """Contract: Store multiple relations in batch."""
        # Store entities first
        await repository.save_entities(sample_entities)

        # Store all relations
        await repository.save_relations(sample_relations)

        # Verify all relations were stored
        for relation in sample_relations:
            retrieved = await repository.get_relation_by_id(relation.id)
            assert retrieved is not None
            assert retrieved.id == relation.id

    async def test_get_relations_by_type(
        self, repository: KGRepository, sample_entities: List[Entity], sample_relations: List[Relation]
    ) -> None:
        """Contract: Filter relations by type."""
        await repository.save_entities(sample_entities)
        await repository.save_relations(sample_relations)

        # Get only KNOWS relations
        knows_relations = await repository.get_relations_by_type("KNOWS", limit=10)
        for relation in knows_relations:
            assert relation.relation_type == "KNOWS"

    async def test_get_entity_neighbors(
        self, repository: KGRepository, sample_entities: List[Entity], sample_relations: List[Relation]
    ) -> None:
        """Contract: Get neighboring entities through relations."""
        await repository.save_entities(sample_entities)
        await repository.save_relations(sample_relations)

        # Get neighbors of first entity
        entity_id = sample_entities[0].id
        neighbors = await repository.get_entity_neighbors(entity_id, max_hops=1)

        assert len(neighbors) > 0
        # Should not include the original entity
        neighbor_ids = [entity.id for entity in neighbors]
        assert entity_id not in neighbor_ids

    async def test_relation_not_found_returns_none(self, repository: KGRepository) -> None:
        """Contract: Return None for non-existent relation."""
        result = await repository.get_relation_by_id("non-existent-id")
        assert result is None

    # Statistics and Analytics Contract Tests

    async def test_get_entity_count_empty(self, repository: KGRepository) -> None:
        """Contract: Return 0 count for empty repository."""
        count = await repository.get_entity_count()
        assert count == 0

    async def test_get_entity_count_with_data(
        self, repository: KGRepository, sample_entities: List[Entity]
    ) -> None:
        """Contract: Return correct count with data."""
        await repository.save_entities(sample_entities)
        count = await repository.get_entity_count()
        assert count == len(sample_entities)

    async def test_get_relation_count_empty(self, repository: KGRepository) -> None:
        """Contract: Return 0 count for empty repository."""
        count = await repository.get_relation_count()
        assert count == 0

    async def test_get_relation_count_with_data(
        self, repository: KGRepository, sample_entities: List[Entity], sample_relations: List[Relation]
    ) -> None:
        """Contract: Return correct count with data."""
        await repository.save_entities(sample_entities)
        await repository.save_relations(sample_relations)
        count = await repository.get_relation_count()
        assert count == len(sample_relations)

    async def test_get_entity_type_stats(
        self, repository: KGRepository, sample_entities: List[Entity]
    ) -> None:
        """Contract: Return entity type statistics."""
        await repository.save_entities(sample_entities)
        
        stats = await repository.get_entity_type_stats()
        
        assert len(stats) > 0
        total_count = sum(stats.values())
        assert total_count == len(sample_entities)

    # Error Handling Contract Tests

    async def test_store_entity_with_duplicate_id_updates(
        self, repository: KGRepository, sample_entity: Entity
    ) -> None:
        """Contract: Storing entity with duplicate ID should update existing."""
        # Store original entity
        await repository.save_entity(sample_entity)

        # Modify entity and store again
        modified_entity = Entity(
            id=sample_entity.id,
            name="更新されたエンティティ",
            entity_type=sample_entity.entity_type,
            chunk_id=sample_entity.chunk_id,
            confidence=0.7,
            created_at=sample_entity.created_at,
            updated_at=datetime.fromisoformat("2024-01-16T10:30:00"),
        )
        await repository.save_entity(modified_entity)

        # Verify entity was updated
        retrieved = await repository.get_entity_by_id(sample_entity.id)
        assert retrieved is not None
        assert retrieved.name == "更新されたエンティティ"
        assert retrieved.confidence == 0.7

    async def test_concurrent_access_safety(
        self, repository: KGRepository, sample_entities: List[Entity]
    ) -> None:
        """Contract: Repository handles concurrent access safely."""
        import asyncio

        # Store entities concurrently
        tasks = [repository.save_entity(entity) for entity in sample_entities]
        await asyncio.gather(*tasks)

        # Verify all entities were stored correctly
        for entity in sample_entities:
            retrieved = await repository.get_entity_by_id(entity.id)
            assert retrieved is not None
            assert retrieved.id == entity.id