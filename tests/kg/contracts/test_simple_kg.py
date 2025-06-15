"""Simple direct test for KG repository to debug skipping issue."""

import pytest
from datetime import datetime

from .in_memory_kg_repository import InMemoryKGRepository
from oboyu.domain.models.knowledge_graph import Entity

@pytest.fixture
def repository():
    """Create a fresh in-memory repository for each test."""
    return InMemoryKGRepository()

@pytest.fixture
def sample_entity():
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

async def test_store_and_retrieve_entity(repository, sample_entity):
    """Test storing and retrieving an entity."""
    # Store entity
    await repository.save_entity(sample_entity)
    
    # Retrieve entity by ID
    retrieved_entity = await repository.get_entity_by_id(sample_entity.id)
    
    assert retrieved_entity is not None
    assert retrieved_entity.id == sample_entity.id
    assert retrieved_entity.name == sample_entity.name
    assert retrieved_entity.entity_type == sample_entity.entity_type
    assert retrieved_entity.confidence == sample_entity.confidence