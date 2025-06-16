"""Tests for EntityEmbeddingService.

This module tests the entity embedding service functionality including
embedding computation, storage, and retrieval.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from oboyu.application.services.entity_embedding_service import EntityEmbeddingService
from oboyu.domain.models.knowledge_graph import Entity


class TestEntityEmbeddingService:
    """Test cases for EntityEmbeddingService."""

    @pytest.fixture
    def mock_kg_repository(self):
        """Create a mock KG repository."""
        repo = AsyncMock()
        repo.get_entity_by_id = AsyncMock()
        repo.update_entity = AsyncMock()
        repo.batch_update_entities = AsyncMock()
        repo.find_entities_by_type = AsyncMock()
        repo.get_all_entities = AsyncMock()
        repo.find_entities_with_stale_embeddings = AsyncMock()
        repo.get_embedding_statistics = AsyncMock()
        return repo

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = MagicMock()
        service.generate_embeddings.return_value = [np.array([0.1, 0.2, 0.3], dtype=np.float32)]
        service.dimensions = 3
        return service

    @pytest.fixture
    def sample_entity(self):
        """Create a sample entity for testing."""
        return Entity(
            id="test-entity-1",
            name="テスト会社",
            entity_type="COMPANY",
            definition="テスト用の会社エンティティ",
            properties={"industry": "technology"},
            confidence=0.9,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @pytest.fixture
    def entity_embedding_service(self, mock_kg_repository, mock_embedding_service):
        """Create EntityEmbeddingService with mocks."""
        return EntityEmbeddingService(
            kg_repository=mock_kg_repository,
            embedding_service=mock_embedding_service,
            embedding_model="test-model",
            batch_size=2,
        )

    def test_create_entity_description(self, entity_embedding_service, sample_entity):
        """Test entity description creation."""
        description = entity_embedding_service.create_entity_description(sample_entity)
        
        assert "Entity: テスト会社" in description
        assert "Type: COMPANY" in description
        assert "Definition: テスト用の会社エンティティ" in description
        assert "industry: technology" in description

    def test_create_entity_description_minimal(self, entity_embedding_service):
        """Test entity description creation with minimal entity data."""
        entity = Entity(
            id="minimal-entity",
            name="Minimal",
            entity_type="PERSON",
        )
        
        description = entity_embedding_service.create_entity_description(entity)
        
        assert "Entity: Minimal" in description
        assert "Type: PERSON" in description
        assert description == "Entity: Minimal | Type: PERSON"

    def test_create_entity_description_with_canonical_name(self, entity_embedding_service):
        """Test entity description creation with canonical name."""
        entity = Entity(
            id="canonical-entity",
            name="Short Name",
            entity_type="COMPANY",
            canonical_name="Full Canonical Company Name",
        )
        
        description = entity_embedding_service.create_entity_description(entity)
        
        assert "Entity: Short Name" in description
        assert "Canonical Name: Full Canonical Company Name" in description

    async def test_compute_and_store_embedding_success(
        self, entity_embedding_service, mock_kg_repository, mock_embedding_service, sample_entity
    ):
        """Test successful embedding computation and storage."""
        # Setup
        mock_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_embedding_service.generate_embeddings.return_value = [mock_embedding]
        
        # Execute
        result = await entity_embedding_service.compute_and_store_embedding(sample_entity)
        
        # Verify
        assert result is True
        assert sample_entity.embedding == mock_embedding.tolist()
        assert sample_entity.embedding_model == "test-model"
        assert sample_entity.embedding_updated_at is not None
        mock_kg_repository.update_entity.assert_called_once_with(sample_entity)

    async def test_compute_and_store_embedding_failure(
        self, entity_embedding_service, mock_kg_repository, mock_embedding_service, sample_entity
    ):
        """Test embedding computation failure handling."""
        # Setup
        mock_embedding_service.generate_embeddings.return_value = []
        
        # Execute
        result = await entity_embedding_service.compute_and_store_embedding(sample_entity)
        
        # Verify
        assert result is False
        assert sample_entity.embedding is None
        mock_kg_repository.update_entity.assert_not_called()

    async def test_batch_compute_embeddings_success(
        self, entity_embedding_service, mock_kg_repository, mock_embedding_service
    ):
        """Test successful batch embedding computation."""
        # Setup
        entities = [
            Entity(id="entity-1", name="Entity 1", entity_type="PERSON"),
            Entity(id="entity-2", name="Entity 2", entity_type="COMPANY"),
        ]
        mock_embeddings = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        mock_embedding_service.generate_embeddings.return_value = mock_embeddings
        
        # Execute
        result = await entity_embedding_service.batch_compute_embeddings(entities, skip_existing=False)
        
        # Verify
        assert result == 2
        for i, entity in enumerate(entities):
            assert entity.embedding == mock_embeddings[i].tolist()
            assert entity.embedding_model == "test-model"
            assert entity.embedding_updated_at is not None
        mock_kg_repository.batch_update_entities.assert_called_once()

    async def test_batch_compute_embeddings_skip_existing(
        self, entity_embedding_service, mock_kg_repository, mock_embedding_service
    ):
        """Test batch computation skipping entities with existing embeddings."""
        # Setup
        entities = [
            Entity(id="entity-1", name="Entity 1", entity_type="PERSON"),
            Entity(
                id="entity-2",
                name="Entity 2",
                entity_type="COMPANY",
                embedding=[0.7, 0.8, 0.9],  # Already has embedding
            ),
        ]
        mock_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_embedding_service.generate_embeddings.return_value = [mock_embedding]
        
        # Execute
        result = await entity_embedding_service.batch_compute_embeddings(entities, skip_existing=True)
        
        # Verify
        assert result == 1  # Only one entity processed
        assert entities[0].embedding == mock_embedding.tolist()
        assert entities[1].embedding == [0.7, 0.8, 0.9]  # Unchanged

    async def test_batch_compute_embeddings_empty_list(
        self, entity_embedding_service, mock_kg_repository, mock_embedding_service
    ):
        """Test batch computation with empty entity list."""
        result = await entity_embedding_service.batch_compute_embeddings([])
        
        assert result == 0
        mock_embedding_service.generate_embeddings.assert_not_called()
        mock_kg_repository.batch_update_entities.assert_not_called()

    async def test_get_entity_embedding_success(
        self, entity_embedding_service, mock_kg_repository
    ):
        """Test successful entity embedding retrieval."""
        # Setup
        entity = Entity(
            id="entity-1",
            name="Entity 1",
            entity_type="PERSON",
            embedding=[0.1, 0.2, 0.3],
        )
        mock_kg_repository.get_entity_by_id.return_value = entity
        
        # Execute
        result = await entity_embedding_service.get_entity_embedding("entity-1")
        
        # Verify
        assert result is not None
        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    async def test_get_entity_embedding_not_found(
        self, entity_embedding_service, mock_kg_repository
    ):
        """Test entity embedding retrieval when entity not found."""
        mock_kg_repository.get_entity_by_id.return_value = None
        
        result = await entity_embedding_service.get_entity_embedding("nonexistent")
        
        assert result is None

    async def test_get_entity_embedding_no_embedding(
        self, entity_embedding_service, mock_kg_repository
    ):
        """Test entity embedding retrieval when entity has no embedding."""
        entity = Entity(id="entity-1", name="Entity 1", entity_type="PERSON")
        mock_kg_repository.get_entity_by_id.return_value = entity
        
        result = await entity_embedding_service.get_entity_embedding("entity-1")
        
        assert result is None

    async def test_update_stale_embeddings(
        self, entity_embedding_service, mock_kg_repository, mock_embedding_service
    ):
        """Test updating stale embeddings."""
        # Setup
        stale_entities = [
            Entity(id="stale-1", name="Stale 1", entity_type="PERSON"),
            Entity(id="stale-2", name="Stale 2", entity_type="COMPANY"),
        ]
        mock_kg_repository.find_entities_with_stale_embeddings.return_value = stale_entities
        mock_embeddings = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        mock_embedding_service.generate_embeddings.return_value = mock_embeddings
        
        # Execute
        result = await entity_embedding_service.update_stale_embeddings(max_age_days=30)
        
        # Verify
        assert result == 2
        mock_kg_repository.find_entities_with_stale_embeddings.assert_called_once()
        mock_kg_repository.batch_update_entities.assert_called_once()

    async def test_update_stale_embeddings_none_found(
        self, entity_embedding_service, mock_kg_repository
    ):
        """Test updating stale embeddings when none are found."""
        mock_kg_repository.find_entities_with_stale_embeddings.return_value = []
        
        result = await entity_embedding_service.update_stale_embeddings()
        
        assert result == 0

    async def test_rebuild_all_embeddings(
        self, entity_embedding_service, mock_kg_repository, mock_embedding_service
    ):
        """Test rebuilding all embeddings."""
        # Setup
        all_entities = [
            Entity(id="entity-1", name="Entity 1", entity_type="PERSON"),
            Entity(id="entity-2", name="Entity 2", entity_type="COMPANY"),
        ]
        mock_kg_repository.get_all_entities.return_value = all_entities
        mock_embeddings = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        mock_embedding_service.generate_embeddings.return_value = mock_embeddings
        
        # Execute
        result = await entity_embedding_service.rebuild_all_embeddings()
        
        # Verify
        assert result == 2
        mock_kg_repository.get_all_entities.assert_called_once()

    async def test_rebuild_embeddings_by_type(
        self, entity_embedding_service, mock_kg_repository, mock_embedding_service
    ):
        """Test rebuilding embeddings for specific entity type."""
        # Setup
        person_entities = [
            Entity(id="person-1", name="Person 1", entity_type="PERSON"),
        ]
        mock_kg_repository.find_entities_by_type.return_value = person_entities
        mock_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_embedding_service.generate_embeddings.return_value = [mock_embedding]
        
        # Execute
        result = await entity_embedding_service.rebuild_all_embeddings(entity_type="PERSON")
        
        # Verify
        assert result == 1
        mock_kg_repository.find_entities_by_type.assert_called_once_with("PERSON")

    async def test_get_embedding_statistics(
        self, entity_embedding_service, mock_kg_repository, mock_embedding_service
    ):
        """Test getting embedding statistics."""
        # Setup
        mock_stats = {
            "total_entities": 100,
            "entities_with_embeddings": 80,
            "entities_without_embeddings": 20,
            "embedding_coverage_percent": 80.0,
        }
        mock_kg_repository.get_embedding_statistics.return_value = mock_stats
        mock_embedding_service.dimensions = 384
        
        # Execute
        result = await entity_embedding_service.get_embedding_statistics()
        
        # Verify
        assert result["total_entities"] == 100
        assert result["entities_with_embeddings"] == 80
        assert result["current_model"] == "test-model"
        assert result["embedding_dimensions"] == 384

    async def test_get_embedding_statistics_error(
        self, entity_embedding_service, mock_kg_repository, mock_embedding_service
    ):
        """Test getting embedding statistics when repository raises error."""
        # Setup
        mock_kg_repository.get_embedding_statistics.side_effect = Exception("Database error")
        
        # Execute
        result = await entity_embedding_service.get_embedding_statistics()
        
        # Verify
        assert "error" in result
        assert result["current_model"] == "test-model"