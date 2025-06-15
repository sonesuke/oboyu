"""Tests for DuckDBKGRepository implementation-specific functionality.

This module tests DuckDB-specific features and implementation details that are not
covered by the general contract tests.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from oboyu.adapters.kg_repositories import DuckDBKGRepository
from oboyu.domain.models.knowledge_graph import Entity, Relation


@pytest.fixture
def mock_connection():
    """Create a mock DuckDB connection."""
    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = []
    connection.execute.return_value.fetchone.return_value = None
    return connection


@pytest.fixture
def repository(mock_connection):
    """Create DuckDBKGRepository with mock connection."""
    return DuckDBKGRepository(mock_connection)


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


@pytest.fixture
def sample_relation():
    """Create a sample relation for testing."""
    return Relation(
        id="test-relation-1",
        source_id="test-entity-1",
        target_id="test-entity-2",
        relation_type="EMPLOYS",
        chunk_id="test-chunk-1",
        confidence=0.8,
        properties={"start_date": "2024-01-01", "role": "engineer"},
        created_at=datetime.fromisoformat("2024-01-15T10:30:00"),
        updated_at=datetime.fromisoformat("2024-01-15T10:30:00"),
    )


class TestDuckDBKGRepositoryImplementation:
    """Test DuckDB-specific implementation details."""

    async def test_save_entity_sql_query(self, repository, mock_connection, sample_entity):
        """Test that save_entity generates correct SQL."""
        await repository.save_entity(sample_entity)

        # Verify SQL was executed
        assert mock_connection.execute.called
        
        # Get the executed query
        query_args = mock_connection.execute.call_args[0]
        query = query_args[0]
        params = query_args[1] if len(query_args) > 1 else []

        # Verify it's an INSERT or REPLACE query for entities
        assert "INSERT" in query.upper() or "REPLACE" in query.upper()
        assert "entities" in query
        
        # Verify entity data is in parameters
        assert sample_entity.id in params
        assert sample_entity.name in params

    async def test_save_relation_sql_query(self, repository, mock_connection, sample_relation):
        """Test that save_relation generates correct SQL."""
        await repository.save_relation(sample_relation)

        # Verify SQL was executed
        assert mock_connection.execute.called
        
        # Get the executed query
        query_args = mock_connection.execute.call_args[0]
        query = query_args[0]
        params = query_args[1] if len(query_args) > 1 else []

        # Verify it's an INSERT or REPLACE query for relations
        assert "INSERT" in query.upper() or "REPLACE" in query.upper()
        assert "relations" in query
        
        # Verify relation data is in parameters
        assert sample_relation.id in params
        assert sample_relation.source_id in params
        assert sample_relation.target_id in params

    async def test_get_entity_by_id_sql_query(self, repository, mock_connection):
        """Test that get_entity_by_id generates correct SQL."""
        # Configure mock to return no results
        mock_connection.execute.return_value.fetchone.return_value = None

        await repository.get_entity_by_id("test-id")

        # Verify SQL was executed
        assert mock_connection.execute.called
        
        # Get the executed query
        query_args = mock_connection.execute.call_args[0]
        query = query_args[0]
        params = query_args[1] if len(query_args) > 1 else []

        # Verify it's a SELECT query for entities
        assert "SELECT" in query.upper()
        assert "entities" in query
        assert "WHERE" in query.upper()
        assert "test-id" in params

    async def test_search_entities_by_name_uses_like_pattern(self, repository, mock_connection):
        """Test that search_entities_by_name uses SQL LIKE pattern."""
        # Configure mock to return no results
        mock_connection.execute.return_value.fetchall.return_value = []

        await repository.search_entities_by_name("テスト", limit=10)

        # Verify SQL was executed
        assert mock_connection.execute.called
        
        # Get the executed query
        query_args = mock_connection.execute.call_args[0]
        query = query_args[0]
        params = query_args[1] if len(query_args) > 1 else []

        # Verify it uses LIKE pattern
        assert "LIKE" in query.upper() or "ILIKE" in query.upper()
        assert any("%" in str(param) for param in params), "Should use % wildcard pattern"

    async def test_get_entity_neighbors_with_joins(self, repository, mock_connection):
        """Test that get_entity_neighbors uses proper JOIN queries."""
        # Configure mock to return no results
        mock_connection.execute.return_value.fetchall.return_value = []

        await repository.get_entity_neighbors("test-entity-1", max_hops=1)

        # Verify SQL was executed
        assert mock_connection.execute.called
        
        # Get the executed query
        query_args = mock_connection.execute.call_args[0]
        query = query_args[0]

        # Verify it uses JOIN to connect entities and relations
        assert "JOIN" in query.upper()
        assert "entities" in query
        assert "relations" in query

    async def test_get_entity_count_returns_integer(self, repository, mock_connection):
        """Test that get_entity_count returns integer count."""
        # Mock count result
        mock_connection.execute.return_value.fetchone.return_value = (42,)

        count = await repository.get_entity_count()

        # Verify it returns the count as integer
        assert count == 42
        assert isinstance(count, int)