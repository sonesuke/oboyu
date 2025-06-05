"""Simple validation tests for contract-based testing components."""

from datetime import datetime
from pathlib import Path

import pytest

from src.oboyu.domain.entities.chunk import Chunk
from src.oboyu.domain.entities.query import Query
from src.oboyu.domain.value_objects.chunk_id import ChunkId
from src.oboyu.domain.value_objects.embedding_vector import EmbeddingVector
from src.oboyu.domain.value_objects.language_code import LanguageCode
from src.oboyu.domain.value_objects.search_mode import SearchMode

from .in_memory_search_repository import InMemorySearchRepository


@pytest.fixture
async def repository():
    """Provide an in-memory repository for testing."""
    repo = InMemorySearchRepository()
    yield repo
    repo.clear()


@pytest.fixture
def sample_chunk():
    """Create a test chunk."""
    now = datetime.now()
    return Chunk(
        id=ChunkId.create(Path("/test/document.txt"), 0),
        document_path=Path("/test/document.txt"),
        title="Test Document",
        content="This is test content for searching and indexing.",
        chunk_index=0,
        language=LanguageCode.ENGLISH,
        created_at=now,
        modified_at=now,
        metadata={"test": True},
        start_char=0,
        end_char=45
    )


async def test_basic_chunk_storage_and_retrieval(repository, sample_chunk):
    """Test basic chunk storage and retrieval."""
    # Store chunk
    await repository.store_chunk(sample_chunk)
    
    # Check it exists
    assert await repository.chunk_exists(sample_chunk.id)
    
    # Retrieve it
    retrieved = await repository.find_by_chunk_id(sample_chunk.id)
    assert retrieved is not None
    assert retrieved.id == sample_chunk.id
    assert retrieved.content == sample_chunk.content


async def test_basic_embedding_storage_and_search(repository, sample_chunk):
    """Test basic embedding storage and vector search."""
    # Store chunk
    await repository.store_chunk(sample_chunk)
    
    # Store embedding
    embedding = EmbeddingVector.create([0.1, 0.2, 0.3, 0.4, 0.5])
    await repository.store_embedding(sample_chunk.id, embedding)
    
    # Search by vector similarity
    results = await repository.find_by_vector_similarity(
        query_vector=embedding,
        top_k=5,
        threshold=0.0
    )
    
    # Should find our chunk
    assert len(results) > 0
    assert results[0].chunk_id == sample_chunk.id


async def test_basic_bm25_search(repository, sample_chunk):
    """Test basic BM25 search."""
    # Store chunk
    await repository.store_chunk(sample_chunk)
    
    # Create query
    query = Query(
        text="test content",
        mode=SearchMode.BM25,
        top_k=10
    )
    
    # Search
    results = await repository.find_by_bm25(query)
    
    # Should find our chunk
    assert len(results) > 0
    assert results[0].chunk_id == sample_chunk.id


async def test_chunk_deletion(repository, sample_chunk):
    """Test chunk deletion."""
    # Store chunk
    await repository.store_chunk(sample_chunk)
    assert await repository.chunk_exists(sample_chunk.id)
    
    # Delete chunk
    await repository.delete_chunk(sample_chunk.id)
    assert not await repository.chunk_exists(sample_chunk.id)
    
    # Should not be retrievable
    retrieved = await repository.find_by_chunk_id(sample_chunk.id)
    assert retrieved is None


async def test_count_operations(repository, sample_chunk):
    """Test count operations."""
    # Initially empty
    assert await repository.get_chunk_count() == 0
    assert await repository.get_embedding_count() == 0
    
    # Store chunk
    await repository.store_chunk(sample_chunk)
    assert await repository.get_chunk_count() == 1
    assert await repository.get_embedding_count() == 0
    
    # Store embedding
    embedding = EmbeddingVector.create([0.1, 0.2, 0.3])
    await repository.store_embedding(sample_chunk.id, embedding)
    assert await repository.get_chunk_count() == 1
    assert await repository.get_embedding_count() == 1


def test_repository_instantiation():
    """Test that we can create repository instances."""
    repo = InMemorySearchRepository()
    assert repo is not None
    
    # Test utility methods
    assert len(repo.get_indexed_terms()) == 0
    repo.clear()  # Should not raise
