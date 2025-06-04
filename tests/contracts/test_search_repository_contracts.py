"""Test runner for SearchRepository contract tests."""

from pathlib import Path

import pytest

from src.oboyu.adapters.database.duckdb_search_repository import DuckDBSearchRepository

from .in_memory_search_repository import InMemorySearchRepository
from .search_repository_contract import SearchRepositoryContract


class TestSearchRepositoryContracts(SearchRepositoryContract):
    """Test runner that applies all SearchRepository contracts to different implementations."""
    
    @pytest.fixture(params=["in_memory"])
    async def repository(self, request, temp_db_path):
        """Fixture that provides different SearchRepository implementations."""
        if request.param == "in_memory":
            # In-memory implementation for fast testing
            repo = InMemorySearchRepository()
            yield repo
            repo.clear()  # Clean up after test
            
        elif request.param == "duckdb":
            # DuckDB implementation for integration testing
            # Note: This requires a real database service
            try:
                from src.oboyu.indexer.storage.database_manager import DatabaseManager
                from src.oboyu.indexer.storage.database_service import DatabaseService
                
                # Create a temporary database
                db_manager = DatabaseManager(str(temp_db_path))
                await db_manager.initialize()
                
                db_service = DatabaseService(db_manager)
                repo = DuckDBSearchRepository(db_service)
                
                yield repo
                
                # Clean up
                await db_manager.close()
            except ImportError:
                pytest.skip("DuckDB dependencies not available")
    
    # Test method overrides to use the parametrized repository fixture
    # These methods inherit the test logic from SearchRepositoryContract
    # but use the parametrized repository fixture
    
    async def test_store_and_retrieve_single_chunk(self, repository, sample_chunk):
        """Test storage and retrieval for all implementations."""
        await super().test_store_and_retrieve_single_chunk(repository, sample_chunk)
    
    async def test_store_and_retrieve_multiple_chunks(self, repository, sample_chunks):
        """Test multiple chunk storage for all implementations."""
        await super().test_store_and_retrieve_multiple_chunks(repository, sample_chunks)
    
    async def test_chunk_exists_behavior(self, repository, sample_chunk):
        """Test chunk existence checking for all implementations."""
        await super().test_chunk_exists_behavior(repository, sample_chunk)
    
    async def test_store_and_search_by_embedding(self, repository, sample_chunk, sample_embedding):
        """Test embedding storage and vector search for all implementations."""
        await super().test_store_and_search_by_embedding(repository, sample_chunk, sample_embedding)
    
    async def test_store_multiple_embeddings(self, repository, sample_chunks):
        """Test multiple embedding storage for all implementations."""
        await super().test_store_multiple_embeddings(repository, sample_chunks)
    
    async def test_bm25_search_basic(self, repository, sample_chunks, sample_query):
        """Test BM25 search for all implementations."""
        await super().test_bm25_search_basic(repository, sample_chunks, sample_query)
    
    async def test_empty_query_behavior(self, repository):
        """Test empty query handling for all implementations."""
        await super().test_empty_query_behavior(repository)
    
    async def test_delete_single_chunk(self, repository, sample_chunk):
        """Test single chunk deletion for all implementations."""
        await super().test_delete_single_chunk(repository, sample_chunk)
    
    async def test_delete_chunks_by_document(self, repository, sample_chunks):
        """Test document-based chunk deletion for all implementations."""
        await super().test_delete_chunks_by_document(repository, sample_chunks)
    
    async def test_chunk_count_accuracy(self, repository, sample_chunks):
        """Test chunk counting for all implementations."""
        await super().test_chunk_count_accuracy(repository, sample_chunks)
    
    async def test_embedding_count_accuracy(self, repository, sample_chunks):
        """Test embedding counting for all implementations."""
        await super().test_embedding_count_accuracy(repository, sample_chunks)
    
    async def test_nonexistent_chunk_retrieval(self, repository):
        """Test nonexistent chunk retrieval for all implementations."""
        await super().test_nonexistent_chunk_retrieval(repository)
    
    async def test_duplicate_chunk_storage(self, repository, sample_chunk):
        """Test duplicate chunk handling for all implementations."""
        await super().test_duplicate_chunk_storage(repository, sample_chunk)
    
    async def test_delete_nonexistent_chunk(self, repository):
        """Test nonexistent chunk deletion for all implementations."""
        await super().test_delete_nonexistent_chunk(repository)
    
    async def test_concurrent_access_safety(self, repository, sample_chunks):
        """Test concurrent access for all implementations."""
        await super().test_concurrent_access_safety(repository, sample_chunks)
    
    async def test_search_result_ordering(self, repository, sample_chunks):
        """Test search result ordering for all implementations."""
        await super().test_search_result_ordering(repository, sample_chunks)
    
    async def test_top_k_limit_respected(self, repository, sample_chunks):
        """Test top_k limit enforcement for all implementations."""
        await super().test_top_k_limit_respected(repository, sample_chunks)
    
    async def test_similarity_threshold_respected(self, repository, sample_chunk, sample_embedding):
        """Test similarity threshold enforcement for all implementations."""
        await super().test_similarity_threshold_respected(repository, sample_chunk, sample_embedding)


# Separate test classes for specific implementations
class TestInMemorySearchRepository(SearchRepositoryContract):
    """Test class specifically for InMemorySearchRepository."""
    
    @pytest.fixture
    async def repository(self):
        """Provide InMemorySearchRepository for testing."""
        repo = InMemorySearchRepository()
        yield repo
        repo.clear()
    
    # Additional tests specific to in-memory implementation
    async def test_memory_cleanup(self, repository):
        """Test that clear() properly resets all internal state."""
        # Store some data
        from datetime import datetime

        from src.oboyu.domain.entities.chunk import Chunk
        from src.oboyu.domain.value_objects.chunk_id import ChunkId
        from src.oboyu.domain.value_objects.embedding_vector import EmbeddingVector
        from src.oboyu.domain.value_objects.language_code import LanguageCode
        
        now = datetime.now()
        chunk = Chunk(
            id=ChunkId.create(Path("/test/doc.txt"), 0),
            document_path=Path("/test/doc.txt"),
            title="Test",
            content="Test content",
            chunk_index=0,
            language=LanguageCode.ENGLISH,
            created_at=now,
            modified_at=now,
            metadata={},
            start_char=0,
            end_char=12
        )
        
        await repository.store_chunk(chunk)
        await repository.store_embedding(chunk.id, EmbeddingVector.create([0.1, 0.2, 0.3]))
        
        # Verify data exists
        assert await repository.get_chunk_count() > 0
        assert await repository.get_embedding_count() > 0
        assert len(repository.get_indexed_terms()) > 0
        
        # Clear and verify everything is reset
        repository.clear()
        assert await repository.get_chunk_count() == 0
        assert await repository.get_embedding_count() == 0
        assert len(repository.get_indexed_terms()) == 0
    
    async def test_bm25_term_indexing(self, repository):
        """Test that BM25 indexing works correctly."""
        from datetime import datetime

        from src.oboyu.domain.entities.chunk import Chunk
        from src.oboyu.domain.value_objects.chunk_id import ChunkId
        from src.oboyu.domain.value_objects.language_code import LanguageCode
        
        now = datetime.now()
        chunk = Chunk(
            id=ChunkId.create(Path("/test/doc.txt"), 0),
            document_path=Path("/test/doc.txt"),
            title="Python Programming",
            content="Learn Python programming with examples",
            chunk_index=0,
            language=LanguageCode.ENGLISH,
            created_at=now,
            modified_at=now,
            metadata={},
            start_char=0,
            end_char=35
        )
        
        await repository.store_chunk(chunk)
        
        # Check that terms are indexed
        indexed_terms = repository.get_indexed_terms()
        assert "python" in indexed_terms
        assert "programming" in indexed_terms
        assert "learn" in indexed_terms
        
        # Check term frequencies
        assert repository.get_term_frequency(chunk.id, "python") == 2  # In title and content
        assert repository.get_term_frequency(chunk.id, "programming") == 2  # In title and content


@pytest.mark.slow
class TestDuckDBSearchRepositoryIntegration:
    """Integration tests for DuckDBSearchRepository.
    
    These tests require actual database setup and are marked as slow tests.
    """
    
    @pytest.fixture
    async def duckdb_repository(self, temp_db_path):
        """Provide DuckDBSearchRepository for integration testing."""
        try:
            from src.oboyu.indexer.storage.database_manager import DatabaseManager
            from src.oboyu.indexer.storage.database_service import DatabaseService
            
            # Create a temporary database
            db_manager = DatabaseManager(str(temp_db_path))
            await db_manager.initialize()
            
            db_service = DatabaseService(db_manager)
            repo = DuckDBSearchRepository(db_service)
            
            yield repo
            
            # Clean up
            await db_manager.close()
        except ImportError:
            pytest.skip("DuckDB dependencies not available")
    
    async def test_duckdb_integration_basic(self, duckdb_repository):
        """Basic integration test for DuckDB implementation."""
        from datetime import datetime

        from src.oboyu.domain.entities.chunk import Chunk
        from src.oboyu.domain.value_objects.chunk_id import ChunkId
        from src.oboyu.domain.value_objects.language_code import LanguageCode
        
        now = datetime.now()
        chunk = Chunk(
            id=ChunkId.create(Path("/test/integration.txt"), 0),
            document_path=Path("/test/integration.txt"),
            title="Integration Test",
            content="This is an integration test for DuckDB",
            chunk_index=0,
            language=LanguageCode.ENGLISH,
            created_at=now,
            modified_at=now,
            metadata={"test_type": "integration"},
            start_char=0,
            end_char=38
        )
        
        # Store and retrieve
        await duckdb_repository.store_chunk(chunk)
        retrieved = await duckdb_repository.find_by_chunk_id(chunk.id)
        
        assert retrieved is not None
        assert retrieved.id == chunk.id
        assert retrieved.content == chunk.content
        assert retrieved.metadata["test_type"] == "integration"
