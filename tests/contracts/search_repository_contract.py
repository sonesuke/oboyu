"""Contract tests for SearchRepository implementations."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

from src.oboyu.domain.entities.chunk import Chunk
from src.oboyu.domain.entities.query import Query
from src.oboyu.domain.value_objects.chunk_id import ChunkId
from src.oboyu.domain.value_objects.embedding_vector import EmbeddingVector
from src.oboyu.domain.value_objects.language_code import LanguageCode
from src.oboyu.domain.value_objects.search_mode import SearchMode
from src.oboyu.ports.repositories.search_repository import SearchRepository


class SearchRepositoryContract:
    """Contract tests that all SearchRepository implementations must satisfy.
    
    This class defines the behavioral contracts that guarantee consistent
    behavior across different SearchRepository implementations.
    """
    
    @pytest.fixture
    def sample_chunk(self) -> Chunk:
        """Create a sample chunk for testing."""
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
    
    @pytest.fixture
    def sample_chunks(self) -> List[Chunk]:
        """Create multiple sample chunks for testing."""
        now = datetime.now()
        return [
            Chunk(
                id=ChunkId.create(Path("/test/doc1.txt"), 0),
                document_path=Path("/test/doc1.txt"),
                title="First Document",
                content="Python programming language tutorial.",
                chunk_index=0,
                language=LanguageCode.ENGLISH,
                created_at=now,
                modified_at=now,
                metadata={"category": "programming"},
                start_char=0,
                end_char=33
            ),
            Chunk(
                id=ChunkId.create(Path("/test/doc1.txt"), 1),
                document_path=Path("/test/doc1.txt"),
                title="First Document",
                content="Advanced data structures and algorithms.",
                chunk_index=1,
                language=LanguageCode.ENGLISH,
                created_at=now,
                modified_at=now,
                metadata={"category": "programming"},
                start_char=34,
                end_char=71
            ),
            Chunk(
                id=ChunkId.create(Path("/test/doc2.txt"), 0),
                document_path=Path("/test/doc2.txt"),
                title="Second Document",
                content="Machine learning and artificial intelligence.",
                chunk_index=0,
                language=LanguageCode.ENGLISH,
                created_at=now,
                modified_at=now,
                metadata={"category": "ai"},
                start_char=0,
                end_char=44
            )
        ]
    
    @pytest.fixture
    def sample_embedding(self) -> EmbeddingVector:
        """Create a sample embedding vector."""
        return EmbeddingVector.create([0.1, 0.2, 0.3, 0.4, 0.5])
    
    @pytest.fixture
    def sample_query(self) -> Query:
        """Create a sample query."""
        return Query(
            text="Python programming",
            mode=SearchMode.BM25,
            top_k=10,
            language=LanguageCode.ENGLISH,
            similarity_threshold=0.0
        )
    
    # Basic Storage and Retrieval Contracts
    
    async def test_store_and_retrieve_single_chunk(
        self, repository: SearchRepository, sample_chunk: Chunk
    ):
        """Contract: Store a chunk and retrieve it by ID."""
        # Store the chunk
        await repository.store_chunk(sample_chunk)
        
        # Retrieve the chunk
        retrieved_chunk = await repository.find_by_chunk_id(sample_chunk.id)
        
        # Verify chunk was stored and retrieved correctly
        assert retrieved_chunk is not None
        assert retrieved_chunk.id == sample_chunk.id
        assert retrieved_chunk.content == sample_chunk.content
        assert retrieved_chunk.title == sample_chunk.title
        assert retrieved_chunk.document_path == sample_chunk.document_path
    
    async def test_store_and_retrieve_multiple_chunks(
        self, repository: SearchRepository, sample_chunks: List[Chunk]
    ):
        """Contract: Store multiple chunks and retrieve them individually."""
        # Store multiple chunks
        await repository.store_chunks(sample_chunks)
        
        # Retrieve each chunk by ID
        for original_chunk in sample_chunks:
            retrieved_chunk = await repository.find_by_chunk_id(original_chunk.id)
            assert retrieved_chunk is not None
            assert retrieved_chunk.id == original_chunk.id
            assert retrieved_chunk.content == original_chunk.content
    
    async def test_chunk_exists_behavior(
        self, repository: SearchRepository, sample_chunk: Chunk
    ):
        """Contract: chunk_exists returns correct boolean values."""
        # Initially chunk should not exist
        assert not await repository.chunk_exists(sample_chunk.id)
        
        # After storing, chunk should exist
        await repository.store_chunk(sample_chunk)
        assert await repository.chunk_exists(sample_chunk.id)
        
        # After deletion, chunk should not exist
        await repository.delete_chunk(sample_chunk.id)
        assert not await repository.chunk_exists(sample_chunk.id)
    
    # Embedding Storage and Vector Search Contracts
    
    async def test_store_and_search_by_embedding(
        self, repository: SearchRepository, sample_chunk: Chunk, sample_embedding: EmbeddingVector
    ):
        """Contract: Store embedding and find by vector similarity."""
        # Store chunk and its embedding
        await repository.store_chunk(sample_chunk)
        await repository.store_embedding(sample_chunk.id, sample_embedding)
        
        # Search by vector similarity
        results = await repository.find_by_vector_similarity(
            query_vector=sample_embedding,
            top_k=5,
            threshold=0.0
        )
        
        # Should find the stored chunk
        assert len(results) > 0
        assert any(result.chunk_id == sample_chunk.id for result in results)
    
    async def test_store_multiple_embeddings(
        self, repository: SearchRepository, sample_chunks: List[Chunk]
    ):
        """Contract: Store multiple embeddings efficiently."""
        # Store chunks first
        await repository.store_chunks(sample_chunks)
        
        # Create embeddings for each chunk
        embeddings = [
            (chunk.id, EmbeddingVector.create([0.1 * i, 0.2 * i, 0.3 * i]))
            for i, chunk in enumerate(sample_chunks, 1)
        ]
        
        # Store all embeddings
        await repository.store_embeddings(embeddings)
        
        # Verify each embedding can be used for search
        for chunk_id, embedding in embeddings:
            results = await repository.find_by_vector_similarity(
                query_vector=embedding,
                top_k=1,
                threshold=0.0
            )
            assert len(results) > 0
            assert results[0].chunk_id == chunk_id
    
    # BM25 Search Contracts
    
    async def test_bm25_search_basic(
        self, repository: SearchRepository, sample_chunks: List[Chunk], sample_query: Query
    ):
        """Contract: BM25 search returns relevant results."""
        # Store chunks
        await repository.store_chunks(sample_chunks)
        
        # Search using BM25
        results = await repository.find_by_bm25(sample_query)
        
        # Should return a list (may be empty)
        assert isinstance(results, list)
        
        # If results found, they should be valid SearchResult objects
        for result in results:
            assert hasattr(result, 'chunk_id')
            assert hasattr(result, 'content')
            assert hasattr(result, 'score')
    
    async def test_empty_query_behavior(self, repository: SearchRepository):
        """Contract: Empty queries return empty results without errors."""
        query = Query(
            text=" ",  # Empty after strip
            mode=SearchMode.BM25,
            top_k=10
        )
        
        try:
            results = await repository.find_by_bm25(query)
            assert isinstance(results, list)
        except ValueError:
            # Acceptable to reject empty queries with ValueError
            pass
    
    # Deletion Contracts
    
    async def test_delete_single_chunk(
        self, repository: SearchRepository, sample_chunk: Chunk
    ):
        """Contract: Delete single chunk removes all associated data."""
        # Store chunk
        await repository.store_chunk(sample_chunk)
        await repository.store_embedding(
            sample_chunk.id,
            EmbeddingVector.create([0.1, 0.2, 0.3])
        )
        
        # Verify chunk exists
        assert await repository.chunk_exists(sample_chunk.id)
        
        # Delete chunk
        await repository.delete_chunk(sample_chunk.id)
        
        # Verify chunk is deleted
        assert not await repository.chunk_exists(sample_chunk.id)
        retrieved_chunk = await repository.find_by_chunk_id(sample_chunk.id)
        assert retrieved_chunk is None
    
    async def test_delete_chunks_by_document(
        self, repository: SearchRepository, sample_chunks: List[Chunk]
    ):
        """Contract: Delete all chunks for a specific document."""
        # Store chunks
        await repository.store_chunks(sample_chunks)
        
        # Delete chunks for first document
        document_path = "/test/doc1.txt"
        await repository.delete_chunks_by_document(document_path)
        
        # Verify chunks from first document are deleted
        doc1_chunks = [c for c in sample_chunks if str(c.document_path) == document_path]
        for chunk in doc1_chunks:
            assert not await repository.chunk_exists(chunk.id)
        
        # Verify chunks from other documents still exist
        other_chunks = [c for c in sample_chunks if str(c.document_path) != document_path]
        for chunk in other_chunks:
            assert await repository.chunk_exists(chunk.id)
    
    # Statistics Contracts
    
    async def test_chunk_count_accuracy(
        self, repository: SearchRepository, sample_chunks: List[Chunk]
    ):
        """Contract: Chunk count reflects actual stored chunks."""
        # Initially should have 0 chunks
        initial_count = await repository.get_chunk_count()
        
        # Store chunks
        await repository.store_chunks(sample_chunks)
        
        # Count should increase by number of stored chunks
        final_count = await repository.get_chunk_count()
        assert final_count == initial_count + len(sample_chunks)
    
    async def test_embedding_count_accuracy(
        self, repository: SearchRepository, sample_chunks: List[Chunk]
    ):
        """Contract: Embedding count reflects actual stored embeddings."""
        # Store chunks first
        await repository.store_chunks(sample_chunks)
        
        # Initially should have 0 embeddings
        initial_count = await repository.get_embedding_count()
        
        # Store embeddings
        embeddings = [
            (chunk.id, EmbeddingVector.create([0.1, 0.2, 0.3]))
            for chunk in sample_chunks
        ]
        await repository.store_embeddings(embeddings)
        
        # Count should increase by number of stored embeddings
        final_count = await repository.get_embedding_count()
        assert final_count == initial_count + len(embeddings)
    
    # Error Handling and Edge Cases Contracts
    
    async def test_nonexistent_chunk_retrieval(self, repository: SearchRepository):
        """Contract: Retrieving nonexistent chunk returns None."""
        nonexistent_id = ChunkId.create(Path("/nonexistent/file.txt"), 999)
        result = await repository.find_by_chunk_id(nonexistent_id)
        assert result is None
    
    async def test_duplicate_chunk_storage(
        self, repository: SearchRepository, sample_chunk: Chunk
    ):
        """Contract: Storing same chunk twice should not create duplicates."""
        # Store chunk twice
        await repository.store_chunk(sample_chunk)
        await repository.store_chunk(sample_chunk)
        
        # Should still have only one chunk
        count = await repository.get_chunk_count()
        assert count == 1
        
        # Should still be able to retrieve the chunk
        retrieved = await repository.find_by_chunk_id(sample_chunk.id)
        assert retrieved is not None
        assert retrieved.id == sample_chunk.id
    
    async def test_delete_nonexistent_chunk(self, repository: SearchRepository):
        """Contract: Deleting nonexistent chunk should not raise errors."""
        nonexistent_id = ChunkId.create(Path("/nonexistent/file.txt"), 999)
        
        # Should not raise an exception
        await repository.delete_chunk(nonexistent_id)
        
        # Repository should remain in consistent state
        count = await repository.get_chunk_count()
        assert isinstance(count, int)
        assert count >= 0
    
    # Performance and Concurrency Contracts
    
    async def test_concurrent_access_safety(
        self, repository: SearchRepository, sample_chunks: List[Chunk]
    ):
        """Contract: Repository handles concurrent access safely."""
        async def store_chunk(chunk: Chunk):
            await repository.store_chunk(chunk)
        
        # Store chunks concurrently
        tasks = [store_chunk(chunk) for chunk in sample_chunks]
        await asyncio.gather(*tasks)
        
        # All chunks should be stored successfully
        for chunk in sample_chunks:
            assert await repository.chunk_exists(chunk.id)
        
        # Total count should be correct
        count = await repository.get_chunk_count()
        assert count == len(sample_chunks)
    
    # Search Result Quality Contracts
    
    async def test_search_result_ordering(
        self, repository: SearchRepository, sample_chunks: List[Chunk]
    ):
        """Contract: Search results are ordered by relevance (descending score)."""
        # Store chunks
        await repository.store_chunks(sample_chunks)
        
        query = Query(
            text="programming",
            mode=SearchMode.BM25,
            top_k=10
        )
        
        results = await repository.find_by_bm25(query)
        
        if len(results) > 1:
            # Results should be ordered by score (descending)
            for i in range(len(results) - 1):
                assert results[i].score.value >= results[i + 1].score.value
    
    async def test_top_k_limit_respected(
        self, repository: SearchRepository, sample_chunks: List[Chunk]
    ):
        """Contract: Search respects top_k limit."""
        # Store many chunks
        await repository.store_chunks(sample_chunks)
        
        # Search with limited results
        top_k = 2
        query = Query(
            text="test",
            mode=SearchMode.BM25,
            top_k=top_k
        )
        
        results = await repository.find_by_bm25(query)
        
        # Should not exceed top_k limit
        assert len(results) <= top_k
    
    async def test_similarity_threshold_respected(
        self, repository: SearchRepository, sample_chunk: Chunk, sample_embedding: EmbeddingVector
    ):
        """Contract: Vector search respects similarity threshold."""
        # Store chunk and embedding
        await repository.store_chunk(sample_chunk)
        await repository.store_embedding(sample_chunk.id, sample_embedding)
        
        # Search with high threshold that should exclude results
        results = await repository.find_by_vector_similarity(
            query_vector=EmbeddingVector.create([1.0, 0.0, 0.0, 0.0, 0.0]),  # Very different vector
            top_k=10,
            threshold=0.9  # High threshold
        )
        
        # Should respect threshold and potentially return no results
        assert isinstance(results, list)
        for result in results:
            assert result.score.value >= 0.9


# Utility function to run all contract tests for a repository
async def run_all_contracts(repository: SearchRepository) -> None:
    """Run all contract tests against a repository implementation.
    
    This function can be used to validate that a repository implementation
    satisfies all the required contracts.
    """
    contract = SearchRepositoryContract()
    
    # Create test fixtures
    sample_chunk = contract.sample_chunk()
    sample_chunks = contract.sample_chunks()
    sample_embedding = contract.sample_embedding()
    sample_query = contract.sample_query()
    
    # Run all contract tests
    test_methods = [
        method for method in dir(contract)
        if method.startswith('test_') and callable(getattr(contract, method))
    ]
    
    for method_name in test_methods:
        method = getattr(contract, method_name)
        try:
            if 'sample_chunk' in method.__code__.co_varnames:
                await method(repository, sample_chunk)
            elif 'sample_chunks' in method.__code__.co_varnames:
                await method(repository, sample_chunks)
            elif 'sample_embedding' in method.__code__.co_varnames:
                await method(repository, sample_chunk, sample_embedding)
            elif 'sample_query' in method.__code__.co_varnames:
                await method(repository, sample_chunks, sample_query)
            else:
                await method(repository)
            print(f"✓ {method_name}")
        except Exception as e:
            print(f"✗ {method_name}: {e}")
            raise
