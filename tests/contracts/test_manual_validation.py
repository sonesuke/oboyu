"""Manual validation tests for contract functionality."""

import asyncio
from datetime import datetime
from pathlib import Path

from src.oboyu.domain.entities.chunk import Chunk
from src.oboyu.domain.entities.query import Query
from src.oboyu.domain.value_objects.chunk_id import ChunkId
from src.oboyu.domain.value_objects.embedding_vector import EmbeddingVector
from src.oboyu.domain.value_objects.language_code import LanguageCode
from src.oboyu.domain.value_objects.search_mode import SearchMode

from .in_memory_search_repository import InMemorySearchRepository


def test_in_memory_repository_basic_functionality():
    """Test basic functionality of the in-memory repository."""
    
    async def run_test():
        # Create repository
        repo = InMemorySearchRepository()
        
        # Create test chunk
        now = datetime.now()
        chunk = Chunk(
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
        
        # Test storage and retrieval
        await repo.store_chunk(chunk)
        assert await repo.chunk_exists(chunk.id)
        
        retrieved = await repo.find_by_chunk_id(chunk.id)
        assert retrieved is not None
        assert retrieved.id == chunk.id
        assert retrieved.content == chunk.content
        
        # Test embedding storage and search
        embedding = EmbeddingVector.create([0.1, 0.2, 0.3, 0.4, 0.5])
        await repo.store_embedding(chunk.id, embedding)
        
        results = await repo.find_by_vector_similarity(
            query_vector=embedding,
            top_k=5,
            threshold=0.0
        )
        assert len(results) > 0
        assert results[0].chunk_id == chunk.id
        
        # Test BM25 search
        query = Query(
            text="test content",
            mode=SearchMode.BM25,
            top_k=10
        )
        
        results = await repo.find_by_bm25(query)
        assert len(results) > 0
        assert results[0].chunk_id == chunk.id
        
        # Test counts
        assert await repo.get_chunk_count() == 1
        assert await repo.get_embedding_count() == 1
        
        # Test deletion
        await repo.delete_chunk(chunk.id)
        assert not await repo.chunk_exists(chunk.id)
        assert await repo.get_chunk_count() == 0
        
        print("✓ All basic functionality tests passed")
    
    # Run the async test
    asyncio.run(run_test())


def test_multiple_chunks_storage():
    """Test storing and managing multiple chunks."""
    
    async def run_test():
        repo = InMemorySearchRepository()
        
        now = datetime.now()
        chunks = [
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
        
        # Store multiple chunks
        await repo.store_chunks(chunks)
        assert await repo.get_chunk_count() == 3
        
        # Test document-based deletion
        await repo.delete_chunks_by_document("/test/doc1.txt")
        assert await repo.get_chunk_count() == 1
        
        # Verify the right chunks were deleted
        assert not await repo.chunk_exists(chunks[0].id)
        assert not await repo.chunk_exists(chunks[1].id)
        assert await repo.chunk_exists(chunks[2].id)
        
        print("✓ Multiple chunks storage test passed")
    
    asyncio.run(run_test())


def test_bm25_indexing_details():
    """Test BM25 indexing specific functionality."""
    
    async def run_test():
        repo = InMemorySearchRepository()
        
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
        
        await repo.store_chunk(chunk)
        
        # Check indexed terms
        indexed_terms = repo.get_indexed_terms()
        assert "python" in indexed_terms
        assert "programming" in indexed_terms
        assert "learn" in indexed_terms
        
        # Check term frequencies
        assert repo.get_term_frequency(chunk.id, "python") == 2  # In title and content
        assert repo.get_term_frequency(chunk.id, "programming") == 2  # In title and content
        
        print("✓ BM25 indexing test passed")
    
    asyncio.run(run_test())
