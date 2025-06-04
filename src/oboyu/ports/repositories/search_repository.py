"""Search repository port interface."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ...domain.entities.chunk import Chunk
from ...domain.entities.query import Query
from ...domain.entities.search_result import SearchResult
from ...domain.value_objects.chunk_id import ChunkId
from ...domain.value_objects.embedding_vector import EmbeddingVector


class SearchRepository(ABC):
    """Abstract interface for search repository operations."""
    
    @abstractmethod
    async def store_chunk(self, chunk: Chunk) -> None:
        """Store a document chunk in the repository."""
        pass
    
    @abstractmethod
    async def store_chunks(self, chunks: List[Chunk]) -> None:
        """Store multiple document chunks in the repository."""
        pass
    
    @abstractmethod
    async def store_embedding(self, chunk_id: ChunkId, embedding: EmbeddingVector) -> None:
        """Store an embedding vector for a chunk."""
        pass
    
    @abstractmethod
    async def store_embeddings(self, embeddings: List[tuple[ChunkId, EmbeddingVector]]) -> None:
        """Store multiple embedding vectors."""
        pass
    
    @abstractmethod
    async def find_by_vector_similarity(self, query_vector: EmbeddingVector,
                                      top_k: int, threshold: float = 0.0) -> List[SearchResult]:
        """Find chunks by vector similarity."""
        pass
    
    @abstractmethod
    async def find_by_bm25(self, query: Query) -> List[SearchResult]:
        """Find chunks using BM25 algorithm."""
        pass
    
    @abstractmethod
    async def find_by_chunk_id(self, chunk_id: ChunkId) -> Optional[Chunk]:
        """Find a specific chunk by ID."""
        pass
    
    @abstractmethod
    async def delete_chunk(self, chunk_id: ChunkId) -> None:
        """Delete a chunk and its associated data."""
        pass
    
    @abstractmethod
    async def delete_chunks_by_document(self, document_path: str) -> None:
        """Delete all chunks for a specific document."""
        pass
    
    @abstractmethod
    async def get_chunk_count(self) -> int:
        """Get total number of chunks in repository."""
        pass
    
    @abstractmethod
    async def get_embedding_count(self) -> int:
        """Get total number of embeddings in repository."""
        pass
    
    @abstractmethod
    async def chunk_exists(self, chunk_id: ChunkId) -> bool:
        """Check if a chunk exists in the repository."""
        pass
