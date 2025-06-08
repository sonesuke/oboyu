"""DuckDB implementation of search repository."""

import logging
from pathlib import Path
from typing import List, Optional

from ...common.types.chunk import Chunk as CommonChunk
from ...domain.entities.chunk import Chunk
from ...domain.entities.query import Query
from ...domain.entities.search_result import SearchResult
from ...domain.value_objects.chunk_id import ChunkId
from ...domain.value_objects.embedding_vector import EmbeddingVector
from ...domain.value_objects.language_code import LanguageCode
from ...domain.value_objects.score import Score
from ...indexer.storage.database_service import DatabaseService
from ...ports.repositories.search_repository import SearchRepository

logger = logging.getLogger(__name__)


class DuckDBSearchRepository(SearchRepository):
    """DuckDB implementation of search repository."""
    
    def __init__(self, database_service: DatabaseService) -> None:
        """Initialize with existing database service."""
        self._db_service = database_service
    
    async def store_chunk(self, chunk: Chunk) -> None:
        """Store a document chunk in the repository."""
        chunk_data = {
            'id': str(chunk.id),
            'path': str(chunk.document_path),
            'title': chunk.title,
            'content': chunk.content,
            'chunk_index': chunk.chunk_index,
            'language': chunk.language.value,
            'created_at': chunk.created_at,
            'modified_at': chunk.modified_at,
            'metadata': chunk.metadata,
            'start_char': chunk.start_char,
            'end_char': chunk.end_char,
            'prefix_content': chunk.prefix_content,
            'prefix_type': chunk.prefix_type
        }
        
        await self._db_service.store_chunk(chunk_data)
    
    async def store_chunks(self, chunks: List[Chunk]) -> None:
        """Store multiple document chunks in the repository."""
        common_chunks = []
        for chunk in chunks:
            common_chunk = CommonChunk(
                id=str(chunk.id),
                path=chunk.document_path,
                title=chunk.title,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                language=chunk.language.value,
                created_at=chunk.created_at,
                modified_at=chunk.modified_at,
                metadata=chunk.metadata,
                prefix_content=chunk.prefix_content,
                prefix_type=chunk.prefix_type
            )
            common_chunks.append(common_chunk)
        
        self._db_service.store_chunks(common_chunks)
    
    async def store_embedding(self, chunk_id: ChunkId, embedding: EmbeddingVector) -> None:
        """Store an embedding vector for a chunk."""
        embedding_data = {
            'chunk_id': str(chunk_id),
            'vector': embedding.to_list(),
            'dimensions': embedding.dimensions
        }
        
        await self._db_service.store_embedding(embedding_data)
    
    async def store_embeddings(self, embeddings: List[tuple[ChunkId, EmbeddingVector]]) -> None:
        """Store multiple embedding vectors."""
        chunk_ids = []
        vectors = []
        for chunk_id, embedding in embeddings:
            chunk_ids.append(str(chunk_id))
            vectors.append(embedding.to_numpy())
        
        self._db_service.store_embeddings(chunk_ids, vectors)
    
    async def find_by_vector_similarity(self, query_vector: EmbeddingVector,
                                      top_k: int, threshold: float = 0.0) -> List[SearchResult]:
        """Find chunks by vector similarity."""
        results = self._db_service.vector_search(
            query_vector=query_vector.to_numpy(),
            limit=top_k,
            threshold=threshold
        )
        
        return [self._convert_to_search_result(result) for result in results]
    
    async def find_by_bm25(self, query: Query) -> List[SearchResult]:
        """Find chunks using BM25 algorithm."""
        results = await self._db_service.bm25_search(
            query_terms=query.get_terms(),
            top_k=query.top_k
        )
        
        return [self._convert_to_search_result(result) for result in results]
    
    async def find_by_chunk_id(self, chunk_id: ChunkId) -> Optional[Chunk]:
        """Find a specific chunk by ID."""
        result = await self._db_service.get_chunk_by_id(str(chunk_id))
        if not result:
            return None
        
        return self._convert_to_chunk(result)
    
    async def delete_chunk(self, chunk_id: ChunkId) -> None:
        """Delete a chunk and its associated data."""
        await self._db_service.delete_chunk(str(chunk_id))
    
    async def delete_chunks_by_document(self, document_path: str) -> None:
        """Delete all chunks for a specific document."""
        self._db_service.delete_chunks_by_path(Path(document_path))
    
    async def get_chunk_count(self) -> int:
        """Get total number of chunks in repository."""
        return await self._db_service.get_chunk_count()
    
    async def get_embedding_count(self) -> int:
        """Get total number of embeddings in repository."""
        return await self._db_service.get_embedding_count()
    
    async def chunk_exists(self, chunk_id: ChunkId) -> bool:
        """Check if a chunk exists in the repository."""
        return await self._db_service.chunk_exists(str(chunk_id))
    
    def _convert_to_search_result(self, result_data: dict) -> SearchResult:
        """Convert database result to domain search result."""
        return SearchResult(
            chunk_id=ChunkId(result_data['chunk_id']),
            document_path=Path(result_data['path']),
            title=result_data['title'],
            content=result_data['content'],
            chunk_index=result_data['chunk_index'],
            language=LanguageCode.from_string(result_data['language']),
            score=Score.create(result_data.get('score', 0.0)),
            metadata=result_data.get('metadata') or {},
            highlighted_content=result_data.get('highlighted_content'),
            context_before=result_data.get('context_before'),
            context_after=result_data.get('context_after'),
            search_terms=result_data.get('search_terms'),
            created_at=result_data.get('created_at')
        )
    
    def _convert_to_chunk(self, chunk_data: dict) -> Chunk:
        """Convert database data to domain chunk."""
        return Chunk(
            id=ChunkId(chunk_data['id']),
            document_path=Path(chunk_data['path']),
            title=chunk_data['title'],
            content=chunk_data['content'],
            chunk_index=chunk_data['chunk_index'],
            language=LanguageCode.from_string(chunk_data['language']),
            created_at=chunk_data['created_at'],
            modified_at=chunk_data['modified_at'],
            metadata=chunk_data.get('metadata') or {},
            start_char=chunk_data['start_char'],
            end_char=chunk_data['end_char'],
            prefix_content=chunk_data.get('prefix_content'),
            prefix_type=chunk_data.get('prefix_type')
        )
