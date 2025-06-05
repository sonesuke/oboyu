"""Search application service - coordinates search use cases."""

import logging
from typing import List, Optional

from ...domain.entities.query import Query
from ...domain.entities.search_result import SearchResult
from ...domain.services.search_engine import SearchEngine
from ...domain.value_objects.search_mode import SearchMode
from ...ports.repositories.search_repository import SearchRepository
from ...ports.services.embedding_service import EmbeddingService
from ...ports.services.reranker_service import RerankerService

logger = logging.getLogger(__name__)


class SearchService:
    """Application service for search operations."""
    
    def __init__(
        self,
        search_repository: SearchRepository,
        embedding_service: EmbeddingService,
        search_engine: SearchEngine,
        reranker_service: Optional[RerankerService] = None
    ) -> None:
        """Initialize with dependencies injected."""
        self._search_repository = search_repository
        self._embedding_service = embedding_service
        self._search_engine = search_engine
        self._reranker_service = reranker_service
    
    async def search(self, query: Query) -> List[SearchResult]:
        """Perform search with the specified query."""
        if query.mode == SearchMode.VECTOR:
            return await self._vector_search(query)
        elif query.mode == SearchMode.BM25:
            return await self._bm25_search(query)
        elif query.mode == SearchMode.HYBRID:
            return await self._hybrid_search(query)
        else:
            raise ValueError(f"Unsupported search mode: {query.mode}")
    
    async def search_with_reranking(self, query: Query) -> List[SearchResult]:
        """Perform search with reranking if available."""
        results = await self.search(query)
        
        if self._reranker_service and await self._reranker_service.is_available():
            results = await self._reranker_service.rerank(query, results)
        
        return results
    
    async def _vector_search(self, query: Query) -> List[SearchResult]:
        """Perform vector similarity search."""
        query_embedding = await self._embedding_service.generate_query_embedding(query.text)
        
        results = await self._search_repository.find_by_vector_similarity(
            query_vector=query_embedding,
            top_k=query.top_k,
            threshold=query.similarity_threshold
        )
        
        return self._search_engine.post_process_results(query, results)
    
    async def _bm25_search(self, query: Query) -> List[SearchResult]:
        """Perform BM25 text search."""
        results = await self._search_repository.find_by_bm25(query)
        return self._search_engine.post_process_results(query, results)
    
    async def _hybrid_search(self, query: Query) -> List[SearchResult]:
        """Perform hybrid search combining vector and BM25."""
        vector_query = query.create_variant_with_mode(SearchMode.VECTOR)
        bm25_query = query.create_variant_with_mode(SearchMode.BM25)
        
        vector_results = await self._vector_search(vector_query)
        bm25_results = await self._bm25_search(bm25_query)
        
        combined_results = self._search_engine.combine_results(
            vector_results, bm25_results, vector_weight=0.7, bm25_weight=0.3
        )
        
        return combined_results[:query.top_k]
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """Get a specific chunk by ID and convert to search result."""
        from ...domain.value_objects.chunk_id import ChunkId
        from ...domain.value_objects.score import Score
        
        chunk = await self._search_repository.find_by_chunk_id(ChunkId(chunk_id))
        if not chunk:
            return None
        
        return SearchResult(
            chunk_id=chunk.id,
            document_path=chunk.document_path,
            title=chunk.title,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            language=chunk.language,
            score=Score.create(1.0),
            metadata=chunk.metadata,
            created_at=chunk.created_at
        )
