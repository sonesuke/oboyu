"""Hexagonal architecture facade for CLI compatibility."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...application.container import Container
from ...application.indexing.indexing_service import IndexingService
from ...application.searching.search_service import SearchService
from ...domain.entities.query import Query
from ...domain.value_objects.language_code import LanguageCode
from ...domain.value_objects.search_mode import SearchMode

logger = logging.getLogger(__name__)


class HexagonalFacade:
    """Facade providing backward-compatible interface using hexagonal architecture."""
    
    def __init__(self, container: Container) -> None:
        """Initialize with dependency injection container."""
        self._container = container
    
    @property
    def indexing_service(self) -> IndexingService:
        """Get indexing service."""
        return self._container.get_indexing_service()
    
    @property
    def search_service(self) -> SearchService:
        """Get search service."""
        return self._container.get_search_service()
    
    async def index_files(self, file_paths: List[Path]) -> None:
        """Index multiple files."""
        await self.indexing_service.index_documents(file_paths)
    
    async def index_directory(self, directory_path: Path,
                            include_patterns: Optional[List[str]] = None,
                            exclude_patterns: Optional[List[str]] = None) -> None:
        """Index a directory."""
        await self.indexing_service.index_directory(
            directory_path,
            include_patterns or [],
            exclude_patterns or []
        )
    
    async def search(self, query_text: str, mode: str = "hybrid",
                   top_k: int = 10, threshold: float = 0.0,
                   language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform search and return results as dictionaries for backward compatibility."""
        search_mode = SearchMode.from_string(mode)
        lang_code = LanguageCode.from_string(language) if language else None
        
        query = Query(
            text=query_text,
            mode=search_mode,
            top_k=top_k,
            language=lang_code,
            similarity_threshold=threshold
        )
        
        results = await self.search_service.search_with_reranking(query)
        
        return [self._convert_result_to_dict(result) for result in results]
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID."""
        result = await self.search_service.get_chunk_by_id(chunk_id)
        return self._convert_result_to_dict(result) if result else None
    
    def _convert_result_to_dict(self, result: Any) -> Dict[str, Any]:  # noqa: ANN401
        """Convert domain search result to dictionary."""
        return {
            "chunk_id": str(result.chunk_id),
            "path": str(result.document_path),
            "title": result.title,
            "content": result.content,
            "chunk_index": result.chunk_index,
            "language": result.language.value,
            "score": result.score.value,
            "metadata": result.metadata,
            "highlighted_content": result.highlighted_content,
            "context_before": result.context_before,
            "context_after": result.context_after,
            "search_terms": result.search_terms,
            "created_at": result.created_at.isoformat() if result.created_at else None
        }
