"""Search engine domain service - pure business logic for search operations."""

import logging
from typing import Dict, List, Optional

from ..entities.query import Query
from ..entities.search_result import SearchResult
from ..value_objects.score import Score

logger = logging.getLogger(__name__)


class SearchEngine:
    """Pure domain service for search logic."""
    
    def combine_results(self, vector_results: List[SearchResult],
                       bm25_results: List[SearchResult],
                       vector_weight: float = 0.7,
                       bm25_weight: float = 0.3) -> List[SearchResult]:
        """Combine vector and BM25 results using weighted scoring."""
        if abs(vector_weight + bm25_weight - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
        
        result_map: Dict[str, SearchResult] = {}
        
        for result in vector_results:
            chunk_id = str(result.chunk_id)
            weighted_score = result.score.value * vector_weight
            
            new_result = SearchResult(
                chunk_id=result.chunk_id,
                document_path=result.document_path,
                title=result.title,
                content=result.content,
                chunk_index=result.chunk_index,
                language=result.language,
                score=Score.create(weighted_score),
                metadata=result.metadata,
                highlighted_content=result.highlighted_content,
                context_before=result.context_before,
                context_after=result.context_after,
                search_terms=result.search_terms,
                created_at=result.created_at
            )
            result_map[chunk_id] = new_result
        
        for result in bm25_results:
            chunk_id = str(result.chunk_id)
            weighted_score = result.score.value * bm25_weight
            
            if chunk_id in result_map:
                existing_result = result_map[chunk_id]
                combined_score = existing_result.score.value + weighted_score
                
                result_map[chunk_id] = SearchResult(
                    chunk_id=existing_result.chunk_id,
                    document_path=existing_result.document_path,
                    title=existing_result.title,
                    content=existing_result.content,
                    chunk_index=existing_result.chunk_index,
                    language=existing_result.language,
                    score=Score.create(combined_score),
                    metadata=existing_result.metadata,
                    highlighted_content=result.highlighted_content or existing_result.highlighted_content,
                    context_before=existing_result.context_before,
                    context_after=existing_result.context_after,
                    search_terms=self._merge_search_terms(existing_result.search_terms, result.search_terms),
                    created_at=existing_result.created_at
                )
            else:
                new_result = SearchResult(
                    chunk_id=result.chunk_id,
                    document_path=result.document_path,
                    title=result.title,
                    content=result.content,
                    chunk_index=result.chunk_index,
                    language=result.language,
                    score=Score.create(weighted_score),
                    metadata=result.metadata,
                    highlighted_content=result.highlighted_content,
                    context_before=result.context_before,
                    context_after=result.context_after,
                    search_terms=result.search_terms,
                    created_at=result.created_at
                )
                result_map[chunk_id] = new_result
        
        combined_results = list(result_map.values())
        return sorted(combined_results, key=lambda r: r.score.value, reverse=True)
    
    def post_process_results(self, query: Query, results: List[SearchResult]) -> List[SearchResult]:
        """Post-process search results with highlighting and filtering."""
        processed_results = []
        
        for result in results:
            if result.score.meets_threshold(query.similarity_threshold):
                processed_result = self._add_highlights(query, result)
                processed_results.append(processed_result)
        
        return processed_results[:query.top_k]
    
    def _add_highlights(self, query: Query, result: SearchResult) -> SearchResult:
        """Add highlighting to search result."""
        query_terms = query.get_terms()
        content = result.content.lower()
        
        highlighted_content = result.content
        found_terms = []
        
        for term in query_terms:
            if term.lower() in content:
                found_terms.append(term)
                highlighted_content = highlighted_content.replace(
                    term, f"**{term}**"
                )
        
        if found_terms:
            return result.create_with_highlights(highlighted_content, found_terms)
        
        return result
    
    def _merge_search_terms(self, terms1: Optional[List[str]] = None, terms2: Optional[List[str]] = None) -> Optional[List[str]]:
        """Merge search terms from multiple results."""
        merged = set()
        
        if terms1:
            merged.update(terms1)
        if terms2:
            merged.update(terms2)
        
        return list(merged) if merged else None
