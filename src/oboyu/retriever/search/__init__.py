"""Search components."""

# Core search functionality
# Core search functionality
from oboyu.common.types import SearchFilters, SearchMode, SearchResult
from oboyu.retriever.search.bm25_search import BM25Search
from oboyu.retriever.search.hybrid_search import HybridSearch
from oboyu.retriever.search.hybrid_search_combiner import HybridSearchCombiner

# Search support components
from oboyu.retriever.search.mode_router import SearchModeRouter
from oboyu.retriever.search.result_merger import ResultMerger
from oboyu.retriever.search.score_normalizer import ScoreNormalizer
from oboyu.retriever.search.search_context import SearchContext
from oboyu.retriever.search.search_engine import SearchEngine

# Text processing components
from oboyu.retriever.search.snippet_extractor import SnippetExtractor
from oboyu.retriever.search.text_highlighter import TextHighlighter

# Search implementations
from oboyu.retriever.search.vector_search import VectorSearch

__all__ = [
    # Core search functionality
    "SearchEngine",
    "SearchResult",
    "SearchMode",
    "SearchFilters",
    "SearchContext",
    
    # Search implementations
    "VectorSearch",
    "BM25Search",
    "HybridSearch",
    
    # Search support components
    "SearchModeRouter",
    "HybridSearchCombiner",
    "ResultMerger",
    "ScoreNormalizer",
    
    
    # Text processing components
    "SnippetExtractor",
    "TextHighlighter",
]
