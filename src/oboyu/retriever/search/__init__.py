"""Search components."""

# Core search functionality
# Indexing components
from oboyu.retriever.search.bm25_indexer import BM25Indexer
from oboyu.retriever.search.bm25_search import BM25Search
from oboyu.retriever.search.bm25_statistics_calculator import BM25StatisticsCalculator
from oboyu.retriever.search.engine import SearchEngine, SearchOrchestrator
from oboyu.retriever.search.hybrid_search import HybridSearch
from oboyu.retriever.search.hybrid_search_combiner import HybridSearchCombiner
from oboyu.retriever.search.inverted_index_builder import InvertedIndexBuilder

# Search support components
from oboyu.retriever.search.mode_router import SearchModeRouter
from oboyu.retriever.search.result_merger import ResultMerger
from oboyu.retriever.search.score_normalizer import ScoreNormalizer
from oboyu.retriever.search.search_context import SearchContext
from oboyu.retriever.search.search_filters import SearchFilters
from oboyu.retriever.search.search_mode import SearchMode
from oboyu.retriever.search.search_result import SearchResult

# Text processing components
from oboyu.retriever.search.snippet_extractor import SnippetExtractor
from oboyu.retriever.search.term_frequency_analyzer import TermFrequencyAnalyzer
from oboyu.retriever.search.text_highlighter import TextHighlighter

# Search implementations
from oboyu.retriever.search.vector_search import VectorSearch

__all__ = [
    # Core search functionality
    "SearchEngine",
    "SearchOrchestrator",  # Legacy alias
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
    
    # Indexing components
    "BM25Indexer",
    "BM25StatisticsCalculator",
    "InvertedIndexBuilder",
    "TermFrequencyAnalyzer",
    
    # Text processing components
    "SnippetExtractor",
    "TextHighlighter",
]
