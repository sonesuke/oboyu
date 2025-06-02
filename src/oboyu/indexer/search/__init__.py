"""Search components."""

# Core search functionality
# Indexing components
from oboyu.indexer.search.bm25_indexer import BM25Indexer
from oboyu.indexer.search.bm25_search import BM25Search
from oboyu.indexer.search.bm25_statistics_calculator import BM25StatisticsCalculator
from oboyu.indexer.search.engine import SearchEngine, SearchOrchestrator
from oboyu.indexer.search.hybrid_search import HybridSearch
from oboyu.indexer.search.hybrid_search_combiner import HybridSearchCombiner
from oboyu.indexer.search.inverted_index_builder import InvertedIndexBuilder

# Search support components
from oboyu.indexer.search.mode_router import SearchModeRouter
from oboyu.indexer.search.result_merger import ResultMerger
from oboyu.indexer.search.score_normalizer import ScoreNormalizer
from oboyu.indexer.search.search_context import SearchContext
from oboyu.indexer.search.search_filters import SearchFilters
from oboyu.indexer.search.search_mode import SearchMode
from oboyu.indexer.search.search_result import SearchResult

# Text processing components
from oboyu.indexer.search.snippet_extractor import SnippetExtractor
from oboyu.indexer.search.term_frequency_analyzer import TermFrequencyAnalyzer
from oboyu.indexer.search.text_highlighter import TextHighlighter

# Search implementations
from oboyu.indexer.search.vector_search import VectorSearch

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
