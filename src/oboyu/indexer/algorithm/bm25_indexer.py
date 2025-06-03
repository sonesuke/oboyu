"""BM25 indexer for Oboyu.

This module provides BM25 indexing functionality for Japanese text search,
orchestrating specialized components for index building, statistics, and term analysis.
"""

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from oboyu.common.services import TokenizerService, create_tokenizer
from oboyu.common.types import Chunk
from oboyu.indexer.algorithm.bm25_statistics_calculator import BM25StatisticsCalculator
from oboyu.indexer.algorithm.inverted_index_builder import InvertedIndexBuilder
from oboyu.indexer.algorithm.term_frequency_analyzer import TermFrequencyAnalyzer

logger = logging.getLogger(__name__)


class BM25Indexer:
    """BM25 indexer that orchestrates specialized components for indexing.

    This class coordinates inverted index building, statistics calculation,
    and term frequency analysis using focused, single-responsibility components.
    """

    def __init__(
        self,
        index_builder: Optional[InvertedIndexBuilder] = None,
        statistics_calculator: Optional[BM25StatisticsCalculator] = None,
        term_analyzer: Optional[TermFrequencyAnalyzer] = None,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer_class: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, object]] = None,
        use_stopwords: bool = False,
        min_doc_frequency: int = 1,
        store_positions: bool = True,
    ) -> None:
        """Initialize BM25 indexer with specialized components.

        Args:
            index_builder: Optional inverted index builder component
            statistics_calculator: Optional BM25 statistics calculator component
            term_analyzer: Optional term frequency analyzer component
            k1: BM25 k1 parameter (term saturation)
            b: BM25 b parameter (length normalization)
            tokenizer_class: Optional tokenizer class name
            tokenizer_kwargs: Optional tokenizer configuration
            use_stopwords: Whether to use stopword filtering
            min_doc_frequency: Minimum document frequency for terms
            store_positions: Whether to store term positions

        """
        self.k1 = k1
        self.b = b
        self.use_stopwords = use_stopwords
        self.min_doc_frequency = min_doc_frequency
        self.store_positions = store_positions

        # Create tokenizer
        tokenizer_kwargs = tokenizer_kwargs or {}
        # Extract specific parameters for create_tokenizer
        min_token_length = tokenizer_kwargs.get("min_token_length", 2)
        if isinstance(min_token_length, int):
            pass  # It's already an int
        else:
            min_token_length = 2  # Default fallback

        # tokenizer_class is actually the language parameter
        language = tokenizer_class or "ja"
        self.tokenizer = create_tokenizer(language=language, min_token_length=min_token_length, use_stopwords=use_stopwords)

        # Store tokenizer config for parallel processing
        self.tokenizer_config = {"language": language, "min_token_length": min_token_length, "use_stopwords": use_stopwords}
        
        # Initialize specialized components
        self.index_builder = index_builder or InvertedIndexBuilder(store_positions=store_positions)
        self.statistics_calculator = statistics_calculator or BM25StatisticsCalculator()
        # Create TokenizerService for the term analyzer if needed
        if term_analyzer is None:
            tokenizer_service = TokenizerService(language, {"min_token_length": min_token_length})
            self.term_analyzer = TermFrequencyAnalyzer(tokenizer_service)
        else:
            self.term_analyzer = term_analyzer
        
        # Backward compatibility properties
        self._term_freq_cache: Dict[int, Dict[str, int]] = {}
    
    # Backward compatibility properties
    @property
    def inverted_index(self) -> Dict[str, List[Tuple[str, int, Optional[List[int]]]]]:
        """Access to inverted index for backward compatibility."""
        return self.index_builder.get_index_data()
    
    @property
    def document_frequencies(self) -> Dict[str, int]:
        """Access to document frequencies for backward compatibility."""
        return self.statistics_calculator.document_frequencies
    
    @property
    def collection_frequencies(self) -> Dict[str, int]:
        """Access to collection frequencies for backward compatibility."""
        return self.statistics_calculator.collection_frequencies
    
    @property
    def document_lengths(self) -> Dict[str, int]:
        """Access to document lengths for backward compatibility."""
        return self.statistics_calculator.document_lengths
    
    @property
    def document_count(self) -> int:
        """Access to document count for backward compatibility."""
        return self.statistics_calculator.document_count
    
    @property
    def total_document_length(self) -> int:
        """Access to total document length for backward compatibility."""
        return self.statistics_calculator.total_document_length

    def index_chunks(self, chunks: List[Chunk], progress_callback: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, int]:
        """Index a batch of chunks for BM25 search.

        Args:
            chunks: List of chunks to index
            progress_callback: Optional callback for progress updates

        Returns:
            Statistics about the indexing operation

        """
        stats = self._initialize_stats()
        
        # Process chunks using specialized components
        self._process_chunks_with_components(chunks, stats, progress_callback)
        
        # Report vocabulary and index building progress
        self._report_post_processing_progress(progress_callback)
        
        # Track filtered terms
        if hasattr(self.tokenizer, "filtered_by_stopwords"):
            self.tokenizer.filtered_by_stopwords = defaultdict(int)
        
        # Debug logging
        self._log_debug_info()
        
        stats["unique_terms"] = len(self.inverted_index)
        return stats
    
    def _log_debug_info(self) -> None:
        """Log debug information about the index."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        
        logger.debug("=== BM25 Inverted Index Analysis ===")
        logger.debug(f"Total unique terms (vocabulary size): {len(self.inverted_index)}")
        logger.debug(f"Total chunks indexed: {self.document_count}")
        
        # Calculate distribution of documents per term
        term_doc_counts: Dict[int, int] = defaultdict(int)
        for term, postings in self.inverted_index.items():
            num_docs = len(postings)
            term_doc_counts[num_docs] += 1
        
        # Log distribution
        logger.debug("\nDistribution of term frequencies:")
        logger.debug("Docs per term | Number of terms")
        logger.debug("-" * 30)
        for num_docs in sorted(term_doc_counts.keys()):
            logger.debug(f"{num_docs:12} | {term_doc_counts[num_docs]:15}")
        
        # Calculate average documents per term
        total_postings = sum(len(postings) for postings in self.inverted_index.values())
        avg_docs_per_term = total_postings / len(self.inverted_index) if self.inverted_index else 0
        logger.debug(f"\nAverage documents per term: {avg_docs_per_term:.2f}")
        
        # Show top terms by document frequency
        top_terms = sorted(self.document_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
        
        logger.debug("\nTop 20 terms by document frequency:")
        logger.debug("Term | Doc Frequency")
        logger.debug("-" * 30)
        for term, doc_freq in top_terms:
            logger.debug(f"{term:20} | {doc_freq}")
    
    def _initialize_stats(self) -> Dict[str, int]:
        """Initialize statistics dictionary."""
        return {
            "chunks_indexed": 0,
            "terms_indexed": 0,
            "unique_terms": 0,
        }
    
    def _process_chunks(
        self,
        chunks: List[Chunk],
        stats: Dict[str, int],
        term_freq_cache: Dict[int, Dict[str, int]],
        progress_callback: Optional[Callable[[str, int, int], None]]
    ) -> None:
        """Legacy method for backward compatibility."""
        # Delegate to new component-based method
        self._process_chunks_with_components(chunks, stats, progress_callback)
    
    def _process_chunks_with_components(
        self,
        chunks: List[Chunk],
        stats: Dict[str, int],
        progress_callback: Optional[Callable[[str, int, int], None]]
    ) -> None:
        """Process chunks using specialized components."""
        import time
        
        total_chunks = len(chunks)
        last_progress_time = time.time()
        
        for idx, chunk in enumerate(chunks):
            self._report_chunk_progress(idx, total_chunks, last_progress_time, time.time(), progress_callback)
            
            # Analyze document using term frequency analyzer
            term_frequencies = self.term_analyzer.analyze_document(chunk.content)
            
            # Extract unique terms
            unique_terms = set(term_frequencies.keys())
            
            # Update statistics using statistics calculator
            self.statistics_calculator.update_collection_statistics(
                chunk.id, term_frequencies, unique_terms
            )
            
            # Update inverted index using index builder
            self.index_builder.update_index(chunk.id, term_frequencies)
            
            stats["chunks_indexed"] += 1
            stats["terms_indexed"] += sum(term_frequencies.values())
        
        # Report final tokenizing progress
        if progress_callback:
            progress_callback("bm25_tokenizing", total_chunks, total_chunks)
    
    def _report_chunk_progress(
        self,
        idx: int,
        total_chunks: int,
        last_progress_time: float,
        current_time: float,
        progress_callback: Optional[Callable[[str, int, int], None]]
    ) -> None:
        """Report progress for chunk processing."""
        progress_interval = max(1, total_chunks // 50)
        should_report = (
            current_time - last_progress_time > 3.0
            or idx % progress_interval == 0
            or idx == 0
            or idx == total_chunks - 1
        )
        if progress_callback and should_report:
            progress_callback("bm25_tokenizing", idx + 1, total_chunks)
    
    def _get_cached_term_frequencies(
        self,
        chunk: Chunk,
        term_freq_cache: Dict[int, Dict[str, int]]
    ) -> Dict[str, int]:
        """Legacy method for backward compatibility."""
        return self.term_analyzer.analyze_document(chunk.content)
    
    def _report_post_processing_progress(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]]
    ) -> None:
        """Report progress for post-processing steps."""
        if not progress_callback:
            return
        
        # Building vocabulary step
        vocab_size = self.index_builder.get_vocabulary_size()
        progress_callback("bm25_vocabulary", 0, vocab_size)
        progress_callback("bm25_vocabulary", vocab_size, vocab_size)
        
        # Filtering step
        progress_callback("bm25_filtering", 0, vocab_size)
        progress_callback("bm25_filtering", vocab_size, vocab_size)
        
        # Storing vocabulary step
        self._report_vocabulary_storage_progress(progress_callback, vocab_size)
        
        # Storing inverted index step
        self._report_inverted_index_storage_progress(progress_callback)
        
        # Storing document stats step
        doc_count = len(self.statistics_calculator.document_lengths)
        progress_callback("bm25_store_document_stats", 0, doc_count)
        progress_callback("bm25_store_document_stats", doc_count, doc_count)
    
    def _report_vocabulary_storage_progress(
        self,
        progress_callback: Callable[[str, int, int], None],
        vocab_size: int
    ) -> None:
        """Report vocabulary storage progress."""
        progress_callback("bm25_store_vocabulary", 0, vocab_size)
        for i, term in enumerate(self.inverted_index.keys()):
            if i % 1000 == 0 or i == vocab_size - 1:
                progress_callback("bm25_store_vocabulary", i + 1, vocab_size)
    
    def _report_inverted_index_storage_progress(
        self,
        progress_callback: Callable[[str, int, int], None]
    ) -> None:
        """Report inverted index storage progress."""
        total_entries = sum(len(postings) for postings in self.inverted_index.values())
        progress_callback("bm25_store_inverted_index", 0, total_entries)
        
        entry_count = 0
        for term, postings in self.inverted_index.items():
            entry_count += len(postings)
            if entry_count % 5000 == 0 or entry_count == total_entries:
                progress_callback("bm25_store_inverted_index", entry_count, total_entries)


    def _get_term_positions(self, text: str, term: str) -> List[int]:
        """Get positions of a term in the text.

        Args:
            text: Original text
            term: Term to find

        Returns:
            List of positions where the term appears

        """
        # This is a simplified implementation
        # In practice, you'd want to use the same tokenization as get_term_frequencies
        positions = []
        tokens = self.tokenizer.tokenize(text)
        for i, token in enumerate(tokens):
            if token == term:
                positions.append(i)
        return positions

    def score(
        self,
        query_terms: List[str],
        chunk_id: str,
        chunk_term_freqs: Dict[str, int],
    ) -> float:
        """Calculate BM25 score for a chunk given query terms using statistics calculator.

        Args:
            query_terms: List of query terms
            chunk_id: ID of the chunk to score
            chunk_term_freqs: Term frequencies in the chunk

        Returns:
            BM25 score

        """
        doc_length = self.statistics_calculator.get_document_length(chunk_id)
        if doc_length == 0:
            return 0.0

        score = 0.0
        for term in query_terms:
            if term not in chunk_term_freqs:
                continue

            # Calculate BM25 term score using statistics calculator
            tf = chunk_term_freqs[term]
            term_score = self.statistics_calculator.calculate_bm25_term_score(
                term, tf, doc_length, self.k1, self.b
            )
            score += term_score

        return score

    def clear(self) -> None:
        """Clear all index data and reset statistics using specialized components."""
        self.index_builder.clear()
        self.statistics_calculator.clear()
        self._term_freq_cache.clear()

    def compute_bm25_score(
        self,
        query_terms: List[str],
        chunk_id: str,
        chunk_term_freqs: Dict[str, int],
    ) -> float:
        """Compute BM25 score for a chunk given query terms.

        This is an alias for the score method to maintain backward compatibility.

        Args:
            query_terms: List of query terms
            chunk_id: ID of the chunk to score
            chunk_term_freqs: Term frequencies in the chunk

        Returns:
            BM25 score

        """
        return self.score(query_terms, chunk_id, chunk_term_freqs)

    def get_collection_stats(self) -> Dict[str, object]:
        """Get collection statistics using statistics calculator.

        Returns:
            Dictionary containing collection statistics

        """
        return self.statistics_calculator.get_collection_stats()
