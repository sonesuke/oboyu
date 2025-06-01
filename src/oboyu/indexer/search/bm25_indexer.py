"""BM25 indexer for Oboyu.

This module provides BM25 indexing functionality for Japanese text search,
including inverted index construction and statistical information management.
"""

import logging
import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

from oboyu.indexer.core.document_processor import Chunk
from oboyu.indexer.models.tokenizer_service import create_tokenizer

logger = logging.getLogger(__name__)


class BM25Indexer:
    """BM25 indexer for building inverted index and computing statistics.

    This class handles the construction of inverted index, document frequency
    calculation, and other statistics required for BM25 scoring.
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer_class: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, object]] = None,
        use_stopwords: bool = False,
        min_doc_frequency: int = 1,
        store_positions: bool = True,
    ) -> None:
        """Initialize BM25 indexer.

        Args:
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
        # Index structures
        self.inverted_index: Dict[str, List[Tuple[str, int, Optional[List[int]]]]] = defaultdict(list)
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.collection_frequencies: Dict[str, int] = defaultdict(int)
        self.document_lengths: Dict[str, int] = {}
        self.document_count = 0
        self.total_document_length = 0

    def index_chunks(self, chunks: List[Chunk], progress_callback: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, int]:
        """Index a batch of chunks for BM25 search.

        Args:
            chunks: List of chunks to index
            progress_callback: Optional callback for progress updates

        Returns:
            Statistics about the indexing operation

        """
        stats = self._initialize_stats()
        term_freq_cache: Dict[int, Dict[str, int]] = {}
        
        # Process chunks
        self._process_chunks(chunks, stats, term_freq_cache, progress_callback)
        
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
        """Process chunks and update indices."""
        import time
        
        total_chunks = len(chunks)
        last_progress_time = time.time()
        
        for idx, chunk in enumerate(chunks):
            self._report_chunk_progress(idx, total_chunks, last_progress_time, time.time(), progress_callback)
            
            # Get term frequencies with caching
            term_frequencies = self._get_cached_term_frequencies(chunk, term_freq_cache)
            
            # Update document statistics
            self._update_document_stats(chunk, term_frequencies)
            
            # Update inverted index
            unique_terms = self._update_inverted_index(chunk, term_frequencies, stats)
            
            # Update document frequencies
            self._update_document_frequencies(unique_terms)
            
            self.document_count += 1
            stats["chunks_indexed"] += 1
        
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
        """Get term frequencies with caching."""
        content_hash = hash(chunk.content)
        if content_hash in term_freq_cache:
            return term_freq_cache[content_hash]
        
        term_frequencies = self.tokenizer.get_term_frequencies(chunk.content)
        if len(term_freq_cache) < 1000:  # Limit cache size
            term_freq_cache[content_hash] = term_frequencies
        return term_frequencies
    
    def _update_document_stats(self, chunk: Chunk, term_frequencies: Dict[str, int]) -> None:
        """Update document length statistics."""
        doc_length = sum(term_frequencies.values())
        self.document_lengths[chunk.id] = doc_length
        self.total_document_length += doc_length
    
    def _update_inverted_index(
        self,
        chunk: Chunk,
        term_frequencies: Dict[str, int],
        stats: Dict[str, int]
    ) -> Set[str]:
        """Update inverted index and return unique terms."""
        unique_terms_in_doc: Set[str] = set()
        
        for term, freq in term_frequencies.items():
            positions: Optional[List[int]] = [] if self.store_positions else None
            self.inverted_index[term].append((chunk.id, freq, positions))
            self.collection_frequencies[term] += freq
            unique_terms_in_doc.add(term)
            stats["terms_indexed"] += freq
        
        return unique_terms_in_doc
    
    def _update_document_frequencies(self, unique_terms: Set[str]) -> None:
        """Update document frequencies for unique terms."""
        for term in unique_terms:
            self.document_frequencies[term] += 1
    
    def _report_post_processing_progress(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]]
    ) -> None:
        """Report progress for post-processing steps."""
        if not progress_callback:
            return
        
        # Building vocabulary step
        vocab_size = len(self.inverted_index)
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
        doc_count = len(self.document_lengths)
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
        """Calculate BM25 score for a chunk given query terms.

        Args:
            query_terms: List of query terms
            chunk_id: ID of the chunk to score
            chunk_term_freqs: Term frequencies in the chunk

        Returns:
            BM25 score

        """
        if chunk_id not in self.document_lengths:
            return 0.0

        score = 0.0
        doc_length = self.document_lengths[chunk_id]
        avg_doc_length = self.total_document_length / self.document_count if self.document_count > 0 else 1.0

        for term in query_terms:
            if term not in chunk_term_freqs:
                continue

            # Term frequency in the chunk
            tf = chunk_term_freqs[term]

            # Document frequency
            df = self.document_frequencies.get(term, 0)
            if df == 0:
                continue

            # IDF calculation: log((N - df + 0.5) / (df + 0.5))
            idf = math.log((self.document_count - df + 0.5) / (df + 0.5))

            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
            term_score = idf * (numerator / denominator)

            score += term_score

        return score

    def clear(self) -> None:
        """Clear all index data and reset statistics."""
        self.inverted_index.clear()
        self.document_frequencies.clear()
        self.collection_frequencies.clear()
        self.document_lengths.clear()
        self.document_count = 0
        self.total_document_length = 0

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
        """Get collection statistics.

        Returns:
            Dictionary containing collection statistics

        """
        avg_doc_length = self.total_document_length / self.document_count if self.document_count > 0 else 0.0
        return {
            "document_count": self.document_count,
            "total_document_length": self.total_document_length,
            "average_document_length": avg_doc_length,
            "avg_document_length": avg_doc_length,  # Alternative key name for backward compatibility
            "vocabulary_size": len(self.inverted_index),
            "total_terms": sum(self.collection_frequencies.values()),
        }
