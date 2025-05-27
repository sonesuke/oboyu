"""BM25 indexer for Oboyu.

This module provides BM25 indexing functionality for Japanese text search,
including inverted index construction and statistical information management.
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from oboyu.indexer.processor import Chunk
from oboyu.indexer.tokenizer import create_tokenizer

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
        self.tokenizer = create_tokenizer(
            language=language,
            min_token_length=min_token_length,
            use_stopwords=use_stopwords
        )
        
        # Index structures
        self.inverted_index: Dict[str, List[Tuple[str, int, List[int]]]] = defaultdict(list)
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.collection_frequencies: Dict[str, int] = defaultdict(int)
        self.document_lengths: Dict[str, int] = {}
        self.document_count = 0
        self.total_document_length = 0
        
    def index_chunks(self, chunks: List[Chunk]) -> Dict[str, int]:
        """Index a batch of chunks for BM25 search.
        
        Args:
            chunks: List of chunks to index
            
        Returns:
            Statistics about the indexing operation

        """
        stats = {
            "chunks_indexed": 0,
            "terms_indexed": 0,
            "unique_terms": 0,
        }
        
        # Debug: Track term distribution
        term_doc_counts: Dict[int, int] = defaultdict(int)  # num_docs -> count of terms
        
        for chunk in chunks:
            # Get term frequencies for the chunk
            term_frequencies = self.tokenizer.get_term_frequencies(chunk.content)
            
            # Update document length
            doc_length = sum(term_frequencies.values())
            self.document_lengths[chunk.id] = doc_length
            self.total_document_length += doc_length
            
            # Track unique terms for this document
            unique_terms_in_doc: Set[str] = set()
            
            # Update inverted index
            for term, freq in term_frequencies.items():
                # Get token positions only if enabled
                positions = self._get_term_positions(chunk.content, term) if self.store_positions else []
                
                # Add to inverted index
                self.inverted_index[term].append((chunk.id, freq, positions))
                
                # Update collection frequency
                self.collection_frequencies[term] += freq
                
                # Track unique terms
                unique_terms_in_doc.add(term)
                stats["terms_indexed"] += freq
            
            # Update document frequencies
            for term in unique_terms_in_doc:
                self.document_frequencies[term] += 1
            
            self.document_count += 1
            stats["chunks_indexed"] += 1
        
        stats["unique_terms"] = len(self.inverted_index)
        
        # Track filtered terms
        if hasattr(self.tokenizer, 'filtered_by_stopwords'):
            self.tokenizer.filtered_by_stopwords = defaultdict(int)
        
        # Debug: Analyze inverted index structure (only in debug mode)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("=== BM25 Inverted Index Analysis ===")
            logger.debug(f"Total unique terms (vocabulary size): {len(self.inverted_index)}")
            logger.debug(f"Total chunks indexed: {self.document_count}")
            
            # Calculate distribution of documents per term
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
            top_terms = sorted(
                self.document_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            logger.debug("\nTop 20 terms by document frequency:")
            logger.debug("Term | Doc Frequency")
            logger.debug("-" * 30)
            for term, doc_freq in top_terms:
                logger.debug(f"{term:20} | {doc_freq}")
        
        return stats
    
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
        avg_doc_length = (
            self.total_document_length / self.document_count
            if self.document_count > 0 else 0.0
        )
        return {
            "document_count": self.document_count,
            "total_document_length": self.total_document_length,
            "average_document_length": avg_doc_length,
            "avg_document_length": avg_doc_length,  # Alternative key name for backward compatibility
            "vocabulary_size": len(self.inverted_index),
            "total_terms": sum(self.collection_frequencies.values()),
        }
