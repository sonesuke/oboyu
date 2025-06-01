"""BM25 statistics calculator for search indexing.

This module provides functionality for calculating BM25-specific statistics
such as IDF scores, document length averages, and collection statistics.
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class BM25StatisticsCalculator:
    """Calculates BM25-specific statistics (IDF, avg doc length, etc.)."""

    def __init__(self) -> None:
        """Initialize the BM25 statistics calculator."""
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.collection_frequencies: Dict[str, int] = defaultdict(int)
        self.document_lengths: Dict[str, int] = {}
        self.document_count = 0
        self.total_document_length = 0

    def calculate_idf_scores(self, term_doc_frequencies: Dict[str, int], total_docs: int) -> Dict[str, float]:
        """Calculate IDF scores for terms.

        Args:
            term_doc_frequencies: Dictionary mapping terms to document frequencies
            total_docs: Total number of documents in the collection

        Returns:
            Dictionary mapping terms to IDF scores

        """
        idf_scores = {}
        
        for term, df in term_doc_frequencies.items():
            if df > 0:
                # IDF calculation: log((N - df + 0.5) / (df + 0.5))
                idf = math.log((total_docs - df + 0.5) / (df + 0.5))
                idf_scores[term] = idf
            else:
                idf_scores[term] = 0.0
        
        return idf_scores

    def calculate_average_document_length(self, document_lengths: List[int]) -> float:
        """Calculate average document length.

        Args:
            document_lengths: List of document lengths

        Returns:
            Average document length

        """
        if not document_lengths:
            return 0.0
        
        return sum(document_lengths) / len(document_lengths)

    def update_collection_statistics(self, chunk_id: str, term_frequencies: Dict[str, int], unique_terms: Set[str]) -> None:
        """Update collection statistics with new document data.

        Args:
            chunk_id: ID of the document chunk
            term_frequencies: Term frequencies for the chunk
            unique_terms: Set of unique terms in the document

        """
        # Update document length
        doc_length = sum(term_frequencies.values())
        self.document_lengths[chunk_id] = doc_length
        self.total_document_length += doc_length
        self.document_count += 1

        # Update document frequencies (one per unique term per document)
        for term in unique_terms:
            self.document_frequencies[term] += 1

        # Update collection frequencies (total term occurrences)
        for term, freq in term_frequencies.items():
            self.collection_frequencies[term] += freq

    def get_document_frequency(self, term: str) -> int:
        """Get document frequency for a term.

        Args:
            term: The term to lookup

        Returns:
            Number of documents containing the term

        """
        return self.document_frequencies.get(term, 0)

    def get_collection_frequency(self, term: str) -> int:
        """Get collection frequency for a term.

        Args:
            term: The term to lookup

        Returns:
            Total number of occurrences of the term in the collection

        """
        return self.collection_frequencies.get(term, 0)

    def get_document_length(self, chunk_id: str) -> int:
        """Get document length for a chunk.

        Args:
            chunk_id: ID of the chunk

        Returns:
            Length of the document in terms

        """
        return self.document_lengths.get(chunk_id, 0)

    def get_average_document_length(self) -> float:
        """Get the current average document length.

        Returns:
            Average document length

        """
        if self.document_count == 0:
            return 0.0
        return self.total_document_length / self.document_count

    def get_collection_stats(self) -> Dict[str, object]:
        """Get comprehensive collection statistics.

        Returns:
            Dictionary containing collection statistics

        """
        avg_doc_length = self.get_average_document_length()
        return {
            "document_count": self.document_count,
            "total_document_length": self.total_document_length,
            "average_document_length": avg_doc_length,
            "avg_document_length": avg_doc_length,  # Alternative key for backward compatibility
            "vocabulary_size": len(self.document_frequencies),
            "total_terms": sum(self.collection_frequencies.values()),
        }

    def calculate_bm25_term_score(
        self,
        term: str,
        term_frequency: int,
        document_length: int,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> float:
        """Calculate BM25 term score for a specific term in a document.

        Args:
            term: The term to score
            term_frequency: Frequency of the term in the document
            document_length: Length of the document
            k1: BM25 k1 parameter (term saturation)
            b: BM25 b parameter (length normalization)

        Returns:
            BM25 term score

        """
        # Get document frequency
        df = self.get_document_frequency(term)
        if df == 0:
            return 0.0

        # Calculate IDF: log((N - df + 0.5) / (df + 0.5))
        idf = math.log((self.document_count - df + 0.5) / (df + 0.5))

        # Calculate average document length
        avg_doc_length = self.get_average_document_length()

        # BM25 term score calculation
        numerator = term_frequency * (k1 + 1)
        denominator = term_frequency + k1 * (1 - b + b * (document_length / avg_doc_length))
        
        return idf * (numerator / denominator)

    def remove_document_statistics(self, chunk_id: str, term_frequencies: Dict[str, int], unique_terms: Set[str]) -> None:
        """Remove statistics for a document from the collection.

        Args:
            chunk_id: ID of the document chunk to remove
            term_frequencies: Term frequencies for the chunk being removed
            unique_terms: Set of unique terms in the document being removed

        """
        if chunk_id not in self.document_lengths:
            return  # Document not in statistics

        # Update document length statistics
        doc_length = self.document_lengths[chunk_id]
        self.total_document_length -= doc_length
        self.document_count -= 1
        del self.document_lengths[chunk_id]

        # Update document frequencies
        for term in unique_terms:
            self.document_frequencies[term] = max(0, self.document_frequencies[term] - 1)
            if self.document_frequencies[term] == 0:
                del self.document_frequencies[term]

        # Update collection frequencies
        for term, freq in term_frequencies.items():
            self.collection_frequencies[term] = max(0, self.collection_frequencies[term] - freq)
            if self.collection_frequencies[term] == 0:
                del self.collection_frequencies[term]

    def clear(self) -> None:
        """Clear all statistics."""
        self.document_frequencies.clear()
        self.collection_frequencies.clear()
        self.document_lengths.clear()
        self.document_count = 0
        self.total_document_length = 0

