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
    
    This indexer provides:
    - Inverted index construction
    - Term frequency and document frequency calculation
    - Collection statistics management
    - Batch processing for efficient indexing
    """
    
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        min_token_length: int = 2,
        use_japanese_tokenizer: bool = True,
    ) -> None:
        """Initialize the BM25 indexer.
        
        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (document length normalization)
            min_token_length: Minimum token length to index
            use_japanese_tokenizer: Whether to use Japanese-specific tokenizer

        """
        self.k1 = k1
        self.b = b
        self.min_token_length = min_token_length
        self.use_japanese_tokenizer = use_japanese_tokenizer
        
        # Create tokenizer
        self.tokenizer = create_tokenizer(
            language="ja" if use_japanese_tokenizer else "en",
            min_token_length=min_token_length,
        )
        
        # Initialize index structures
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
                # Get token positions (if available)
                positions = self._get_term_positions(chunk.content, term)
                
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
        return stats
    
    def compute_bm25_score(
        self,
        query_terms: List[str],
        chunk_id: str,
        term_frequencies: Dict[str, int],
    ) -> float:
        """Compute BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query terms
            chunk_id: ID of the chunk to score
            term_frequencies: Term frequencies in the chunk
            
        Returns:
            BM25 score

        """
        if chunk_id not in self.document_lengths:
            return 0.0
        
        doc_length = self.document_lengths[chunk_id]
        avg_doc_length = self.total_document_length / max(self.document_count, 1)
        
        score = 0.0
        
        for term in query_terms:
            if term not in term_frequencies:
                continue
            
            # Get term frequency in document
            tf = term_frequencies[term]
            
            # Get document frequency
            df = self.document_frequencies.get(term, 0)
            
            # Compute IDF component
            # Adding 0.5 to avoid negative IDF for terms that appear in more than half the documents
            idf = math.log((self.document_count - df + 0.5) / (df + 0.5) + 1.0)
            
            # Compute normalized term frequency component
            norm_tf = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
            )
            
            # Add to total score
            score += idf * norm_tf
        
        return score
    
    def search(
        self,
        query: str,
        limit: int = 10,
        filter_chunk_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Search the BM25 index for matching documents.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            filter_chunk_ids: Optional set of chunk IDs to search within
            
        Returns:
            List of (chunk_id, score) tuples sorted by score

        """
        # Tokenize query
        query_terms = self.tokenizer.tokenize(query)
        
        if not query_terms:
            return []
        
        # Find candidate documents
        candidate_chunks: Set[str] = set()
        chunk_term_frequencies: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        for term in query_terms:
            if term in self.inverted_index:
                for chunk_id, freq, _ in self.inverted_index[term]:
                    # Apply filter if provided
                    if filter_chunk_ids and chunk_id not in filter_chunk_ids:
                        continue
                    
                    candidate_chunks.add(chunk_id)
                    chunk_term_frequencies[chunk_id][term] = freq
        
        # Score candidate documents
        scored_chunks = []
        for chunk_id in candidate_chunks:
            score = self.compute_bm25_score(
                query_terms,
                chunk_id,
                chunk_term_frequencies[chunk_id],
            )
            scored_chunks.append((chunk_id, score))
        
        # Sort by score (descending) and return top results
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:limit]
    
    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary.
        
        Returns:
            Number of unique terms in the index

        """
        return len(self.inverted_index)
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection statistics

        """
        return {
            "document_count": self.document_count,
            "vocabulary_size": self.get_vocabulary_size(),
            "total_terms": sum(self.collection_frequencies.values()),
            "avg_document_length": int(self.total_document_length / max(self.document_count, 1)),
        }
    
    def _get_term_positions(self, text: str, term: str) -> List[int]:
        """Get positions of a term in text.
        
        This is a simplified implementation that returns empty list.
        Position tracking can be added if needed for phrase queries.
        
        Args:
            text: Document text
            term: Term to find positions for
            
        Returns:
            List of positions (currently empty)

        """
        # TODO: Implement position tracking if needed for phrase queries
        return []
    
    def clear(self) -> None:
        """Clear all index data."""
        self.inverted_index.clear()
        self.document_frequencies.clear()
        self.collection_frequencies.clear()
        self.document_lengths.clear()
        self.document_count = 0
        self.total_document_length = 0
