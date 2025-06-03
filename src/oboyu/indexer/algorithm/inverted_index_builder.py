"""Inverted index builder for BM25 search.

This module provides functionality for building and managing inverted index structures.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from oboyu.common.types import Chunk

logger = logging.getLogger(__name__)


class InvertedIndexBuilder:
    """Responsible for building and managing the inverted index structure."""

    def __init__(self, store_positions: bool = True) -> None:
        """Initialize the inverted index builder.

        Args:
            store_positions: Whether to store term positions in the index

        """
        self.store_positions = store_positions
        self.inverted_index: Dict[str, List[Tuple[str, int, Optional[List[int]]]]] = defaultdict(list)

    def build_index(self, chunks: List[Chunk], term_frequencies: Dict[str, Dict[str, int]]) -> Dict[str, int]:
        """Build inverted index from chunks and their term frequencies.

        Args:
            chunks: List of chunks to index
            term_frequencies: Pre-computed term frequencies for each chunk

        Returns:
            Dictionary containing index statistics

        """
        stats = {"chunks_indexed": 0, "terms_indexed": 0}
        
        for chunk in chunks:
            chunk_term_freqs = term_frequencies.get(chunk.id, {})
            self.update_index(chunk.id, chunk_term_freqs)
            
            stats["chunks_indexed"] += 1
            stats["terms_indexed"] += sum(chunk_term_freqs.values())
        
        return stats

    def update_index(self, chunk_id: str, term_frequencies: Dict[str, int]) -> Set[str]:
        """Update inverted index with terms from a document.

        Args:
            chunk_id: ID of the chunk
            term_frequencies: Term frequencies for the chunk

        Returns:
            Set of unique terms in the document

        """
        unique_terms: Set[str] = set()
        
        for term, freq in term_frequencies.items():
            positions: Optional[List[int]] = [] if self.store_positions else None
            self.inverted_index[term].append((chunk_id, freq, positions))
            unique_terms.add(term)
        
        return unique_terms

    def remove_from_index(self, chunk_id: str) -> None:
        """Remove a chunk from the inverted index.

        Args:
            chunk_id: ID of the chunk to remove

        """
        # Find and remove all entries for this chunk
        terms_to_remove = []
        
        for term, postings in self.inverted_index.items():
            # Filter out postings for the specified chunk
            updated_postings = [(doc_id, freq, positions) for doc_id, freq, positions in postings
                              if doc_id != chunk_id]
            
            if updated_postings:
                self.inverted_index[term] = updated_postings
            else:
                # Mark term for removal if no documents remain
                terms_to_remove.append(term)
        
        # Remove terms that have no documents
        for term in terms_to_remove:
            del self.inverted_index[term]

    def get_term_postings(self, term: str) -> List[Tuple[str, int, Optional[List[int]]]]:
        """Get postings list for a specific term.

        Args:
            term: The term to lookup

        Returns:
            List of (chunk_id, frequency, positions) tuples

        """
        return self.inverted_index.get(term, [])

    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary (number of unique terms).

        Returns:
            Number of unique terms in the index

        """
        return len(self.inverted_index)

    def get_all_terms(self) -> Set[str]:
        """Get all terms in the vocabulary.

        Returns:
            Set of all terms in the index

        """
        return set(self.inverted_index.keys())

    def clear(self) -> None:
        """Clear the inverted index."""
        self.inverted_index.clear()

    def get_index_data(self) -> Dict[str, List[Tuple[str, int, Optional[List[int]]]]]:
        """Get the raw inverted index data.

        Returns:
            The inverted index dictionary

        """
        return dict(self.inverted_index)

