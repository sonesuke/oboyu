"""In-memory implementation of SearchRepository for fast testing."""

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set

from src.oboyu.domain.entities.chunk import Chunk
from src.oboyu.domain.entities.query import Query
from src.oboyu.domain.entities.search_result import SearchResult
from src.oboyu.domain.value_objects.chunk_id import ChunkId
from src.oboyu.domain.value_objects.embedding_vector import EmbeddingVector
from src.oboyu.domain.value_objects.score import Score
from src.oboyu.ports.repositories.search_repository import SearchRepository


class InMemorySearchRepository(SearchRepository):
    """In-memory implementation of SearchRepository for testing.
    
    This implementation stores all data in memory using dictionaries and lists,
    providing fast test execution without database dependencies.
    """
    
    def __init__(self):
        """Initialize empty in-memory storage."""
        self._chunks: Dict[ChunkId, Chunk] = {}
        self._embeddings: Dict[ChunkId, EmbeddingVector] = {}
        self._inverted_index: Dict[str, Set[ChunkId]] = defaultdict(set)
        self._term_frequencies: Dict[ChunkId, Dict[str, int]] = defaultdict(dict)
        self._document_lengths: Dict[ChunkId, int] = {}
    
    async def store_chunk(self, chunk: Chunk) -> None:
        """Store a document chunk in memory."""
        self._chunks[chunk.id] = chunk
        self._build_inverted_index_for_chunk(chunk)
    
    async def store_chunks(self, chunks: List[Chunk]) -> None:
        """Store multiple document chunks in memory."""
        for chunk in chunks:
            await self.store_chunk(chunk)
    
    async def store_embedding(self, chunk_id: ChunkId, embedding: EmbeddingVector) -> None:
        """Store an embedding vector for a chunk."""
        self._embeddings[chunk_id] = embedding
    
    async def store_embeddings(self, embeddings: List[tuple[ChunkId, EmbeddingVector]]) -> None:
        """Store multiple embedding vectors."""
        for chunk_id, embedding in embeddings:
            await self.store_embedding(chunk_id, embedding)
    
    async def find_by_vector_similarity(
        self, query_vector: EmbeddingVector, top_k: int, threshold: float = 0.0
    ) -> List[SearchResult]:
        """Find chunks by vector similarity using cosine similarity."""
        if not self._embeddings:
            return []
        
        # Calculate similarities for all embeddings
        similarities = []
        for chunk_id, embedding in self._embeddings.items():
            if chunk_id not in self._chunks:
                continue
            
            try:
                similarity = query_vector.cosine_similarity(embedding)
                if similarity >= threshold:
                    similarities.append((chunk_id, similarity))
            except ValueError:
                # Skip embeddings with dimension mismatch
                continue
        
        # Sort by similarity (descending) and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[:top_k]
        
        # Convert to SearchResult objects
        results = []
        for chunk_id, similarity in similarities:
            chunk = self._chunks[chunk_id]
            result = SearchResult(
                chunk_id=chunk.id,
                document_path=chunk.document_path,
                title=chunk.title,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                language=chunk.language,
                score=Score(similarity),
                metadata=chunk.metadata.copy(),
                created_at=chunk.created_at
            )
            results.append(result)
        
        return results
    
    async def find_by_bm25(self, query: Query) -> List[SearchResult]:
        """Find chunks using BM25 algorithm."""
        if not query.text.strip():
            return []
        
        query_terms = self._tokenize(query.text.lower())
        if not query_terms:
            return []
        
        # Calculate BM25 scores
        scores = self._calculate_bm25_scores(query_terms)
        
        # Sort by score (descending) and take top_k
        # Include all results with non-zero scores (even negative ones for single document cases)
        scored_chunks = [(chunk_id, score) for chunk_id, score in scores.items() if score != 0]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        scored_chunks = scored_chunks[:query.top_k]
        
        # Convert to SearchResult objects
        results = []
        for chunk_id, bm25_score in scored_chunks:
            chunk = self._chunks[chunk_id]
            # Use Score.create to normalize negative scores to 0
            result = SearchResult(
                chunk_id=chunk.id,
                document_path=chunk.document_path,
                title=chunk.title,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                language=chunk.language,
                score=Score.create(bm25_score),
                metadata=chunk.metadata.copy(),
                created_at=chunk.created_at
            )
            results.append(result)
        
        return results
    
    async def find_by_chunk_id(self, chunk_id: ChunkId) -> Optional[Chunk]:
        """Find a specific chunk by ID."""
        return self._chunks.get(chunk_id)
    
    async def delete_chunk(self, chunk_id: ChunkId) -> None:
        """Delete a chunk and its associated data."""
        if chunk_id in self._chunks:
            # Remove from chunks
            chunk = self._chunks.pop(chunk_id)
            
            # Remove from embeddings
            self._embeddings.pop(chunk_id, None)
            
            # Remove from inverted index
            self._remove_from_inverted_index(chunk)
            
            # Remove from term frequencies and document lengths
            self._term_frequencies.pop(chunk_id, None)
            self._document_lengths.pop(chunk_id, None)
    
    async def delete_chunks_by_document(self, document_path: str) -> None:
        """Delete all chunks for a specific document."""
        # Find all chunks for the document
        chunks_to_delete = [
            chunk_id for chunk_id, chunk in self._chunks.items()
            if str(chunk.document_path) == document_path
        ]
        
        # Delete each chunk
        for chunk_id in chunks_to_delete:
            await self.delete_chunk(chunk_id)
    
    async def get_chunk_count(self) -> int:
        """Get total number of chunks in repository."""
        return len(self._chunks)
    
    async def get_embedding_count(self) -> int:
        """Get total number of embeddings in repository."""
        return len(self._embeddings)
    
    async def chunk_exists(self, chunk_id: ChunkId) -> bool:
        """Check if a chunk exists in the repository."""
        return chunk_id in self._chunks
    
    # Private helper methods for BM25 implementation
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and punctuation."""
        # Remove punctuation and convert to lowercase
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split on whitespace and filter empty strings
        tokens = [token.strip() for token in cleaned_text.split() if token.strip()]
        return tokens
    
    def _build_inverted_index_for_chunk(self, chunk: Chunk) -> None:
        """Build inverted index entries for a chunk."""
        # Tokenize content and title
        content_tokens = self._tokenize(chunk.content)
        title_tokens = self._tokenize(chunk.title)
        all_tokens = content_tokens + title_tokens
        
        # Count term frequencies
        term_freq = defaultdict(int)
        for token in all_tokens:
            term_freq[token] += 1
            self._inverted_index[token].add(chunk.id)
        
        self._term_frequencies[chunk.id] = dict(term_freq)
        self._document_lengths[chunk.id] = len(all_tokens)
    
    def _remove_from_inverted_index(self, chunk: Chunk) -> None:
        """Remove chunk from inverted index."""
        # Get tokens for this chunk
        if chunk.id in self._term_frequencies:
            for token in self._term_frequencies[chunk.id]:
                self._inverted_index[token].discard(chunk.id)
                # Clean up empty sets
                if not self._inverted_index[token]:
                    del self._inverted_index[token]
    
    def _calculate_bm25_scores(self, query_terms: List[str]) -> Dict[ChunkId, float]:
        """Calculate BM25 scores for query terms."""
        # BM25 parameters
        k1 = 1.2
        b = 0.75
        
        # Calculate average document length
        if not self._document_lengths:
            return {}
        
        avg_doc_length = sum(self._document_lengths.values()) / len(self._document_lengths)
        total_docs = len(self._chunks)
        
        scores = defaultdict(float)
        
        for term in set(query_terms):  # Use set to avoid duplicate calculations
            # Get documents containing the term
            docs_with_term = self._inverted_index.get(term, set())
            if not docs_with_term:
                continue
            
            # Calculate IDF (Inverse Document Frequency)
            df = len(docs_with_term)  # Document frequency
            idf = math.log((total_docs - df + 0.5) / (df + 0.5))
            
            # Calculate term frequency component for each document
            for chunk_id in docs_with_term:
                if chunk_id not in self._chunks:
                    continue
                
                tf = self._term_frequencies[chunk_id].get(term, 0)
                doc_length = self._document_lengths[chunk_id]
                
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                
                term_score = idf * (numerator / denominator)
                scores[chunk_id] += term_score
        
        return scores
    
    # Utility methods for testing and debugging
    
    def clear(self) -> None:
        """Clear all stored data (useful for test cleanup)."""
        self._chunks.clear()
        self._embeddings.clear()
        self._inverted_index.clear()
        self._term_frequencies.clear()
        self._document_lengths.clear()
    
    def get_indexed_terms(self) -> Set[str]:
        """Get all terms in the inverted index (useful for testing)."""
        return set(self._inverted_index.keys())
    
    def get_term_frequency(self, chunk_id: ChunkId, term: str) -> int:
        """Get term frequency for a specific chunk (useful for testing)."""
        return self._term_frequencies.get(chunk_id, {}).get(term, 0)
