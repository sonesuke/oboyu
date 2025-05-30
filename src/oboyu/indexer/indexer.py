"""Main indexer implementation for Oboyu.

This module provides the Indexer class for processing, embedding, and storing
document chunks with specialized handling for Japanese content.
"""

import concurrent.futures
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from oboyu.crawler.crawler import CrawlerResult
from oboyu.indexer.bm25_indexer import BM25Indexer
from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.database import Database
from oboyu.indexer.embedding import EmbeddingGenerator
from oboyu.indexer.processor import Chunk, DocumentProcessor
from oboyu.indexer.reranker import BaseReranker, create_reranker
from oboyu.indexer.tokenizer import create_tokenizer


@dataclass
class SearchResult:
    """Result of a vector search operation."""

    chunk_id: str
    """ID of the matching chunk."""

    path: str
    """Path to the source document."""

    title: str
    """Title of the document or chunk."""

    content: str
    """Chunk text content."""

    chunk_index: int
    """Position of this chunk in the original document."""

    language: str
    """Language code of the content."""

    metadata: Dict[str, object]
    """Additional metadata about the chunk."""

    score: float
    """Similarity score (0-1, where 1 is perfect match)."""


class Indexer:
    """Document indexer for processing, embedding, and storing document chunks."""

    def __init__(
        self,
        config: Optional[IndexerConfig] = None,
        processor: Optional[DocumentProcessor] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        database: Optional[Database] = None,
        reranker: Optional[BaseReranker] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config: Indexer configuration
            processor: Document processor for chunking
            embedding_generator: Generator for document embeddings
            database: Database for storing chunks and embeddings
            reranker: Reranker for improving search results
            bm25_indexer: BM25 indexer for keyword search

        """
        # Initialize configuration
        self.config = config or IndexerConfig()

        # Initialize components with configuration
        self.processor = processor or DocumentProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            document_prefix=self.config.document_prefix,
        )

        self.embedding_generator = embedding_generator or EmbeddingGenerator(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
            batch_size=self.config.batch_size,
            max_seq_length=self.config.max_seq_length,
            query_prefix=self.config.query_prefix,
            use_onnx=self.config.use_onnx,
            onnx_quantization_config=self.config.onnx_quantization_config if self.config.use_onnx else None,
            onnx_optimization_level=self.config.onnx_optimization_level if self.config.use_onnx else "none",
        )

        # Dimensions can't be None here, but mypy doesn't know that
        embedding_dims = self.embedding_generator.dimensions
        if embedding_dims is None:
            # This should never happen, but we need this for type safety
            embedding_dims = 256  # Default for Ruri v3-30m model

        self.database = database or Database(
            db_path=self.config.db_path,
            embedding_dimensions=embedding_dims,
            ef_construction=self.config.ef_construction,
            ef_search=self.config.ef_search,
            m=self.config.m,
            m0=self.config.m0,
        )

        # Set up the database schema
        self.database.setup()

        # Initialize BM25 indexer with optimizations
        self.bm25_indexer = bm25_indexer or BM25Indexer(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
            tokenizer_kwargs={
                "min_token_length": self.config.bm25_min_token_length,
            },
            use_stopwords=True,  # Enable stopword filtering
            min_doc_frequency=2,  # Filter terms appearing in less than 2 documents
            store_positions=True,  # Enable position storage for phrase search
        )

        # Initialize reranker if enabled
        self.reranker = reranker
        if self.reranker is None and self.config.use_reranker:
            self.reranker = create_reranker(
                model_name=self.config.reranker_model,
                use_onnx=self.config.reranker_use_onnx,
                device=self.config.reranker_device,
                batch_size=self.config.reranker_batch_size,
                max_length=self.config.reranker_max_length,
                quantization_config=self.config.onnx_quantization_config if self.config.reranker_use_onnx else None,
                optimization_level=self.config.onnx_optimization_level if self.config.reranker_use_onnx else "none",
            )

        # Keep track of processed files
        self._processed_files: Set[Path] = set()

        # Create a ThreadPoolExecutor that we'll reuse
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers)

    def index_documents(self, crawler_results: List[CrawlerResult], progress_callback: Optional[Callable[[str, int, int], None]] = None) -> int:
        """Process and index documents from crawler results.

        Args:
            crawler_results: List of crawler results to index
            progress_callback: Optional callback for progress updates
                              (stage, current, total)

        Returns:
            Number of chunks indexed

        """
        # Skip already processed files
        new_docs = self._filter_processed_documents(crawler_results)
        if not new_docs:
            return 0

        # Process documents into chunks
        all_chunks = self._process_documents_with_progress(new_docs, progress_callback)
        if not all_chunks:
            return 0

        # Store chunks and report progress
        self._store_chunks_with_progress(all_chunks, progress_callback)

        # Build BM25 index
        self._build_bm25_index_with_progress(all_chunks, progress_callback)

        # Generate and store embeddings with progress reporting
        embeddings = self._generate_embeddings_with_progress(all_chunks, progress_callback)

        # Store embeddings and report progress
        self._store_embeddings_with_progress(embeddings, progress_callback)

        # Recompact index if needed
        self._recompact_index_if_needed(all_chunks, progress_callback)

        return len(all_chunks)

    def _filter_processed_documents(self, crawler_results: List[CrawlerResult]) -> List[CrawlerResult]:
        """Filter out already processed documents.

        Args:
            crawler_results: List of crawler results to filter

        Returns:
            List of unprocessed documents

        """
        new_docs = []
        for result in crawler_results:
            if result.path not in self._processed_files:
                new_docs.append(result)
                self._processed_files.add(result.path)
        return new_docs

    def _process_documents_with_progress(self, documents: List[CrawlerResult], progress_callback: Optional[Callable[[str, int, int], None]]) -> List[Chunk]:
        """Process documents into chunks with progress tracking.

        Args:
            documents: Documents to process
            progress_callback: Progress callback function

        Returns:
            List of document chunks

        """
        # Report starting document processing
        if progress_callback:
            progress_callback("processing", 0, len(documents))

        # Process documents into chunks
        all_chunks: List[Chunk] = []
        processed_count = 0

        # Use the shared ThreadPoolExecutor for parallel processing
        future_to_doc = {self._executor.submit(self._process_document, doc): doc for doc in documents}

        # Collect chunks as they complete
        for future in concurrent.futures.as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                chunks = future.result()
                if chunks:
                    all_chunks.extend(chunks)

                # Update progress
                processed_count += 1
                if progress_callback:
                    progress_callback("processing", processed_count, len(documents))
            except Exception as e:
                logging.error(f"Error processing {doc.path}: {e}")
                # Still count as processed for progress tracking
                processed_count += 1
                if progress_callback:
                    progress_callback("processing_error", processed_count, len(documents))

        return all_chunks

    def _store_chunks_with_progress(self, chunks: List[Chunk], progress_callback: Optional[Callable[[str, int, int], None]]) -> None:
        """Store chunks with progress reporting.

        Args:
            chunks: Chunks to store
            progress_callback: Progress callback function

        """
        # Report storing chunks
        if progress_callback:
            progress_callback("storing", 0, 1)

        # Store chunks in database
        self.database.store_chunks(chunks)

        # Report chunk storage complete
        if progress_callback:
            progress_callback("storing", 1, 1)

    def _generate_embeddings_with_progress(
        self, chunks: List[Chunk], progress_callback: Optional[Callable[[str, int, int], None]]
    ) -> List[Tuple[str, str, NDArray[np.float32], datetime]]:
        """Generate embeddings with progress tracking.

        Args:
            chunks: Chunks to generate embeddings for
            progress_callback: Progress callback function

        Returns:
            List of embeddings

        """
        # Report starting embedding generation
        if progress_callback:
            progress_callback("embedding", 0, len(chunks))

        # Create embedding progress callback
        def embedding_progress(current: int, total: int, status: str) -> None:
            if progress_callback:
                progress_callback("embedding", current, total)

        # Generate embeddings with progress tracking
        return self.embedding_generator.generate_embeddings(chunks, progress_callback=embedding_progress)

    def _store_embeddings_with_progress(
        self, embeddings: List[Tuple[str, str, NDArray[np.float32], datetime]], progress_callback: Optional[Callable[[str, int, int], None]]
    ) -> None:
        """Store embeddings with progress reporting.

        Args:
            embeddings: Embeddings to store
            progress_callback: Progress callback function

        """
        # Report storing embeddings
        if progress_callback:
            progress_callback("storing_embeddings", 0, 1)

        # Store embeddings
        self.database.store_embeddings(embeddings, self.config.embedding_model)

        # Report embeddings stored
        if progress_callback:
            progress_callback("storing_embeddings", 1, 1)

    def _recompact_index_if_needed(self, chunks: List[Chunk], progress_callback: Optional[Callable[[str, int, int], None]]) -> None:
        """Recompact index if needed with progress reporting.

        Args:
            chunks: Chunks that were added
            progress_callback: Progress callback function

        """
        # Recompact index periodically
        if len(chunks) > 100:  # Only recompact after significant additions
            # Report recompacting
            if progress_callback:
                progress_callback("recompacting", 0, 1)

            self.database.recompact_index()

            # Report recompacting complete
            if progress_callback:
                progress_callback("recompacting", 1, 1)

    def _build_bm25_index_with_progress(self, chunks: List[Chunk], progress_callback: Optional[Callable[[str, int, int], None]]) -> None:
        """Build BM25 index with progress tracking.

        Args:
            chunks: Chunks to index for BM25
            progress_callback: Progress callback function

        """
        # Report starting BM25 indexing
        if progress_callback:
            progress_callback("bm25_indexing", 0, 5)

        # Define BM25 progress callback
        def bm25_progress(current: int, total: int) -> None:
            if progress_callback:
                # Show tokenization progress as step 1
                progress_callback("bm25_tokenizing", current, total)

        # Index chunks for BM25 with progress tracking
        self.bm25_indexer.index_chunks(chunks, progress_callback=bm25_progress)

        # Report BM25 chunks indexed
        if progress_callback:
            progress_callback("bm25_indexing", 1, 5)

        # Prepare data for database storage with minimum frequency filtering
        vocabulary = {}
        filtered_terms = set()
        min_doc_freq = self.bm25_indexer.min_doc_frequency

        for term, doc_freq in self.bm25_indexer.document_frequencies.items():
            if doc_freq >= min_doc_freq:
                coll_freq = self.bm25_indexer.collection_frequencies[term]
                vocabulary[term] = (doc_freq, coll_freq)
            else:
                filtered_terms.add(term)

        # Prepare document statistics
        document_stats = {}
        # First, build a reverse index for efficient lookup
        doc_unique_terms = {}
        for term, postings in self.bm25_indexer.inverted_index.items():
            for chunk_id, _, _ in postings:
                if chunk_id not in doc_unique_terms:
                    doc_unique_terms[chunk_id] = 0
                doc_unique_terms[chunk_id] += 1

        for chunk_id, doc_length in self.bm25_indexer.document_lengths.items():
            # Calculate statistics
            total_terms = doc_length
            unique_terms = doc_unique_terms.get(chunk_id, 0)
            avg_term_freq = total_terms / max(unique_terms, 1)
            document_stats[chunk_id] = (total_terms, unique_terms, avg_term_freq)

        # Prepare collection statistics
        collection_stats = {
            "total_documents": self.bm25_indexer.document_count,
            "total_terms": self.bm25_indexer.total_document_length,
            "avg_document_length": self.bm25_indexer.total_document_length / max(self.bm25_indexer.document_count, 1),
        }

        # Filter inverted index to exclude low-frequency terms
        filtered_inverted_index = {}
        for term, postings in self.bm25_indexer.inverted_index.items():
            if term not in filtered_terms:
                filtered_inverted_index[term] = postings

        # Report vocabulary building
        if progress_callback:
            progress_callback("bm25_indexing", 2, 5)

        # Report filtering low-frequency terms
        if progress_callback:
            progress_callback("bm25_indexing", 3, 5)

        # Report preparing to store
        if progress_callback:
            progress_callback("bm25_indexing", 4, 5)

        # Store BM25 index in database
        self.database.store_bm25_index(
            vocabulary=vocabulary,
            inverted_index=filtered_inverted_index,
            document_stats=document_stats,
            collection_stats=collection_stats,
        )

        # Report BM25 indexing complete
        if progress_callback:
            progress_callback("bm25_indexing", 5, 5)

    def _process_document(self, doc: CrawlerResult) -> List[Chunk]:
        """Process a single document into chunks.

        Args:
            doc: Crawler result to process

        Returns:
            List of document chunks

        """
        try:
            return self.processor.process_document(
                path=doc.path,
                content=doc.content,
                title=doc.title,
                language=doc.language,
                metadata=doc.metadata,
            )
        except Exception as e:
            logging.error(f"Error processing document {doc.path}: {e}")
            return []

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
        language: Optional[str] = None,
        use_reranker: Optional[bool] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List[SearchResult]:
        """Search for documents similar to the query.

        Args:
            query: Search query
            limit: Maximum number of results to return
            mode: Search mode ("vector", "bm25", "hybrid")
            language: Optional language filter
            use_reranker: Whether to use reranker (None = use config setting)
            vector_weight: Weight for vector scores in hybrid search
            bm25_weight: Weight for BM25 scores in hybrid search

        Returns:
            List of search results

        """
        # Determine whether to use reranker
        should_rerank = use_reranker if use_reranker is not None else self.config.use_reranker

        # Initialize reranker if needed but not yet initialized
        if should_rerank and self.reranker is None:
            self.reranker = create_reranker(
                model_name=self.config.reranker_model,
                use_onnx=self.config.reranker_use_onnx,
                device=self.config.reranker_device,
                batch_size=self.config.reranker_batch_size,
                max_length=self.config.reranker_max_length,
            )

        # Adjust initial retrieval limit if using reranker
        initial_limit = limit
        if should_rerank and self.reranker is not None:
            initial_limit = limit * self.config.reranker_top_k_multiplier

        # Perform search based on mode
        if mode == "vector":
            # Vector search
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            db_results = self.database.search(query_embedding, initial_limit, language)
        elif mode == "bm25":
            # BM25 search
            tokenizer = create_tokenizer(
                language="ja" if self.config.use_japanese_tokenizer else "en",
                min_token_length=self.config.bm25_min_token_length,
            )
            query_terms = tokenizer.tokenize(query)
            db_results = self.database.search_bm25(query_terms, initial_limit, language)
        elif mode == "hybrid":
            # Hybrid search - combine vector and BM25
            # Reduce search results to avoid excessive reranking
            hybrid_multiplier = 1.5 if should_rerank else 2  # Less results when reranking

            # Vector search
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            vector_results = self.database.search(query_embedding, int(initial_limit * hybrid_multiplier), language)

            # BM25 search
            tokenizer = create_tokenizer(
                language="ja" if self.config.use_japanese_tokenizer else "en",
                min_token_length=self.config.bm25_min_token_length,
            )
            query_terms = tokenizer.tokenize(query)
            bm25_results = self.database.search_bm25(query_terms, int(initial_limit * hybrid_multiplier), language)

            # Combine results
            db_results = self._combine_search_results(vector_results, bm25_results, vector_weight, bm25_weight, initial_limit)
        else:
            raise ValueError(f"Unknown search mode: {mode}")

        # Convert to SearchResult objects
        search_results = []
        for result in db_results:
            # Convert result items to appropriate types for SearchResult
            chunk_id = str(result["chunk_id"])
            path = str(result["path"])
            title = str(result["title"])
            content = str(result["content"])
            language = str(result["language"])

            # Handle potentially problematic conversions
            # Handle chunk_index conversion
            chunk_index_raw = result["chunk_index"]
            if isinstance(chunk_index_raw, int):
                chunk_index = chunk_index_raw
            else:
                try:
                    chunk_index = int(str(chunk_index_raw))
                except (TypeError, ValueError):
                    chunk_index = 0  # Default value if conversion fails

            # Handle score conversion
            score_raw = result["score"]
            if isinstance(score_raw, (float, int)):
                score = float(score_raw)
            else:
                try:
                    score = float(str(score_raw))
                except (TypeError, ValueError):
                    score = 0.0  # Default value if conversion fails

            # Handle metadata which could be any type
            metadata_raw = result["metadata"]
            if isinstance(metadata_raw, dict):
                metadata = metadata_raw  # Already a dict
            else:
                # Create an empty dict if metadata is not a dict
                metadata = {"original": str(metadata_raw)}

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    path=path,
                    title=title,
                    content=content,
                    chunk_index=chunk_index,
                    language=language,
                    metadata=metadata,
                    score=score,
                )
            )

        # Apply reranking if enabled
        if should_rerank and self.reranker is not None:
            # Rerank the results
            import logging
            import time

            logger = logging.getLogger(__name__)
            rerank_start = time.time()
            logger.info(f"Starting reranking of {len(search_results)} results...")
            search_results = self.reranker.rerank(
                query=query,
                results=search_results,
                top_k=limit,
                threshold=self.config.reranker_threshold,
            )
            rerank_time = time.time() - rerank_start
            logger.info(f"Reranking completed in {rerank_time:.2f}s, returned {len(search_results)} results")
        else:
            # If not reranking, just limit the results
            search_results = search_results[:limit]

        return search_results

    def delete_document(self, path: Union[str, Path]) -> int:
        """Delete a document and its chunks from the index.

        Args:
            path: Path to the document to delete

        Returns:
            Number of chunks deleted

        """
        # Delete from database
        deleted_count = self.database.delete_chunks_by_path(path)

        # Remove from processed files tracking
        path_obj = Path(path)
        if path_obj in self._processed_files:
            self._processed_files.remove(path_obj)

        return deleted_count

    def index_directory(
        self, directory: Union[str, Path], incremental: bool = True, progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> tuple[int, int]:
        """Index documents in a directory using the integrated crawler.

        Args:
            directory: Directory to index
            incremental: Whether to only index new files
            progress_callback: Optional callback for progress updates
                              (stage, current, total)

        Returns:
            Tuple of (number of chunks indexed, number of files processed)

        """
        from oboyu.crawler.config import load_default_config
        from oboyu.crawler.crawler import Crawler

        # Initialize crawler with default configuration
        crawler_config = load_default_config()
        crawler = Crawler(
            depth=crawler_config.depth,
            include_patterns=crawler_config.include_patterns,
            exclude_patterns=crawler_config.exclude_patterns,
            max_file_size=crawler_config.max_file_size,
            follow_symlinks=crawler_config.follow_symlinks,
            japanese_encodings=crawler_config.japanese_encodings,
            max_workers=crawler_config.max_workers,
        )

        # If incremental indexing, initialize crawler's processed files
        if incremental:
            crawler._processed_files = self._processed_files.copy()

        # Report starting crawler
        if progress_callback:
            progress_callback("crawling", 0, 1)

        # Crawl directory - no verbosity
        results = crawler.crawl(Path(directory))

        # Report crawling complete
        if progress_callback:
            progress_callback("crawling", 1, 1)

        # Count the number of files processed
        files_processed = len(results)

        # Index results with progress callback
        chunks_indexed = self.index_documents(results, progress_callback)

        return chunks_indexed, files_processed

    def batch_index(self, batch_size: int = 100) -> None:
        """Index in batches to control memory usage.

        Args:
            batch_size: Number of documents to process in each batch

        """
        pass

    def clear_index(self) -> None:
        """Clear all data from the index.

        This method removes all chunks and embeddings from the database
        while preserving the database schema and structure.
        """
        # Clear the database
        self.database.clear()

        # Clear BM25 indexer
        self.bm25_indexer.clear()

        # Reset processed files tracking
        self._processed_files.clear()

    def _combine_search_results(
        self, vector_results: List[Dict[str, object]], bm25_results: List[Dict[str, object]], vector_weight: float, bm25_weight: float, limit: int
    ) -> List[Dict[str, object]]:
        """Combine vector and BM25 search results using weighted scores.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            vector_weight: Weight for vector scores
            bm25_weight: Weight for BM25 scores
            limit: Maximum number of results to return

        Returns:
            Combined and re-ranked results

        """
        # Normalize weights
        total_weight = vector_weight + bm25_weight
        vector_weight = vector_weight / total_weight
        bm25_weight = bm25_weight / total_weight

        # Create score dictionaries
        vector_scores: Dict[object, float] = {}
        for result in vector_results:
            chunk_id = result["chunk_id"]
            score = result["score"]
            if isinstance(score, (int, float)):
                vector_scores[chunk_id] = float(score)
            else:
                vector_scores[chunk_id] = 0.0

        bm25_scores: Dict[object, float] = {}
        for result in bm25_results:
            chunk_id = result["chunk_id"]
            score = result["score"]
            if isinstance(score, (int, float)):
                bm25_scores[chunk_id] = float(score)
            else:
                bm25_scores[chunk_id] = 0.0

        # Normalize scores using min-max normalization
        if vector_scores:
            min_vec = min(vector_scores.values())
            max_vec = max(vector_scores.values())
            range_vec = max_vec - min_vec
            if range_vec > 0:
                vector_scores = {k: (v - min_vec) / range_vec for k, v in vector_scores.items()}

        if bm25_scores:
            min_bm25 = min(bm25_scores.values())
            max_bm25 = max(bm25_scores.values())
            range_bm25 = max_bm25 - min_bm25
            if range_bm25 > 0:
                bm25_scores = {k: (v - min_bm25) / range_bm25 for k, v in bm25_scores.items()}

        # Combine all unique results
        all_results = {}

        # Add vector results
        for result in vector_results:
            chunk_id = result["chunk_id"]
            all_results[chunk_id] = result.copy()
            all_results[chunk_id]["vector_score"] = vector_scores.get(chunk_id, 0)
            all_results[chunk_id]["bm25_score"] = bm25_scores.get(chunk_id, 0)

        # Add BM25 results not in vector results
        for result in bm25_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in all_results:
                all_results[chunk_id] = result.copy()
                all_results[chunk_id]["vector_score"] = vector_scores.get(chunk_id, 0)
                all_results[chunk_id]["bm25_score"] = bm25_scores.get(chunk_id, 0)

        # Calculate combined scores
        for chunk_id, result in all_results.items():
            vector_score = result.get("vector_score", 0)
            bm25_score = result.get("bm25_score", 0)

            # Ensure scores are numeric
            if isinstance(vector_score, (int, float)):
                vector_score = float(vector_score)
            else:
                vector_score = 0.0

            if isinstance(bm25_score, (int, float)):
                bm25_score = float(bm25_score)
            else:
                bm25_score = 0.0

            result["score"] = vector_score * vector_weight + bm25_score * bm25_weight

        # Sort by combined score and return top results
        def get_score(x: Dict[str, object]) -> float:
            score = x.get("score", 0)
            return float(score) if isinstance(score, (int, float)) else 0.0

        sorted_results = sorted(all_results.values(), key=get_score, reverse=True)
        return sorted_results[:limit]

    def close(self) -> None:
        """Close the indexer and its resources."""
        # Shut down the executor first
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)

        # Then close the database
        if self.database:
            self.database.close()
