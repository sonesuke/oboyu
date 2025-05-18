"""Main indexer implementation for Oboyu.

This module provides the Indexer class for processing, embedding, and storing
document chunks with specialized handling for Japanese content.
"""

import concurrent.futures
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from oboyu.crawler.crawler import CrawlerResult
from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.database import Database
from oboyu.indexer.embedding import EmbeddingGenerator
from oboyu.indexer.processor import Chunk, DocumentProcessor


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
    """Similarity score (lower is better for cosine distance)."""


class Indexer:
    """Document indexer for processing, embedding, and storing document chunks."""

    def __init__(
        self,
        config: Optional[IndexerConfig] = None,
        processor: Optional[DocumentProcessor] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        database: Optional[Database] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config: Indexer configuration
            processor: Document processor for chunking
            embedding_generator: Generator for document embeddings
            database: Database for storing chunks and embeddings

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

        # Keep track of processed files
        self._processed_files: Set[Path] = set()

    def index_documents(self, crawler_results: List[CrawlerResult]) -> int:
        """Process and index documents from crawler results.

        Args:
            crawler_results: List of crawler results to index

        Returns:
            Number of chunks indexed

        """
        # Skip already processed files
        new_docs = []
        for result in crawler_results:
            if result.path not in self._processed_files:
                new_docs.append(result)
                self._processed_files.add(result.path)

        if not new_docs:
            return 0

        # Process documents into chunks
        all_chunks: List[Chunk] = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit document processing tasks
            future_to_doc = {
                executor.submit(self._process_document, doc): doc
                for doc in new_docs
            }

            # Collect chunks as they complete
            for future in concurrent.futures.as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    chunks = future.result()
                    if chunks:
                        all_chunks.extend(chunks)
                except Exception as e:
                    logging.error(f"Error processing {doc.path}: {e}")

        if not all_chunks:
            return 0

        # Store chunks in database
        self.database.store_chunks(all_chunks)

        # Generate and store embeddings
        embeddings = self.embedding_generator.generate_embeddings(all_chunks)
        self.database.store_embeddings(embeddings, self.config.embedding_model)

        # Recompact index periodically
        if len(all_chunks) > 100:  # Only recompact after significant additions
            self.database.recompact_index()

        return len(all_chunks)

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
        language: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for documents similar to the query.

        Args:
            query: Search query
            limit: Maximum number of results to return
            language: Optional language filter

        Returns:
            List of search results

        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)

        # Search the database
        db_results = self.database.search(query_embedding, limit, language)

        # Convert to SearchResult objects
        search_results = []
        for result in db_results:
            search_results.append(SearchResult(
                chunk_id=result["chunk_id"],
                path=result["path"],
                title=result["title"],
                content=result["content"],
                chunk_index=result["chunk_index"],
                language=result["language"],
                metadata=result["metadata"],
                score=result["score"],
            ))

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

    def index_directory(self, directory: Union[str, Path], incremental: bool = True) -> int:
        """Index documents in a directory using the integrated crawler.

        Args:
            directory: Directory to index
            incremental: Whether to only index new files

        Returns:
            Number of chunks indexed

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

        # Crawl directory
        results = crawler.crawl(Path(directory))

        # Index results
        return self.index_documents(results)

    def batch_index(self, batch_size: int = 100) -> None:
        """Index in batches to control memory usage.

        Args:
            batch_size: Number of documents to process in each batch

        """
        pass

    def close(self) -> None:
        """Close the indexer and its resources."""
        if self.database:
            self.database.close()
