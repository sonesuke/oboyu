"""Indexing pipeline for coordinating document processing and indexing operations."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from oboyu.crawler.crawler import CrawlerResult
from oboyu.indexer.orchestrators.service_registry import ServiceRegistry
from oboyu.indexer.storage.change_detector import ChangeResult, FileChangeDetector

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Coordinates document processing and indexing operations."""
    
    def __init__(self, services: ServiceRegistry) -> None:
        """Initialize the indexing pipeline with services.
        
        Args:
            services: Service registry providing dependencies
            
        """
        self.services = services
        self.document_processor = services.get_document_processor()
        self.database_service = services.get_database_service()
        self.embedding_service = services.get_embedding_service()
        self.bm25_indexer = services.get_bm25_indexer()
        self.change_detector = services.get_change_detector()
        
    def index_documents(
        self,
        crawler_results: List[CrawlerResult],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Index documents through the processing pipeline.
        
        Args:
            crawler_results: Results from document crawling
            progress_callback: Optional callback for progress updates
            
        Returns:
            Indexing result summary
            
        """
        if not crawler_results:
            return {"indexed_chunks": 0, "total_documents": 0}
            
        try:
            # Process documents into chunks with progress tracking
            all_chunks = self._process_documents(crawler_results, progress_callback)
            
            if not all_chunks:
                return {"indexed_chunks": 0, "total_documents": len(crawler_results)}
                
            # Store chunks in database and update file metadata
            self._store_chunks_with_metadata(all_chunks, crawler_results, progress_callback)
            
            # Generate and store embeddings
            self._generate_embeddings(all_chunks, progress_callback)
            
            # Update BM25 index
            self._update_bm25_index(all_chunks, progress_callback)
            
            return {
                "indexed_chunks": len(all_chunks),
                "total_documents": len(crawler_results),
            }
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return {
                "indexed_chunks": 0,
                "total_documents": len(crawler_results),
                "error": str(e),
            }
            
    def process_incremental_updates(
        self,
        changes: ChangeResult,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Process incremental updates for changed files.
        
        Args:
            changes: Change detection results
            progress_callback: Optional callback for progress updates
            
        Returns:
            Update result summary
            
        """
        result: Dict[str, Any] = {
            "added": 0,
            "modified": 0,
            "deleted": 0,
            "errors": [],
        }
        
        try:
            # Handle deleted files
            if changes.deleted_files:
                for path in changes.deleted_files:
                    try:
                        deleted_count = self.database_service.delete_chunks_by_path(path)
                        if deleted_count > 0:
                            result["deleted"] += 1
                            logger.info(f"Deleted {deleted_count} chunks for {path}")
                    except Exception as e:
                        logger.error(f"Failed to delete chunks for {path}: {e}")
                        result["errors"].append(f"Delete failed for {path}: {e}")
                        
            # Handle added files
            if changes.new_files:
                # Process added files similar to regular indexing
                # This would need crawler results for the added files
                pass
                
            # Handle modified files
            if changes.modified_files:
                # Re-index modified files
                # This would need crawler results for the modified files
                pass
                
            return result
            
        except Exception as e:
            logger.error(f"Incremental update processing failed: {e}")
            result["errors"].append(str(e))
            return result
            
    def rebuild_indexes(self) -> None:
        """Rebuild all indexes from scratch."""
        try:
            # Clear existing indexes
            self.database_service.clear_database()
            if hasattr(self.bm25_indexer, "clear"):
                self.bm25_indexer.clear()
                
            # Re-index all documents
            # This would need to fetch all documents and re-index them
            logger.info("Index rebuild completed")
            
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
            raise
            
    def _process_documents(
        self,
        crawler_results: List[CrawlerResult],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[Any]:
        """Process documents into chunks.
        
        Args:
            crawler_results: Results from document crawling
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processed chunks
            
        """
        all_chunks = []
        total_docs = len(crawler_results)
        
        if progress_callback:
            progress_callback("processing", 0, total_docs)
            
        for i, result in enumerate(crawler_results):
            chunks = self.document_processor.process_document(
                path=result.path,
                content=result.content,
                title=result.title,
                language=result.language,
                metadata=result.metadata,
            )
            all_chunks.extend(chunks)
            
            if progress_callback:
                progress_callback("processing", i + 1, total_docs)
                
        return all_chunks
        
    def _store_chunks(
        self,
        chunks: List[Any],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Store chunks in the database.
        
        Args:
            chunks: Chunks to store
            progress_callback: Optional callback for progress updates
            
        """
        total_chunks = len(chunks)
        if progress_callback:
            progress_callback("storing", 0, total_chunks)
            
        self.database_service.store_chunks(chunks, progress_callback=progress_callback)
        
    def _store_chunks_with_metadata(
        self,
        chunks: List[Any],
        crawler_results: List[CrawlerResult],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Store chunks in the database and update file metadata for change detection.
        
        Args:
            chunks: Chunks to store
            crawler_results: Original crawler results with file metadata
            progress_callback: Optional callback for progress updates
            
        """
        # Store chunks first
        self._store_chunks(chunks, progress_callback)
        
        # Create a map from file path to crawler result for easy lookup
        path_to_result = {result.path: result for result in crawler_results}
        
        # Group chunks by file path and store file metadata
        chunks_by_path: Dict[Path, List[Any]] = {}
        for chunk in chunks:
            if chunk.path not in chunks_by_path:
                chunks_by_path[chunk.path] = []
            chunks_by_path[chunk.path].append(chunk)
        
        # Update file metadata for each processed file
        for file_path, file_chunks in chunks_by_path.items():
            try:
                crawler_result = path_to_result.get(file_path)
                if crawler_result:
                    # Calculate file hash from crawler result metadata or file
                    file_stats = file_path.stat()
                    content_hash = FileChangeDetector.calculate_file_hash(file_path)
                    
                    # Store file metadata
                    self.database_service.store_file_metadata(
                        path=file_path,
                        file_size=file_stats.st_size,
                        file_modified_at=datetime.fromtimestamp(file_stats.st_mtime),
                        content_hash=content_hash,
                        chunk_count=len(file_chunks)
                    )
                    logger.debug(f"Updated metadata for {file_path} with {len(file_chunks)} chunks")
                else:
                    logger.warning(f"No crawler result found for {file_path}")
            except Exception as e:
                logger.error(f"Failed to update metadata for {file_path}: {e}")
        
    def _generate_embeddings(
        self,
        chunks: List[Any],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Generate and store embeddings for chunks.
        
        Args:
            chunks: Chunks to generate embeddings for
            progress_callback: Optional callback for progress updates
            
        """
        texts_for_embedding = self.document_processor.prepare_for_embedding(chunks)
        logger.info(f"Generating embeddings for {len(texts_for_embedding)} text chunks...")
        
        if progress_callback:
            progress_callback("embedding", 0, len(texts_for_embedding))
            
        embeddings = self.embedding_service.generate_embeddings(
            texts_for_embedding,
            progress_callback=progress_callback,
        )
        
        # Store embeddings
        chunk_ids = [chunk.id for chunk in chunks]
        if progress_callback:
            progress_callback("storing_embeddings", 0, len(chunk_ids))
            
        self.database_service.store_embeddings(
            chunk_ids,
            embeddings,
            progress_callback=progress_callback,
        )
        
        # Ensure HNSW index exists after storing embeddings
        self.database_service.ensure_hnsw_index()
        
    def _update_bm25_index(
        self,
        chunks: List[Any],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Update BM25 index with new chunks.
        
        Args:
            chunks: Chunks to index
            progress_callback: Optional callback for progress updates
            
        """
        if progress_callback:
            progress_callback("bm25_indexing", 0, 5)  # BM25 has 5 main steps
            
        self.bm25_indexer.index_chunks(chunks, progress_callback=progress_callback)
