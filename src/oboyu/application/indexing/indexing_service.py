"""Indexing application service - coordinates document indexing use cases."""

import logging
from pathlib import Path
from typing import List

from ...domain.entities.chunk import Chunk
from ...domain.services.document_processor import DocumentProcessor
from ...ports.external.filesystem_port import FilesystemPort
from ...ports.repositories.search_repository import SearchRepository
from ...ports.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class IndexingService:
    """Application service for indexing documents."""
    
    def __init__(
        self,
        search_repository: SearchRepository,
        embedding_service: EmbeddingService,
        filesystem_port: FilesystemPort,
        document_processor: DocumentProcessor
    ) -> None:
        """Initialize with dependencies injected."""
        self._search_repository = search_repository
        self._embedding_service = embedding_service
        self._filesystem_port = filesystem_port
        self._document_processor = document_processor
    
    async def index_document(self, document_path: Path) -> None:
        """Index a single document."""
        document = await self._filesystem_port.read_document(document_path)
        
        if not document.should_be_processed():
            logger.info(f"Skipping document {document_path} - not suitable for processing")
            return
        
        chunks = self._document_processor.create_chunks(document)
        meaningful_chunks = [chunk for chunk in chunks if chunk.should_be_indexed()]
        
        if not meaningful_chunks:
            logger.info(f"No meaningful chunks found in document {document_path}")
            return
        
        await self._store_chunks_and_embeddings(meaningful_chunks)
        logger.info(f"Indexed document {document_path} with {len(meaningful_chunks)} chunks")
    
    async def index_documents(self, document_paths: List[Path]) -> None:
        """Index multiple documents."""
        for document_path in document_paths:
            try:
                await self.index_document(document_path)
            except Exception as e:
                logger.error(f"Failed to index document {document_path}: {e}")
    
    async def index_directory(self, directory_path: Path,
                            include_patterns: List[str] = None,
                            exclude_patterns: List[str] = None) -> None:
        """Index all documents in a directory."""
        discovered_files = await self._filesystem_port.discover_files(
            directory_path, include_patterns, exclude_patterns
        )
        
        logger.info(f"Discovered {len(discovered_files)} files in {directory_path}")
        await self.index_documents(discovered_files)
    
    async def reindex_document(self, document_path: Path) -> None:
        """Reindex a document (remove old chunks and create new ones)."""
        await self._search_repository.delete_chunks_by_document(str(document_path))
        await self.index_document(document_path)
    
    async def remove_document(self, document_path: Path) -> None:
        """Remove a document from the index."""
        await self._search_repository.delete_chunks_by_document(str(document_path))
        logger.info(f"Removed document {document_path} from index")
    
    async def _store_chunks_and_embeddings(self, chunks: List[Chunk]) -> None:
        """Store chunks and generate embeddings."""
        await self._search_repository.store_chunks(chunks)
        
        texts_for_embedding = []
        chunk_ids = []
        
        for chunk in chunks:
            embedding_text = chunk.content
            if chunk.prefix_content:
                embedding_text = f"{chunk.prefix_content} {chunk.content}"
            
            texts_for_embedding.append(embedding_text)
            chunk_ids.append(chunk.id)
        
        embeddings = await self._embedding_service.generate_embeddings(texts_for_embedding)
        
        embedding_pairs = list(zip(chunk_ids, embeddings))
        await self._search_repository.store_embeddings(embedding_pairs)
