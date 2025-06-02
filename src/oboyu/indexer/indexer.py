"""Refactored indexer facade for document indexing operations."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from oboyu.crawler.crawler import CrawlerResult
from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.orchestrators.indexing_pipeline import IndexingPipeline
from oboyu.indexer.orchestrators.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)


class Indexer:
    """Main facade that coordinates indexing operations through orchestrators."""

    def __init__(self, config: Optional[IndexerConfig] = None) -> None:
        """Initialize the indexer orchestrator with configuration.

        Args:
            config: Indexer configuration

        """
        # Initialize configuration
        self.config = config or IndexerConfig()
        
        # Initialize service registry with all dependencies
        self.services = ServiceRegistry(self.config)
        
        # Initialize orchestrators
        self.indexing_pipeline = IndexingPipeline(self.services)
        
        # Backward compatibility: expose services directly
        self.database_service = self.services.get_database_service()
        self.embedding_service = self.services.get_embedding_service()
        self.document_processor = self.services.get_document_processor()
        self.bm25_indexer = self.services.get_bm25_indexer()
        self.tokenizer_service = self.services.get_tokenizer_service()
        self.change_detector = self.services.get_change_detector()

    def index_documents(self, crawler_results: List[CrawlerResult], progress_callback: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
        """Index documents using the indexing pipeline.

        Args:
            crawler_results: Results from document crawling
            progress_callback: Optional callback for progress updates

        Returns:
            Indexing result summary

        """
        return self.indexing_pipeline.index_documents(crawler_results, progress_callback)

    def delete_document(self, path: Path) -> int:
        """Delete a document and all its chunks.

        Args:
            path: Path to the document to delete

        Returns:
            Number of chunks deleted

        """
        return self.database_service.delete_chunks_by_path(path)

    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics.

        Returns:
            Dictionary with indexer statistics

        """
        # Ensure config is properly initialized
        assert self.config.model is not None, "ModelConfig should be initialized"
        
        return {
            "total_chunks": self.database_service.get_chunk_count(),
            "indexed_paths": len(self.database_service.get_paths_with_chunks()),
            "embedding_model": self.config.model.embedding_model,
        }

    def clear_index(self) -> None:
        """Clear all indexed data."""
        self.database_service.clear_database()
        if hasattr(self.bm25_indexer, "clear"):
            self.bm25_indexer.clear()

    def close(self) -> None:
        """Close all services and connections."""
        self.services.close()

    @classmethod
    def from_path(cls, db_path: Union[str, Path]) -> "Indexer":
        """Create an indexer from a database path.

        Args:
            db_path: Path to the database file

        Returns:
            Initialized Indexer instance

        """
        from oboyu.indexer.config.indexer_config import IndexerConfig
        
        config = IndexerConfig()
        config.db_path = db_path  # Use the property which handles the assertion
        return cls(config)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics

        """
        return self.database_service.get_database_stats()
