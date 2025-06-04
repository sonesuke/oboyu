"""Refactored indexer facade for document indexing operations."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
        return self.database_service.delete_document(path)

    def get_paths_with_chunks(self) -> List[str]:
        """Get all paths that have chunks in the database.

        Returns:
            List of file paths

        """
        return self.database_service.get_paths_with_chunks()

    def check_index_health(self) -> Dict[str, Any]:
        """Check the health of the index.

        Returns:
            Health check results including index statistics and status

        """
        return self.database_service.check_health()

    def clear_index(self) -> None:
        """Clear all data from the index while preserving schema.
        
        This method clears all chunks and embeddings from the database
        but keeps the database structure intact.
        """
        if hasattr(self, 'database_service') and self.database_service:
            self.database_service.clear_database()
    
    def close(self) -> None:
        """Close all resources properly."""
        # Database service handles all connection closures
        if hasattr(self, 'database_service') and self.database_service:
            self.database_service.close()

    def __enter__(self) -> "Indexer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_val: BaseException | None,
                 exc_tb: object) -> None:
        """Context manager exit."""
        self.close()

    # Compatibility methods
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with index statistics

        """
        # Ensure config is properly initialized
        assert self.config.model is not None, "ModelConfig should be initialized"
        
        return {
            "total_chunks": self.database_service.get_chunk_count(),
            "indexed_paths": len(self.database_service.get_paths_with_chunks()),
            "embedding_model": self.config.model.embedding_model,
        }

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics

        """
        return self.database_service.get_database_stats()
