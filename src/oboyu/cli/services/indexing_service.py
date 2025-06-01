"""Indexing service for handling all indexing-related business logic."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from oboyu.cli.index_config import build_indexer_config, create_indexer_config
from oboyu.common.config import ConfigManager
from oboyu.crawler.config import load_default_config
from oboyu.crawler.crawler import Crawler
from oboyu.crawler.discovery import discover_documents
from oboyu.indexer import Indexer
from oboyu.indexer.config.indexer_config import IndexerConfig


@dataclass
class IndexingResult:
    """Result of an indexing operation."""
    
    total_files: int
    total_chunks: int
    elapsed_time: float
    directories_processed: List[Path]


@dataclass
class StatusResult:
    """Result of a status check operation."""
    
    new_files: int
    modified_files: int
    deleted_files: int
    total_indexed: int
    directory: Path


@dataclass
class DiffResult:
    """Result of a diff operation."""
    
    new_files: List[str]
    modified_files: List[str]
    deleted_files: List[str]
    directory: Path


class IndexingService:
    """Service for handling indexing operations."""
    
    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize the indexing service.
        
        Args:
            config_manager: Configuration manager instance

        """
        self.config_manager = config_manager
    
    def create_indexer_config(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        db_path: Optional[Path] = None,
    ) -> IndexerConfig:
        """Create indexer configuration.
        
        Args:
            chunk_size: Optional chunk size override
            chunk_overlap: Optional chunk overlap override
            embedding_model: Optional embedding model override
            db_path: Optional database path override
            
        Returns:
            IndexerConfig instance

        """
        indexer_config_dict = create_indexer_config(
            self.config_manager,
            chunk_size,
            chunk_overlap,
            embedding_model,
            db_path,
        )
        return build_indexer_config(indexer_config_dict)
    
    def execute_indexing(
        self,
        directories: List[Path],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        db_path: Optional[Path] = None,
        change_detection: Optional[str] = None,
        cleanup_deleted: Optional[bool] = None,
        verify_integrity: bool = False,
        force: bool = False,
        max_depth: Optional[int] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> IndexingResult:
        """Execute indexing operation.
        
        Args:
            directories: List of directories to index
            chunk_size: Optional chunk size override
            chunk_overlap: Optional chunk overlap override
            embedding_model: Optional embedding model override
            db_path: Optional database path override
            change_detection: Change detection strategy
            cleanup_deleted: Whether to cleanup deleted files
            verify_integrity: Whether to verify file integrity
            force: Whether to force reindexing
            max_depth: Maximum directory depth
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            progress_callback: Optional progress callback
            
        Returns:
            IndexingResult with operation results

        """
        indexer_config = self.create_indexer_config(
            chunk_size, chunk_overlap, embedding_model, db_path
        )
        
        start_time = time.time()
        total_chunks = 0
        total_files = 0
        
        indexer = Indexer(config=indexer_config)
        
        try:
            for directory in directories:
                if progress_callback:
                    progress_callback(f"Scanning directory {directory}...")
                
                # Note: detection_strategy and should_cleanup would be used in actual change detection
                # For now, using basic crawler without change detection integration
                
                crawler = Crawler(
                    depth=max_depth or 10,
                    include_patterns=include_patterns or ["*.txt", "*.md", "*.html", "*.py", "*.java"],
                    exclude_patterns=exclude_patterns or ["*/node_modules/*", "*/venv/*"],
                    max_workers=4,
                    respect_gitignore=True,
                )
                
                crawler_results = crawler.crawl(directory)
                result = indexer.index_documents(crawler_results)
                
                chunks_indexed = result.get("indexed_chunks", 0)
                files_processed = result.get("total_documents", 0)
                
                total_chunks += chunks_indexed
                total_files += files_processed
                
                if progress_callback:
                    progress_callback(f"Indexed {chunks_indexed} chunks from {files_processed} documents")
        finally:
            indexer.close()
        
        elapsed_time = time.time() - start_time
        
        return IndexingResult(
            total_files=total_files,
            total_chunks=total_chunks,
            elapsed_time=elapsed_time,
            directories_processed=directories,
        )
    
    def get_status(
        self,
        directories: List[Path],
        db_path: Optional[Path] = None,
    ) -> List[StatusResult]:
        """Get indexing status for directories.
        
        Args:
            directories: List of directories to check
            db_path: Optional database path override
            
        Returns:
            List of StatusResult for each directory

        """
        indexer_config_dict = create_indexer_config(
            self.config_manager,
            None,  # chunk_size
            None,  # chunk_overlap
            None,  # embedding_model
            db_path,
        )
        
        from oboyu.cli.index_config import build_status_indexer_config
        indexer_config = build_status_indexer_config(indexer_config_dict)
        indexer = Indexer(config=indexer_config)
        
        results = []
        
        try:
            crawler_config = load_default_config()
            
            for directory in directories:
                discovered_paths = discover_documents(
                    Path(directory),
                    patterns=crawler_config.include_patterns,
                    exclude_patterns=crawler_config.exclude_patterns,
                    max_depth=crawler_config.depth,
                )
                
                paths_only = [path for path, metadata in discovered_paths]
                changes = indexer.change_detector.detect_changes(paths_only, strategy="smart")
                stats = indexer.change_detector.get_processing_stats()
                
                results.append(StatusResult(
                    new_files=len(changes.new_files),
                    modified_files=len(changes.modified_files),
                    deleted_files=len(changes.deleted_files),
                    total_indexed=stats.get('completed', 0),
                    directory=directory,
                ))
        finally:
            indexer.close()
        
        return results
    
    def get_diff(
        self,
        directories: List[Path],
        db_path: Optional[Path] = None,
        change_detection: Optional[str] = None,
    ) -> List[DiffResult]:
        """Get diff of what would be updated.
        
        Args:
            directories: List of directories to check
            db_path: Optional database path override
            change_detection: Change detection strategy
            
        Returns:
            List of DiffResult for each directory

        """
        indexer_config_dict = create_indexer_config(
            self.config_manager,
            None,  # chunk_size
            None,  # chunk_overlap
            None,  # embedding_model
            db_path,
        )
        
        from oboyu.cli.index_config import build_status_indexer_config
        indexer_config = build_status_indexer_config(indexer_config_dict)
        indexer = Indexer(config=indexer_config)
        
        results = []
        
        try:
            crawler_config = load_default_config()
            detection_strategy = change_detection or "smart"
            
            for directory in directories:
                discovered_paths = discover_documents(
                    Path(directory),
                    patterns=crawler_config.include_patterns,
                    exclude_patterns=crawler_config.exclude_patterns,
                    max_depth=crawler_config.depth,
                )
                
                paths_only = [path for path, metadata in discovered_paths]
                changes = indexer.change_detector.detect_changes(paths_only, strategy=detection_strategy)
                
                results.append(DiffResult(
                    new_files=[str(f) for f in sorted(changes.new_files)],
                    modified_files=[str(f) for f in sorted(changes.modified_files)],
                    deleted_files=[str(f) for f in sorted(changes.deleted_files)],
                    directory=directory,
                ))
        finally:
            indexer.close()
        
        return results
    
    def clear_index(self, db_path: Optional[Path] = None) -> None:
        """Clear the index database.
        
        Args:
            db_path: Optional database path override

        """
        indexer_config = self.create_indexer_config(db_path=db_path)
        indexer = Indexer(config=indexer_config)
        
        try:
            indexer.clear_index()
        finally:
            indexer.close()
    
    def get_database_path(self, db_path: Optional[Path] = None) -> str:
        """Get the resolved database path.
        
        Args:
            db_path: Optional database path override
            
        Returns:
            Resolved database path as string

        """
        indexer_config_dict = create_indexer_config(
            self.config_manager,
            None,  # chunk_size
            None,  # chunk_overlap
            None,  # embedding_model
            db_path,
        )
        return str(indexer_config_dict["db_path"])
